"""RTSP input: Jetson NVDEC hardware decode, with an honest software fallback.

``health.decode`` in the MQTT contract exists because every platform has a
silent-CPU-fallback failure mode. On Jetson it is ``nvv4l2decoder``: the plugin
lives in the L4T multimedia stack rather than in stock GStreamer, so a container
built from a plain base image, a wiped ``GST_PLUGIN_PATH`` or a stale registry
in ``~/.cache/gstreamer-1.0`` all produce the same result -- the obvious
implementation opens the stream with ffmpeg on the CPU and everything keeps
working, slower, forever, on a box bought for its GPU.

Two things prevent that here:

* the pipeline is built element by element. ``ElementFactory.make`` returns
  ``None`` for a missing plugin, which is raised rather than papered over, so
  "NVDEC is not available" cannot be mistaken for "NVDEC is available and slow";
* the active path is reported per frame and, by default (``require_hw: true``),
  a fallback is refused outright rather than reported.

The decoder does NOT scale. ``nvvidconv`` converts NVMM NV12 to system-memory
BGRx at the source resolution and the letterbox is done here in OpenCV, by the
same :func:`fit_geometry` every other platform in this repository calls. That
is a deliberate trade: the VIC could do the resize, but then this platform's
forward transform would live in GStreamer caps while its inverse lived in
Python, and the two would be free to disagree. ``coordinate_space: frame_norm``
is only worth anything if the forward and inverse geometry come from one place.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from .letterbox import LetterboxTransform, fit_geometry, pad_into_canvas

LOG = logging.getLogger("esk.jetson.video")

DECODE_HW = "hw"
DECODE_SW = "sw"


class FrameBundle:
    """One decoded, letterboxed frame plus the geometry needed to invert it.

    ``canvas`` is the padded square model input in RGB; ``transform`` maps its
    pixels back to the original frame. ``decode_ms`` is the blocking appsink
    pull and ``letterbox_ms`` the OpenCV resize/pad, kept apart so the stage
    breakdown can say where the frame budget actually goes.
    """

    __slots__ = ("canvas", "transform", "decode", "decode_ms", "letterbox_ms")

    def __init__(
        self,
        canvas: np.ndarray,
        transform: LetterboxTransform,
        decode: str,
        decode_ms: float = 0.0,
        letterbox_ms: float = 0.0,
    ):
        self.canvas = canvas
        self.transform = transform
        self.decode = decode
        self.decode_ms = decode_ms
        self.letterbox_ms = letterbox_ms

    def active_rgb(self) -> np.ndarray:
        """The real pixels, padding removed -- a genuine frame, just scaled."""
        left, top, width, height = self.transform.active_region
        return self.canvas[top:top + height, left:left + width]


class HardwareDecodeUnavailable(RuntimeError):
    """Raised when the NVDEC path cannot be established and fallback is refused."""


def _letterbox_bgrx(view: np.ndarray, size: int, transform: LetterboxTransform):
    """BGRx (or BGR) source rows -> a padded square uint8 RGB canvas.

    The resize runs on the four-channel image and the alpha byte is dropped
    afterwards, on the already-shrunk frame: converting first would mean a
    full-resolution colour pass for a byte that is about to be thrown away.
    """
    import cv2

    scaled_w, scaled_h, _, _ = fit_geometry(transform.src_w, transform.src_h, size)
    interp = cv2.INTER_AREA if scaled_w <= transform.src_w else cv2.INTER_LINEAR
    scaled = cv2.resize(view, (scaled_w, scaled_h), interpolation=interp)
    if scaled.shape[2] == 4:
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    return pad_into_canvas(rgb, size)


class GStreamerNVDEC:
    """RTSP -> ``nvv4l2decoder`` (NVDEC) -> ``nvvidconv`` (BGRx) -> appsink."""

    decode_path = DECODE_HW

    def __init__(
        self,
        url: str,
        size: int = 640,
        transport: str = "tcp",
        codec: str = "h264",
        latency_ms: int = 100,
        appsink_timeout_ms: int = 2000,
        appsink_queue: int = 3,
    ) -> None:
        self.url = url
        self.size = int(size)
        self.transport = transport
        self.codec = codec.lower()
        self.latency_ms = int(latency_ms)
        self.timeout_ns = int(appsink_timeout_ms) * 1_000_000
        self.appsink_queue = max(1, int(appsink_queue))
        self.pipeline = self.sink = self.bus = self.decoder = None
        self.Gst = self.GstVideo = None
        self.transform: LetterboxTransform | None = None
        self.source_size: tuple[int, int] | None = None
        # Evidence fields, read back off the running pipeline to prove which
        # element actually produced the frame. A health flag can be wrong; the
        # instantiated factory name and the negotiated caps cannot.
        self.decoder_factory: str | None = None
        self.negotiated_caps: str | None = None

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise HardwareDecodeUnavailable(
                f"GStreamer element '{factory}' is not registered. On Jetson this "
                f"usually means the L4T multimedia plugins are not on "
                f"GST_PLUGIN_PATH (/usr/lib/aarch64-linux-gnu/gstreamer-1.0), or a "
                f"stale plugin registry in ~/.cache/gstreamer-1.0 is being reused."
            )
        return element

    def start(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstRtsp", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstRtsp, GstVideo
        except Exception as exc:
            raise HardwareDecodeUnavailable(
                "PyGObject GStreamer bindings are unavailable"
            ) from exc
        if self.codec not in ("h264", "h265"):
            raise ValueError(f"unsupported RTSP codec for NVDEC: {self.codec}")

        Gst.init(None)
        self.Gst, self.GstVideo = Gst, GstVideo

        pipeline = Gst.Pipeline.new("esk-jetson-rtsp")
        source = self._make("rtspsrc", "source")
        depay = self._make(f"rtp{self.codec}depay", "depay")
        parser = self._make(f"{self.codec}parse", "parser")
        decoder = self._make("nvv4l2decoder", "decoder")
        converter = self._make("nvvidconv", "converter")
        capsfilter = self._make("capsfilter", "bgrx-caps")
        sink = self._make("appsink", "sink")

        source.set_property("location", self.url)
        source.set_property("latency", self.latency_ms)
        source.set_property("drop-on-latency", True)
        source.set_property(
            "protocols",
            GstRtsp.RTSPLowerTrans.TCP
            if self.transport == "tcp"
            else GstRtsp.RTSPLowerTrans.UDP,
        )
        # Frames are consumed one at a time and the newest one is the only one
        # that matters for a live rule engine; without this the decoder holds
        # its full capture-plane pool and latency grows until it is dropped.
        decoder.set_property("disable-dpb", True)
        # nvvidconv outputs system memory when the downstream caps do not ask
        # for memory:NVMM, which is exactly what appsink needs.
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRx"))
        sink.set_property("sync", False)
        # Depth 1 would mean read() can only ever get the frame arriving next:
        # finish early and it blocks, finish late and that frame was dropped, so
        # any jitter costs a whole frame period. A short queue absorbs it.
        sink.set_property("max-buffers", self.appsink_queue)
        sink.set_property("drop", True)
        sink.set_property("emit-signals", False)

        for element in (source, depay, parser, decoder, converter, capsfilter, sink):
            pipeline.add(element)
        if not (
            depay.link(parser)
            and parser.link(decoder)
            and decoder.link(converter)
            and converter.link(capsfilter)
            and capsfilter.link(sink)
        ):
            raise HardwareDecodeUnavailable("failed to link the NVDEC GStreamer pipeline")

        def on_parser_event(_pad, probe_info):
            event = probe_info.get_event()
            if event is None or event.type != Gst.EventType.CAPS:
                return Gst.PadProbeReturn.OK
            caps = event.parse_caps()
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            if structure is None:
                return Gst.PadProbeReturn.OK
            ok_w, source_w = structure.get_int("width")
            ok_h, source_h = structure.get_int("height")
            if ok_w and ok_h and source_w > 0 and source_h > 0:
                self.source_size = (source_w, source_h)
                LOG.info("NVDEC negotiated source %dx%d", source_w, source_h)
            return Gst.PadProbeReturn.OK

        parser.get_static_pad("src").add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM, on_parser_event
        )

        expected = self.codec.upper()

        def on_pad_added(_source, pad):
            caps = pad.get_current_caps() or pad.query_caps(None)
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            if (
                structure
                and structure.get_string("media") == "video"
                and structure.get_string("encoding-name") == expected
            ):
                pad.link(depay.get_static_pad("sink"))

        source.connect("pad-added", on_pad_added)

        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise HardwareDecodeUnavailable("NVDEC pipeline refused the PLAYING state")
        self.pipeline, self.sink, self.bus = pipeline, sink, pipeline.get_bus()
        self.decoder = decoder
        self.decoder_factory = decoder.get_factory().get_name()

    def _bus_error(self):
        if self.bus is None:
            return None
        message = self.bus.pop_filtered(
            self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS
        )
        if message is None:
            return None
        if message.type == self.Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            return RuntimeError(f"GStreamer NVDEC error: {error}; {debug or ''}")
        return RuntimeError("GStreamer NVDEC stream reached EOS")

    def read(self) -> FrameBundle | None:
        if self.pipeline is None:
            self.start()
        error = self._bus_error()
        if error:
            self.close()
            raise error

        started = time.perf_counter()
        sample = self.sink.emit("try-pull-sample", self.timeout_ns)
        decode_ms = (time.perf_counter() - started) * 1000.0
        if sample is None:
            error = self._bus_error()
            if error:
                self.close()
                raise error
            # A pull that merely timed out is not fatal. Tearing the pipeline
            # down here means the next read() rebuilds the whole RTSP session,
            # which cannot complete inside one appsink timeout either -- so
            # NVDEC never survives its own startup and every deployment "falls
            # back" to CPU decode for a reason that was never a real failure.
            return None

        sample_caps = sample.get_caps()
        structure = sample_caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        if self.negotiated_caps is None:
            self.negotiated_caps = sample_caps.to_string()
            LOG.info("NVDEC appsink negotiated: %s", self.negotiated_caps)

        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map the GStreamer appsink buffer")
        try:
            info = self.GstVideo.VideoInfo.new_from_caps(sample_caps)
            stride = int(info.stride[0])
            # nvvidconv is capsfiltered to BGRx, so four bytes per pixel. The
            # stride is read rather than assumed: NVMM-backed conversions align
            # rows, and reading a padded buffer as tightly packed shears the
            # image diagonally into something the detector finds nothing in.
            channels = 4 if structure.get_string("format") == "BGRx" else 3
            if stride < width * channels or len(mapped.data) < stride * height:
                raise RuntimeError("invalid BGRx stride from nvvidconv")
            if self.transform is None or (
                self.transform.src_w,
                self.transform.src_h,
            ) != (width, height):
                self.transform = LetterboxTransform.for_source(width, height, self.size)
            # Borrowed view over the mapped buffer with its real stride. The
            # resize reads straight out of it, so the full-resolution frame is
            # never copied into a Python-owned array at all.
            borrowed = np.ndarray(
                (height, width, channels),
                dtype=np.uint8,
                buffer=mapped.data,
                strides=(stride, channels, 1),
            )
            letterbox_started = time.perf_counter()
            canvas = _letterbox_bgrx(borrowed, self.size, self.transform)
            letterbox_ms = (time.perf_counter() - letterbox_started) * 1000.0
        finally:
            buffer.unmap(mapped)
        return FrameBundle(canvas, self.transform, DECODE_HW, decode_ms, letterbox_ms)

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)
        self.pipeline = self.sink = self.bus = self.decoder = None


class OpenCVFFmpeg:
    """CPU fallback: OpenCV/FFmpeg decode, then the same software letterbox."""

    decode_path = DECODE_SW
    decoder_factory = "opencv-ffmpeg"
    negotiated_caps = None

    def __init__(self, url: str, size: int = 640, transport: str = "tcp") -> None:
        self.url = url
        self.size = int(size)
        self.transport = transport
        self.capture = None
        self.transform: LetterboxTransform | None = None
        self.source_size: tuple[int, int] | None = None

    def start(self) -> None:
        import cv2

        if self.url.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.transport}"
            )
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"OpenCV/FFmpeg could not open {self.url}")

    def read(self) -> FrameBundle | None:
        if self.capture is None:
            self.start()
        started = time.perf_counter()
        ok, bgr = self.capture.read()
        decode_ms = (time.perf_counter() - started) * 1000.0
        if not ok or bgr is None:
            self.close()
            return None
        src_h, src_w = bgr.shape[:2]
        if self.transform is None or (self.transform.src_w, self.transform.src_h) != (
            src_w,
            src_h,
        ):
            self.transform = LetterboxTransform.for_source(src_w, src_h, self.size)
            self.source_size = (src_w, src_h)
        letterbox_started = time.perf_counter()
        canvas = _letterbox_bgrx(bgr, self.size, self.transform)
        letterbox_ms = (time.perf_counter() - letterbox_started) * 1000.0
        return FrameBundle(canvas, self.transform, DECODE_SW, decode_ms, letterbox_ms)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def open_source(
    url: str,
    size: int,
    transport: str,
    codec: str,
    require_hw: bool,
    latency_ms: int = 100,
    appsink_timeout_ms: int = 2000,
    appsink_queue: int = 3,
):
    """Open ``url``, preferring NVDEC. Returns the backend; never lies about it."""
    try:
        backend = GStreamerNVDEC(
            url,
            size=size,
            transport=transport,
            codec=codec,
            latency_ms=latency_ms,
            appsink_timeout_ms=appsink_timeout_ms,
            appsink_queue=appsink_queue,
        )
        backend.start()
        return backend
    except HardwareDecodeUnavailable:
        if require_hw:
            raise
        LOG.warning(
            "hardware decode unavailable, falling back to CPU ffmpeg; "
            "health.decode will report sw and fallback_active will be true"
        )
        backend = OpenCVFFmpeg(url, size=size, transport=transport)
        backend.start()
        return backend

"""RTSP input: Rockchip MPP hardware decode, with an honest software fallback.

``health.decode`` in the MQTT contract exists because every platform has a
silent-CPU-fallback failure mode, and on Rockchip it is the normal one: if the
``mppvideodec`` plugin is not visible to GStreamer -- missing from the image,
shadowed by a stale plugin registry, or hidden because ``PYTHONPATH`` was
replaced rather than appended and PyGObject vanished with it -- the obvious
implementation quietly opens the stream with ffmpeg on the CPU and everything
keeps working, slower, forever.

Two things prevent that here:

* the pipeline is built element by element. ``ElementFactory.make`` returns
  ``None`` for a missing plugin, which is raised rather than papered over, so
  "MPP is not installed" cannot be mistaken for "MPP is installed and slow";
* the active path is reported per frame and, by default
  (``require_hw: true``), a fallback is refused outright rather than reported.
  A deployment that would rather degrade than stop sets ``require_hw: false``
  and gets ``decode: sw`` with ``fallback_active: true`` on every message.

The decoder is also the scaler: ``mppvideodec`` is told the aspect-fitted output
size, so RGA does the resize and the CPU only copies rows into the padded
canvas. That copy must respect the negotiated stride -- MPP aligns rows, and
reading the buffer as tightly packed shears the image into diagonal garbage in
which the detector finds nothing.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .letterbox import (
    LetterboxTransform,
    copy_strided_to_canvas,
    fit_geometry,
    pad_into_canvas,
)

LOG = logging.getLogger("esk.rknn.video")

DECODE_HW = "hw"
DECODE_SW = "sw"


class FrameBundle:
    """One decoded frame in the shape the rest of the pipeline needs.

    ``canvas`` is the padded square model input; ``transform`` maps its pixels
    back to the original frame. The original full-resolution image is
    deliberately absent: on the hardware path it never exists in system memory,
    and inventing it for the software path would give the two decoders
    different preview and snapshot behaviour.
    """

    __slots__ = ("canvas", "transform", "decode")

    def __init__(self, canvas: np.ndarray, transform: LetterboxTransform, decode: str):
        self.canvas = canvas
        self.transform = transform
        self.decode = decode

    def active_rgb(self) -> np.ndarray:
        """The real pixels, padding removed -- a genuine frame, just scaled."""
        left, top, width, height = self.transform.active_region
        return self.canvas[top:top + height, left:left + width]


class HardwareDecodeUnavailable(RuntimeError):
    """Raised when the MPP path cannot be established and fallback is refused."""


class GStreamerMPP:
    """RTSP -> ``mppvideodec`` (RGA scale, RGB out) -> appsink."""

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
        self.source_size: tuple[int, int] | None = None
        self.transform: LetterboxTransform | None = None
        # Evidence fields: read back after the first frame to prove which
        # element actually produced it.
        self.decoder_factory: str | None = None
        self.negotiated_caps: str | None = None

    def _make(self, factory: str, name: str):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise HardwareDecodeUnavailable(
                f"GStreamer element '{factory}' is not registered. On Rockchip this "
                f"usually means libgstrockchipmpp.so is not on GST_PLUGIN_PATH, or a "
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
            raise ValueError(f"unsupported RTSP codec for MPP: {self.codec}")

        Gst.init(None)
        self.Gst, self.GstVideo = Gst, GstVideo

        pipeline = Gst.Pipeline.new("esk-rknn-rtsp")
        source = self._make("rtspsrc", "source")
        depay = self._make(f"rtp{self.codec}depay", "depay")
        parser = self._make(f"{self.codec}parse", "parser")
        decoder = self._make("mppvideodec", "decoder")
        capsfilter = self._make("capsfilter", "rgb-caps")
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
        decoder.set_property("fast-mode", True)
        # Filled in by the parser CAPS probe below, before the decoder
        # negotiates, so RGA emits the aspect-fitted size directly.
        decoder.set_property("width", 0)
        decoder.set_property("height", 0)
        decoder.set_property("format", 15)  # GstMppVideoDecFormat RGB
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))
        sink.set_property("sync", False)
        # Depth 1 would mean read() can only ever get the frame arriving next:
        # finish early and it blocks, finish late and that frame was dropped, so
        # any jitter costs a whole frame period. A short queue absorbs it.
        sink.set_property("max-buffers", self.appsink_queue)
        sink.set_property("drop", True)
        sink.set_property("emit-signals", False)

        for element in (source, depay, parser, decoder, capsfilter, sink):
            pipeline.add(element)
        if not (
            depay.link(parser)
            and parser.link(decoder)
            and decoder.link(capsfilter)
            and capsfilter.link(sink)
        ):
            raise HardwareDecodeUnavailable("failed to link the MPP GStreamer pipeline")

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
                scaled_w, scaled_h, _, _ = fit_geometry(source_w, source_h, self.size)
                decoder.set_property("width", scaled_w)
                decoder.set_property("height", scaled_h)
                self.source_size = (source_w, source_h)
                self.transform = LetterboxTransform.for_source(
                    source_w, source_h, self.size
                )
                LOG.info(
                    "MPP negotiated source %dx%d -> RGA %dx%d (letterbox %d)",
                    source_w, source_h, scaled_w, scaled_h, self.size,
                )
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
            raise HardwareDecodeUnavailable("MPP pipeline refused the PLAYING state")
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
            return RuntimeError(f"GStreamer MPP error: {error}; {debug or ''}")
        return RuntimeError("GStreamer MPP stream reached EOS")

    def read(self) -> FrameBundle | None:
        if self.pipeline is None:
            self.start()
        error = self._bus_error()
        if error:
            self.close()
            raise error

        sample = self.sink.emit("try-pull-sample", self.timeout_ns)
        if sample is None:
            error = self._bus_error()
            if error:
                self.close()
                raise error
            # A pull that merely timed out is not fatal. Tearing the pipeline
            # down here means the next read() rebuilds the whole RTSP session,
            # which cannot complete inside one appsink timeout either -- so MPP
            # never survives its own startup and every deployment "falls back"
            # to CPU decode for a reason that was never a real failure.
            return None

        sample_caps = sample.get_caps()
        structure = sample_caps.get_structure(0)
        width, height = int(structure.get_value("width")), int(structure.get_value("height"))
        if self.negotiated_caps is None:
            self.negotiated_caps = sample_caps.to_string()
            LOG.info("MPP appsink negotiated: %s", self.negotiated_caps)

        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map the GStreamer appsink buffer")
        try:
            info = self.GstVideo.VideoInfo.new_from_caps(sample_caps)
            stride = int(info.stride[0])
            if stride < width * 3 or len(mapped.data) < stride * height:
                raise RuntimeError("invalid RGB stride from mppvideodec")

            transform = self.transform
            if transform is None or (self.source_size and
                                     (width, height) != fit_geometry(
                                         *self.source_size, self.size)[:2]):
                # Some streams do not carry coded dimensions in the parser CAPS,
                # so RGA passed the source through unscaled. Derive the geometry
                # from what actually arrived rather than trusting the probe.
                transform = None

            if width <= self.size and height <= self.size:
                if transform is None:
                    transform = _transform_from_scaled(width, height, self.size)
                canvas = copy_strided_to_canvas(
                    mapped.data, width, height, stride, self.size
                )
            else:
                import cv2

                borrowed = np.ndarray(
                    (height, width, 3), dtype=np.uint8,
                    buffer=mapped.data, strides=(stride, 3, 1),
                )
                scaled_w, scaled_h, _, _ = fit_geometry(width, height, self.size)
                scaled = cv2.resize(
                    borrowed, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
                )
                canvas = pad_into_canvas(scaled, self.size)
                transform = LetterboxTransform.for_source(width, height, self.size)
        finally:
            buffer.unmap(mapped)
        return FrameBundle(canvas, transform, DECODE_HW)

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)
        self.pipeline = self.sink = self.bus = self.decoder = None


def _transform_from_scaled(
    scaled_w: int, scaled_h: int, size: int
) -> LetterboxTransform:
    """Recover a transform when only the post-RGA size is known.

    The source dimensions are then unknown, so the transform is expressed
    against the scaled image as if it were the source. That is still correct for
    ``frame_norm`` -- normalized coordinates do not care about absolute source
    pixels -- but it does mean ``frame.w/h`` reports the decoded size.
    """
    return LetterboxTransform(
        scale_x=1.0,
        scale_y=1.0,
        pad_x=float((size - scaled_w) // 2),
        pad_y=float((size - scaled_h) // 2),
        src_w=scaled_w,
        src_h=scaled_h,
        dst=size,
    )


class OpenCVFFmpeg:
    """CPU fallback: OpenCV/FFmpeg decode, then a software letterbox."""

    decode_path = DECODE_SW

    def __init__(self, url: str, size: int = 640, transport: str = "tcp") -> None:
        self.url = url
        self.size = int(size)
        self.transport = transport
        self.capture = None
        self.transform: LetterboxTransform | None = None

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
        import cv2

        if self.capture is None:
            self.start()
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            self.close()
            return None
        src_h, src_w = bgr.shape[:2]
        if self.transform is None or (self.transform.src_w, self.transform.src_h) != (src_w, src_h):
            self.transform = LetterboxTransform.for_source(src_w, src_h, self.size)
        scaled_w, scaled_h, _, _ = fit_geometry(src_w, src_h, self.size)
        interp = cv2.INTER_AREA if scaled_w <= src_w else cv2.INTER_LINEAR
        scaled = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=interp)
        canvas = pad_into_canvas(scaled[:, :, ::-1].copy(), self.size)
        return FrameBundle(canvas, self.transform, DECODE_SW)

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
    """Open ``url``, preferring MPP. Returns the backend; never lies about it.

    With ``require_hw`` (the default) a missing MPP path raises instead of
    silently degrading -- on this board a fallback means the NPU is being fed by
    a CPU that is also decoding, which is precisely the configuration whose
    symptom is "the rules look broken".
    """
    try:
        backend = GStreamerMPP(
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

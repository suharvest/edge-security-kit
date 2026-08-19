#!/usr/bin/env python3
"""
intrusion-detection -- edge-security-kit HUB-MODE detector for reCamera Pro.

This is the third detector platform behind the ``sensecraft.detection/1``
contract (`edge-security-kit/contracts/MQTT.md`), after ``platforms/generic``
(ONNX, x86) and ``platforms/rknn`` (RK3588). Its job is to prove the contract is
platform-independent, so the parts the contract pins down are shared code, not
re-implementations:

* ``esk/letterbox.py`` and ``esk/tracker.py`` are byte-identical copies of the
  RK3588 modules -- the coordinate inverse and the ``track_id >= 1`` semantics
  are exactly what a second implementation would get subtly wrong;
* ``esk/zoo_head.py`` keeps the RK3588 NumPy decode of the
  ``airockchip/rknn_model_zoo`` nine-output YOLOv8 head verbatim, and only
  swaps the runtime handle (kit already loaded the model);
* ``esk/publisher.py`` is the RK3588 MQTT surface lifted out of its capture
  loop.

**Hub mode**: this app publishes detections + status only. It runs no zone,
loitering or line rule; the hub judges those and asks for evidence over
``cmd/snapshot``. That is why the tracker is mandatory here -- the hub keeps
per-track rule state only for ``track_id >= 1`` (HUB_SPEC §2.1), so a detector
without tracking would have to run single-box mode instead.

What is reCamera-specific:

* the frame source is the device's own sensor, taken **before** any encoder:
  the firmware's extension API publishes ISP NV12 dma-buf frames on
  ``/run/recamera/frame.sock``, and the kit's capability registry selects
  ``OfficialFrameSource`` for them automatically. Nothing in this process
  decodes H.264/H.265 -- there is no encode/decode round trip to make hardware-
  accelerated, which is why chasing an MPP decoder was the wrong question. RGA
  does NV12->RGB and the letterbox resize. ``health.decode`` is ``hw``, and
  ``/debug/decode`` serves the ``/proc/self`` evidence behind that word;
* the replay path (``ESK_SOURCE_FILE``/``source_file``) is a *test fixture*: a
  local H.264 clip really is decoded by ffmpeg on the CPU, and reports ``sw``;
* inference is the RV1126B's single-core NPU through ``rknnlite``;
* JPEG work uses Pillow, not OpenCV, which is not on this Buildroot image.

Run on device::

    /userdata/rknnenv/bin/python -m kit.run \\
        /userdata/local/apps/intrusion-detection --sink stdout --quiet

Configuration comes from the manifest's ``config_schema`` (auto-bound onto
``self`` by ``kit.app.App.start``), overridden by ``ESK_*`` environment
variables for headless/CI runs. ``ESK_SOURCE_FILE`` swaps the camera for a local
video file so the shared truth clip can be replayed -- see ``esk/file_source.py``
for why that path exists at all.
"""

from __future__ import annotations

import collections
import os
import signal
import time

import numpy as np

from kit.app import App, run_app

from esk import cpufreq
from esk.letterbox import (
    LetterboxTransform,
    frame_norm_to_pixels,
    pad_into_canvas,
)
from esk.preview import FrameStore, encode_jpeg, start_preview_server
from esk.publisher import Publisher
from esk.tracker import IoUTracker
from esk.zoo_head import PERSON_CLASS_NAME, PersonDetector

APP_VERSION = "0.2.0"


def _env(name: str, default=None):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


class IntrusionDetectionApp(App):
    id = "intrusion-detection"
    name = "Intrusion Detection"
    owns_loop = True
    # The letterbox is done here, not by `self.pre()`: the contract requires the
    # published coordinates to be an EXACT inverse of the forward transform, and
    # `esk/letterbox.py` derives the inverse from the integer geometry actually
    # applied (a separate scale per axis) while the kit's LetterboxInfo carries
    # the ideal `min()` ratio. The two differ by up to half a source pixel,
    # which is small but is precisely the class of error the truth clip exists
    # to catch. Sharing the RK3588 module also means a coordinate bug cannot
    # hide on one platform.
    #
    # ``hw`` asks the kit for BOTH: ``frame.data`` stays original-resolution RGB
    # (the preview, the snapshot reply and the annotated evidence frame all need
    # source pixels) while ``frame.model_data`` is an RGA-produced, already
    # letterboxed model image. The Python letterbox above is kept as the
    # fallback and as the reference the hardware geometry is checked against
    # every frame -- see ``_model_canvas``. On a source that yields no
    # ``model_data`` (the replay fixture, or an RGA that latched off) the app
    # behaves exactly as before.
    model_frame = "hw"

    # config_schema fallbacks (the manifest supplies a default for each).
    confidence = 0.35
    track_iou = 0.2
    track_max_lost = 0.75
    device_id = "recamera-pro-01"
    stream_id = "cam-0"
    mqtt_host = "127.0.0.1"
    mqtt_port = 1883
    mqtt_username = ""
    mqtt_password = ""
    status_interval = 30.0
    preview_port = 8099
    preview_advertise_host = ""
    source_file = ""
    annotate_path = ""
    mem_probe = 0
    malloc_tune = 0
    # Which native path drives the NPU. Not a tuning knob -- it selects between
    # a leaking implementation and a non-leaking one, and the default is the
    # fix. `rknnlite` is kept only so the regression can be re-measured on
    # demand without reinstalling an old package.
    #
    #   rknnlite -> kit.runtime.engine.RknnLiteModel, i.e. RKNNLite.inference
    #               over rknn_runtime.cpython-311-aarch64-linux-gnu.so. Leaks
    #               43.8 kB per inference on this graph (README, "The bare loop
    #               settles it") -- ~2.2 MB/min at 18.8 fps, an OOM in hours.
    #   ctypes   -> kit.runtime.ctypes_rknn.CtypesRknnModel, which performs the
    #               *same* librknnrt sequence the Cython extension does
    #               (rknn_inputs_set + rknn_run + rknn_outputs_get +
    #               rknn_outputs_release, want_float=1) and is flat: 6465
    #               inferences moved [heap] by zero bytes.
    #
    # That pair is the whole argument. Both call the identical vendor entry
    # points with the identical arguments against the identical graph; the only
    # difference is whether the Cython extension is in the call path. So the
    # missing free is in the extension, not in librknnrt, and removing the
    # extension is a fix rather than a workaround -- no periodic context rebuild,
    # no leak budget, nothing to tune.
    #
    # Since that held, the ctypes path moved into `kit` and became the default
    # for every app on the board; this key now only picks which kit backend to
    # ask for (`ESK_RKNN_BACKEND`). It stays because the fixture scripts drive
    # the A/B through it.
    engine = "ctypes"
    # Diagnostics. Both default to off and neither is a product setting:
    # `cpu_governor` pins the clock so a DVFS-free latency baseline can be taken
    # (and is restored on every exit path -- see esk/cpufreq.GovernorLock), and
    # `bench_infer` replaces the whole pipeline with a bare inference loop to
    # test whether the RSS growth survives without the frame path.
    cpu_governor = ""
    bench_infer = 0
    # appmgr owns the app's environment, so a diagnostic that is only reachable
    # through ESK_* variables is not reachable on the one device it has to run
    # on. These are config_schema keys for that reason, not env-only knobs.
    bench_model = ""
    bench_mode = "infer"
    bench_seconds = 600.0
    bench_tag = "bench"
    # `ctypes_leak` is the positive control and it retains every output buffer
    # on purpose: nine float32 tensors, ~4.9 MB per inference. Unthrottled that
    # is ~120 MB/s against ~1.2 GB of free RAM, which takes the board off the
    # network in about ten seconds -- observed, not hypothetical. The control is
    # only required to make a leak *visible*, so it runs rate-capped and
    # iteration-capped; both default to off for every other mode.
    bench_fps = 0.0
    bench_max_iterations = 0

    # ------------------------------------------------------------- model load

    def _load_model(self, path: str):
        """Select the kit's NPU backend for this run, then let kit build it.

        The ctypes implementation itself now lives in
        ``kit.runtime.ctypes_rknn`` and is the kit-wide default, so this method
        no longer constructs anything -- it only translates this app's
        ``engine`` config key into the kit's ``ESK_RKNN_BACKEND`` switch. The
        key is kept because the fixture scripts drive the A/B through it and
        because re-measuring the rknnlite leak must not require reinstalling an
        older package; the fall back when ctypes cannot initialise is kit's job
        and kit prints it.
        """
        backend = "rknnlite" if str(self.engine).lower() in (
            "rknnlite", "lite", "0", "false") else "ctypes"
        os.environ["ESK_RKNN_BACKEND"] = backend
        model = super()._load_model(path)
        print(f"[intrusion] engine={getattr(model, 'backend', backend)} "
              f"(requested {backend}) for {path}", flush=True)
        return model

    # ------------------------------------------------------------------ setup

    def setup(self, config):
        super().setup(config)

        # Environment overrides. appmgr drives the config_schema; a headless or
        # CI run has no panel, so the same knobs are reachable as ESK_* vars.
        self.device_id = _env("ESK_DEVICE_ID", self.device_id)
        self.stream_id = _env("ESK_STREAM_ID", self.stream_id)
        self.mqtt_host = _env("ESK_MQTT_HOST", self.mqtt_host)
        self.mqtt_port = int(_env("ESK_MQTT_PORT", self.mqtt_port))
        self.mqtt_username = _env("ESK_MQTT_USER", self.mqtt_username)
        self.mqtt_password = _env("ESK_MQTT_PASS", self.mqtt_password)
        self.confidence = float(_env("ESK_CONF", self.confidence))
        self.status_interval = float(_env("ESK_STATUS_INTERVAL", self.status_interval))
        self.preview_port = int(_env("ESK_PREVIEW_PORT", self.preview_port))
        self.preview_advertise_host = _env(
            "ESK_PREVIEW_HOST", self.preview_advertise_host
        )
        self.source_file = _env("ESK_SOURCE_FILE", self.source_file)
        self._annotate_path = _env("ESK_ANNOTATE", self.annotate_path)
        self._bench = str(_env("ESK_BENCH_INFER", self.bench_infer)) not in (
            "0", "", "False",
        )
        self._bench_mode = _env("ESK_BENCH_MODE", self.bench_mode)
        self._bench_model = _env("ESK_BENCH_MODEL", self.bench_model)
        self._bench_seconds = float(_env("ESK_BENCH_SECONDS", self.bench_seconds))
        self._bench_fps = float(_env("ESK_BENCH_FPS", self.bench_fps))
        self._bench_max_iter = int(_env("ESK_BENCH_MAX_ITER", self.bench_max_iterations))
        self._bench_tag = _env("ESK_BENCH_TAG", self.bench_tag)
        self._bench_out = _env(
            "ESK_BENCH_OUT", f"/userdata/esk-fixture/bench-{self._bench_tag}.json"
        )
        self._max_seconds = float(_env("ESK_SECONDS", 0.0))
        self._max_frames = int(_env("ESK_FRAMES", 0))
        # Replay mode drives its own source, so the kit must not ALSO open the
        # camera -- two ffmpeg decoders on a 2 GB board for one consumed stream.
        # start() reads `needs_frames` after setup() returns, which is the only
        # window where a config-driven answer is still in time.
        if self.source_file:
            self.needs_frames = False
        # The bench loop feeds itself a constant array; opening the camera would
        # put the ISP path back into the very process the experiment exists to
        # empty out.
        if self._bench:
            self.needs_frames = False

        # Clock control. Applied before the model is loaded so the first
        # inference is already at the pinned frequency, restored by run()'s
        # finally block and by the atexit hook below -- appmgr's app switch
        # sends SIGTERM, and a governor left on `performance` would outlive this
        # process and quietly burn power on an idle camera.
        self._gov_lock = None
        self.cpu_governor = _env("ESK_CPU_GOVERNOR", self.cpu_governor)
        self.cpufreq_before = cpufreq.snapshot()
        if self.cpu_governor:
            import atexit

            self._gov_lock = cpufreq.GovernorLock(self.cpu_governor)
            applied = self._gov_lock.apply()
            atexit.register(self._restore_governor)
            print(f"[intrusion] cpufreq lock: {applied}", flush=True)
        # Sampled unconditionally: a latency number quoted without the clock it
        # was measured at is not comparable with anything.
        self.freq_sampler = cpufreq.FreqSampler(1.0).start()

        # start() resolved the model input side from the manifest before
        # calling setup(); the class attribute is only the fallback.
        self.net_size = self._pre_size or self.input_size
        self.detector = PersonDetector(
            self.models.det,
            conf_threshold=self.confidence,
            iou_threshold=self.iou,
            input_size=self.net_size,
        )
        self.tracker = IoUTracker(float(self.track_iou), float(self.track_max_lost))

        self.frame_id = 0
        # "stopped" until frames flow, not "starting": the contract enum is
        # running/reconnecting/stopped, and a value outside it makes the hub
        # drop the whole status message. That used to leave the registry a
        # generation behind for 30 s, long enough for the next heartbeat to
        # reset a track that was mid-dwell.
        self.stream_state = "stopped"
        # Replay is a CPU decode by construction; the camera path is expected to
        # be the zero-copy ISP broker. `decode_path` is not fixed here -- it is
        # RESOLVED from the source object once frames start flowing
        # (`_resolve_decode_path`), so `health.decode` can never disagree with
        # what /debug/decode reads out of /proc/self.
        self.decode_primary = "sw" if self.source_file else "hw"
        self.decode_path = "unknown"
        self.frame_times = collections.deque(maxlen=60)
        self.inference_times = collections.deque(maxlen=600)
        self.pipeline_times = collections.deque(maxlen=600)
        # Per-stage timings, so "pipeline_ms minus inference_time_ms" stops
        # being attributed by guesswork. Same bound as the series above.
        self.stage_times = {
            k: collections.deque(maxlen=600)
            for k in ("letterbox", "store", "infer", "track", "publish", "emit")
        }
        self.letterbox_hw = 0
        self.letterbox_cpu = 0
        self._geometry_mismatch = None
        self.store = FrameStore()
        self._stop = False
        self._annotated_written = False
        self._file_source = None

        # Preview/live endpoints. Advertised in status only when a reachable
        # host was configured: the device cannot guess which address a browser
        # will use, and a wrong URL is worse for the hub UI than an absent one.
        # Allocator tuning is OFF by default. Pinning glibc's mmap threshold was
        # tried against the RSS growth and MEASURED NOT TO FIX IT (growth
        # continued at 30 MB/min) while costing ~19 ms per frame in extra page
        # faults, so it is kept only as a switch for re-testing the hypothesis
        # on a different firmware, never as a default. See `tune_allocator`.
        alloc = {"applied": False, "reason": "disabled by default"}
        if str(_env("ESK_MALLOC_TUNE", self.malloc_tune)) not in ("0", "", "False"):
            from esk.mem_probe import tune_allocator

            alloc = tune_allocator()
        print(f"[intrusion] allocator: {alloc}", flush=True)

        if str(_env("ESK_MEM_PROBE", self.mem_probe)) not in ("0", "", "False"):
            from esk.mem_probe import start_tracing

            start_tracing()

        self.preview_server = start_preview_server(
            self.store, self.stream_id, "0.0.0.0", self.preview_port,
            debug_source=lambda: self._debug_source(),
            debug_extra=lambda: self._debug_app(),
        )
        preview_url = live_url = ""
        if self.preview_advertise_host:
            base = f"http://{self.preview_advertise_host}:{self.preview_port}"
            preview_url = f"{base}/preview/{self.stream_id}.jpg"
            live_url = f"{base}/live/{self.stream_id}"

        self.pub = Publisher(
            device_id=self.device_id,
            stream_id=self.stream_id,
            host=self.mqtt_host,
            port=self.mqtt_port,
            username=self.mqtt_username,
            password=self.mqtt_password,
            status_interval_s=self.status_interval,
            app_version=APP_VERSION,
            model_path=self.models.det.path,
            health_fn=self.health,
            fps_fn=self.measured_fps,
            stream_state_fn=lambda: self.stream_state,
            frame_store=self.store,
            preview_url=preview_url,
            live_url=live_url,
        )

        print(
            f"[intrusion] device={self.device_id} stream={self.stream_id} "
            f"mqtt={self.mqtt_host}:{self.mqtt_port} conf={self.confidence} "
            f"iou={self.iou} track(iou={self.track_iou} lost={self.track_max_lost}s) "
            f"preview=:{self.preview_port} backend={self.detector.backend} "
            f"session={self.pub.session_id}",
            flush=True,
        )

    def on_params_changed(self, changed):
        """Mirror the live-tunable thresholds into the objects that hold copies.

        The tracker is mutated in place rather than rebuilt: it owns one entry
        per live identity, and a fresh instance would renumber every track,
        which the hub reads as every person in the scene vanishing and a new set
        appearing.
        """
        if "confidence" in changed:
            self.detector.conf_threshold = float(self.confidence)
        if "iou" in changed:
            self.detector.iou_threshold = float(self.iou)
        if "track_iou" in changed:
            self.tracker.threshold = float(self.track_iou)
        if "track_max_lost" in changed:
            self.tracker.max_lost_sec = float(self.track_max_lost)

    # ----------------------------------------------------------------- health

    def measured_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return 0.0 if span <= 0 else (len(self.frame_times) - 1) / span

    def health(self) -> dict:
        return {
            "fps": round(self.measured_fps(), 2),
            "decode": self.decode_path,
            "backend": self.detector.backend,
            # There is no CPU inference path -- the app refuses to start without
            # the NPU, because rknnlite is the only way it has to run the model.
            # Decode is the only field that can move: the camera path expects
            # the zero-copy ISP broker (`hw`), and falling back to an ffmpeg
            # RTSP decode -- which is what happens if the extension API is not
            # running -- flips this true.
            "fallback_active": self.decode_path != self.decode_primary,
        }

    def _resolve_decode_path(self, source) -> str:
        """Name the live path from the SOURCE OBJECT, never from a constant.

        ``OfficialFrameSource`` hands over ISP dma-buf frames: no compressed
        bitstream is involved, so no CPU decode happens and ``hw`` is the honest
        answer to "is the frame path accelerated". Anything ffmpeg-backed is
        ``sw``. An unrecognised source is reported as ``unknown`` rather than
        guessed into one of the two contract values.
        """
        name = type(source).__name__
        if name == "OfficialFrameSource":
            return "hw"
        if name in ("FfmpegRtspSource", "SnapshotSource", "FfmpegFileSource"):
            return "sw"
        return "unknown"

    def _debug_source(self):
        """The live source object, for /debug/decode."""
        if self._file_source is not None:
            return self._file_source
        rt = self._rt
        return None if rt is None else rt.get("src")

    def _debug_app(self) -> dict:
        """App-side counters served alongside the kernel evidence."""
        def pct(values, q):
            if not values:
                return None
            ordered = sorted(values)
            return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))], 3)

        return {
            "decode_path": self.decode_path,
            "decode_primary": self.decode_primary,
            "model_frame_mode": self.model_frame,
            "letterbox_hw_frames": self.letterbox_hw,
            "letterbox_cpu_frames": self.letterbox_cpu,
            "letterbox_geometry_mismatch": self._geometry_mismatch,
            "frames": self.frame_id,
            "fps": round(self.measured_fps(), 2),
            "stage_ms_p50": {k: pct(v, 0.5) for k, v in self.stage_times.items()},
            "stage_ms_p95": {k: pct(v, 0.95) for k, v in self.stage_times.items()},
            "inference_ms_p50": pct(self.inference_times, 0.5),
            "inference_ms_p95": pct(self.inference_times, 0.95),
            "pipeline_ms_p50": pct(self.pipeline_times, 0.5),
            "pipeline_ms_p95": pct(self.pipeline_times, 0.95),
            # Latency without the clock it was measured at is not comparable
            # across runs on a board whose governor moves 594 MHz - 1.608 GHz.
            "cpu_governor_requested": self.cpu_governor,
            "cpufreq_now": cpufreq.snapshot(),
            "cpufreq_samples": self.freq_sampler.report(),
        }

    def _model_canvas(self, frame, tf: LetterboxTransform):
        """The 640x640 model input: RGA's if its geometry matches, else Python's.

        The published coordinates are inverted with ``tf``, which is built from
        ``esk.letterbox.fit_geometry``. A hardware letterbox is only usable if it
        placed the image at exactly the same offsets and size -- otherwise every
        box is off by the difference, silently. The two agree for every geometry
        this platform runs, but the check is per-frame and cheap (four integer
        comparisons), and a mismatch degrades to the verified CPU path instead of
        publishing shifted boxes.
        """
        info = getattr(frame, "model_info", None)
        canvas = getattr(frame, "model_data", None)
        if canvas is not None and info is not None:
            scaled_w = int(round(frame.w * info.scale))
            scaled_h = int(round(frame.h * info.scale))
            if (
                canvas.shape[0] == tf.dst
                and canvas.shape[1] == tf.dst
                and scaled_w == tf.scaled_w
                and scaled_h == tf.scaled_h
                and int(info.pad_w) == int(tf.pad_x)
                and int(info.pad_h) == int(tf.pad_y)
            ):
                self.letterbox_hw += 1
                return canvas
            if self._geometry_mismatch is None:
                self._geometry_mismatch = {
                    "hw": [scaled_w, scaled_h, int(info.pad_w), int(info.pad_h)],
                    "cpu": [tf.scaled_w, tf.scaled_h, int(tf.pad_x), int(tf.pad_y)],
                }
        self.letterbox_cpu += 1
        return self._letterbox(frame.data, tf)

    # --------------------------------------------------------------- pipeline

    def _letterbox(self, rgb: np.ndarray, tf: LetterboxTransform) -> np.ndarray:
        """Forward transform matching ``tf`` exactly.

        ``tf`` is built from :func:`fit_geometry`, so the resize target and the
        pad offsets here come from the same integer arithmetic the inverse will
        use. Producing the canvas from anything else -- PIL's own rounding, a
        second ``min()`` -- is how the two halves drift apart by a pixel.
        """
        if (tf.scaled_w, tf.scaled_h) == (rgb.shape[1], rgb.shape[0]):
            scaled = rgb
        else:
            from PIL import Image

            scaled = np.asarray(
                Image.fromarray(rgb).resize(
                    (tf.scaled_w, tf.scaled_h), Image.BILINEAR
                ),
                dtype=np.uint8,
            )
        return pad_into_canvas(scaled, tf.dst)

    def _input_frames(self):
        """The camera (kit-owned) or a replayed file (ESK_SOURCE_FILE)."""
        path = self.source_file
        if not path:
            yield from self.frames()
            return
        from esk.file_source import FfmpegFileSource

        loop = _env("ESK_SOURCE_LOOP", "1") != "0"
        self._file_source = FfmpegFileSource(path, loop=loop, realtime=True)
        print(
            f"[intrusion] replay source {path} "
            f"{self._file_source.w}x{self._file_source.h} loop={loop}",
            flush=True,
        )
        # The kit's warm-up frame is skipped for the camera path inside
        # frames(); replay gets the same treatment here so the first inference's
        # cold-start latency never lands in the published numbers.
        warmed = False
        with self._file_source as source:
            for frame in source.frames():
                if self._stop:
                    break
                if not warmed:
                    warmed = True
                    tf = LetterboxTransform.for_source(
                        frame.w, frame.h, self.net_size
                    )
                    try:
                        self.detector.detect(self._letterbox(frame.data, tf), tf)
                    except Exception as exc:
                        print(f"[intrusion] warm-up skipped ({exc})", flush=True)
                    self.inference_times.clear()
                    continue
                self._cur_frame = frame
                self.tick()
                yield frame

    def _restore_governor(self) -> None:
        """Put the governor back. Idempotent -- three exit paths call it."""
        if self._gov_lock is None:
            return
        result = self._gov_lock.restore()
        if result.get("restored"):
            print(f"[intrusion] cpufreq restore: {result}", flush=True)

    def _run_equiv(self):
        """`bench_mode="equiv"`: prove the two engines return the same numbers.

        Shares the bench entry point because it needs the same thing bench does
        and nothing else does -- to be an appmgr app, so that ``/dev/rknpu`` is
        openable. It is not a benchmark: it runs a fixed, small number of
        inferences and writes a diff, not a curve.
        """
        from esk.equiv_check import run_equiv

        model_path = self._bench_model or self.models.det.path
        video = self.source_file
        out = _env("ESK_EQUIV_OUT", "/userdata/esk-fixture/equiv.json")
        print(f"[intrusion] EQUIV model={model_path} video={video or '(none)'} "
              f"out={out}", flush=True)
        try:
            result = run_equiv(
                model_path=model_path,
                video_path=video,
                out_path=out,
                n_frames=int(self._bench_max_iter or 60),
                input_size=self.net_size,
                conf_threshold=float(self.confidence),
                iou_threshold=float(self.iou),
            )
        finally:
            self.freq_sampler.stop()
            self._restore_governor()
            if self.preview_server is not None:
                self.preview_server.shutdown()
        print(f"[intrusion] EQUIV done: {result.get('summary')}", flush=True)

    def _run_bench(self):
        """Bare inference loop instead of the pipeline (diagnostic mode).

        Everything the normal ``run()`` would start -- MQTT, the frame source,
        the tracker -- is skipped, so that a RSS curve measured here can only be
        attributed to the RKNN call itself. The preview server stays up because
        it is the only way to read the process's own ``/debug/memory`` while it
        runs, and it allocates nothing per iteration.
        """
        from esk.bench_infer import run_bench

        self.stream_state = "running"
        if self._bench_mode == "equiv":
            self._run_equiv()
            return
        model = None if self._bench_model else self.models.det
        print(
            f"[intrusion] BENCH mode={self._bench_mode} "
            f"model={self._bench_model or self.models.det.path} "
            f"seconds={self._bench_seconds} fps={self._bench_fps or 'max'} "
            f"out={self._bench_out}",
            flush=True,
        )
        try:
            result = run_bench(
                model=model,
                model_path=self._bench_model,
                input_shape=(1, self.net_size, self.net_size, 3),
                seconds=self._bench_seconds,
                sample_every_s=30.0,
                out_path=self._bench_out,
                mode=self._bench_mode,
                target_fps=self._bench_fps,
                max_iterations=self._bench_max_iter,
                label=f"{self._bench_mode}@{self.cpu_governor or 'unlocked'}",
            )
        finally:
            self.freq_sampler.stop()
            self._restore_governor()
            if self.preview_server is not None:
                self.preview_server.shutdown()
        # The clock histogram has to land in the SAME file as the RSS curve:
        # stdout here is the root-owned app.log the fleet account cannot read,
        # so anything printed and not written is evidence that does not exist.
        result["cpufreq"] = self.freq_sampler.report()
        result["cpufreq_before"] = self.cpufreq_before
        result["cpufreq_after"] = cpufreq.snapshot()
        try:
            import json

            with open(self._bench_out, "w") as fh:
                json.dump(result, fh, indent=1)
            os.chmod(self._bench_out, 0o644)
        except OSError as exc:
            print(f"[intrusion] BENCH result rewrite failed: {exc}", flush=True)
        print(f"[intrusion] BENCH done: {result.get('rss_slope_mb_per_min')} MB/min, "
              f"{result.get('iterations')} iterations, "
              f"infer p50={result.get('infer_ms_p50')}ms, "
              f"freq={result['cpufreq']}", flush=True)

    def run(self):
        if self._bench:
            self._run_bench()
            return
        self.pub.connect()
        self._install_stop_handlers()
        deadline = (
            time.monotonic() + self._max_seconds if self._max_seconds > 0 else float("inf")
        )
        self.stream_state = "running"
        try:
            for frame in self._input_frames():
                if self._stop or time.monotonic() >= deadline:
                    break
                # The clock starts after the read returns. read() blocks until
                # the next frame exists, so timing from before it folds the idle
                # wait for a 5 fps source into pipeline_ms and reports a steady
                # ~200 ms no matter how fast the board is -- which would disable
                # the diagnostic that pipeline_ms is there to provide.
                captured_at = time.monotonic()
                self.frame_times.append(captured_at)
                if self.decode_path == "unknown":
                    self.decode_path = self._resolve_decode_path(
                        self._debug_source()
                    )

                tf = LetterboxTransform.for_source(frame.w, frame.h, self.net_size)
                canvas = self._model_canvas(frame, tf)
                t_lb = time.monotonic()
                self.store.put(frame.data)
                t_store = time.monotonic()

                detections = self.detector.detect(canvas, tf)
                t_infer = time.monotonic()
                self.inference_times.append(self.detector.last_inference_ms)
                tracked = self.tracker.update(detections, time.monotonic())
                t_track = time.monotonic()
                self.stage_times["letterbox"].append((t_lb - captured_at) * 1000.0)
                self.stage_times["store"].append((t_store - t_lb) * 1000.0)
                self.stage_times["infer"].append((t_infer - t_store) * 1000.0)
                self.stage_times["track"].append((t_track - t_infer) * 1000.0)

                items = []
                osd = []
                for track, det in tracked:
                    x1, y1, x2, y2 = det.box  # frame_norm xyxy
                    items.append(
                        {
                            "track_id": track.track_id,
                            "class": PERSON_CLASS_NAME,
                            "score": round(det.score, 4),
                            "bbox": [
                                round((x1 + x2) / 2, 6),
                                round((y1 + y2) / 2, 6),
                                round(x2 - x1, 6),
                                round(y2 - y1, 6),
                            ],
                        }
                    )
                    # The kit sink normalizes by the frame size itself, so the
                    # OSD/WS copy is handed ORIGINAL-frame pixels.
                    osd.append(
                        {
                            "box": [
                                x1 * frame.w, y1 * frame.h, x2 * frame.w, y2 * frame.h
                            ],
                            "cls": 0,
                            "cls_name": PERSON_CLASS_NAME,
                            "score": float(det.score),
                            "track_id": track.track_id,
                        }
                    )

                self.frame_id += 1
                pipeline_ms = (time.monotonic() - captured_at) * 1000.0
                self.pipeline_times.append(pipeline_ms)
                payload = self.pub.detection_payload(
                    frame_id=self.frame_id,
                    src_w=tf.src_w,
                    src_h=tf.src_h,
                    items=items,
                    inference_time_ms=self.detector.last_inference_ms,
                    pipeline_ms=pipeline_ms,
                    health=self.health(),
                )
                self.pub.publish_detections(payload)
                t_pub = time.monotonic()
                # Local fan-out too: the /appcenter overlay and any WS consumer
                # see the same frame the hub does.
                self.emit(events=[], results=osd, ts=frame.pts)
                self.stage_times["publish"].append((t_pub - t_track) * 1000.0)
                self.stage_times["emit"].append((time.monotonic() - t_pub) * 1000.0)

                if self._annotate_path and items and not self._annotated_written:
                    write_annotated(frame.data, payload, self._annotate_path)
                    self._annotated_written = True

                if self.pub.status_due():
                    self.pub.publish_status()

                if self._max_frames and self.pub.published >= self._max_frames:
                    break
        finally:
            self.stream_state = "stopped"
            # A clean exit sends a DISCONNECT, which makes the broker DISCARD
            # the will -- so the goodbye has to be published here or the last
            # retained status stays `online: true` forever and the hub shows a
            # zombie device (MQTT.md, "Status and LWT").
            self.pub.shutdown()
            if self.preview_server is not None:
                self.preview_server.shutdown()
            self.freq_sampler.stop()
            self._restore_governor()
            self._print_summary()

    def _install_stop_handlers(self):
        def handler(_signum, _frame):
            print("[intrusion] stop requested", flush=True)
            self._stop = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _print_summary(self):
        def pct(values, q):
            if not values:
                return 0.0
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]

        print(
            f"[intrusion] published {self.pub.published} detection messages, "
            f"{self.measured_fps():.2f} fps, "
            f"inference p50={pct(self.inference_times, 0.5):.2f}ms "
            f"p95={pct(self.inference_times, 0.95):.2f}ms, "
            f"pipeline p50={pct(self.pipeline_times, 0.5):.2f}ms "
            f"p95={pct(self.pipeline_times, 0.95):.2f}ms, "
            f"snapshots={self.pub.snapshot_count}",
            flush=True,
        )
        stages = " ".join(
            f"{name}={pct(series, 0.5):.2f}" for name, series in self.stage_times.items()
        )
        print(
            f"[intrusion] decode={self.decode_path} (primary {self.decode_primary}) "
            f"letterbox hw={self.letterbox_hw} cpu={self.letterbox_cpu} | "
            f"stage p50 ms: {stages}",
            flush=True,
        )


def write_annotated(rgb: np.ndarray, payload: dict, path: str) -> None:
    """Draw boxes decoded from the PUBLISHED frame_norm values.

    Reconstructing from the payload rather than from the internal pixel boxes is
    what makes the image evidence: a wrong letterbox inverse shows up as a
    squashed or offset rectangle instead of as a plausible-looking number.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    draw = ImageDraw.Draw(img)
    for item in payload["detections"]:
        x1, y1, x2, y2 = frame_norm_to_pixels(item["bbox"], img.width, img.height)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=2)
        draw.text(
            (x1 + 2, max(2, y1 - 12)),
            f"id{item['track_id']} {item['class']} {item['score']:.2f}",
            fill=(0, 220, 0),
        )
    draw.text(
        (8, img.height - 14),
        f"src {payload['frame']['w']}x{payload['frame']['h']} "
        f"decode={payload['health']['decode']} frame_id={payload['frame_id']}",
        fill=(0, 200, 255),
    )
    # Same encoder the preview and snapshot paths use: Pillow cannot write JPEG
    # on this firmware (libjpeg 8 runtime under a Pillow built for 9), so the
    # evidence frame would be the one artefact that never got produced.
    data = encode_jpeg(np.asarray(img, dtype=np.uint8), max_bytes=4 * 1024 * 1024)
    if data is None:
        print(f"[intrusion] annotated frame encode failed for {path}", flush=True)
        return
    with open(path, "wb") as fh:
        fh.write(data)
    print(f"[intrusion] annotated evidence frame written to {path}", flush=True)


if __name__ == "__main__":
    run_app(IntrusionDetectionApp())

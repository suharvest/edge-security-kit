"""Latest-frame store, JPEG size capping and the preview HTTP endpoint.

Same routes and the same contract as the RK3588 and generic platforms:

``/preview.jpg`` (and ``/preview/<stream_id>.jpg``)
    one current frame as JPEG -- what ``status.streams[].preview_url``
    advertises and what the hub's rule canvas fetches as its backdrop;
``/live`` (and ``/``)
    a minimal page reloading that JPEG on a timer -- what
    ``status.streams[].live_url`` advertises;
``/debug/decode``
    objective evidence of which decode path is live, read off the running
    source object rather than off a health field the app filled in itself.

The one port from the RK3588 original is the encoder: that platform reaches for
OpenCV, which is not on this Buildroot image and has no business being added to
it for one ``imencode``. Pillow 9.4 is already installed (the kit's preprocess
uses it) and does the same job.

The served frame is the ORIGINAL decoded frame, not the letterbox canvas: this
platform decodes into system memory anyway, so unlike the MPP path there is no
reason to serve the padded/scaled version.
"""

from __future__ import annotations

import io
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SNAPSHOT_MAX_BYTES = 200 * 1024
LIVE_REFRESH_MS = 1000

LIVE_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{stream_id} · live</title>
<style>
 html,body{{margin:0;height:100%%;background:#111;color:#ddd;
   font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
 header{{padding:.5rem .75rem;display:flex;gap:.75rem;align-items:center}}
 code{{color:#7fd}}
 #wrap{{display:flex;align-items:center;justify-content:center;
   height:calc(100%% - 2.5rem)}}
 img{{max-width:100%%;max-height:100%%;object-fit:contain;background:#000}}
 #err{{color:#f88}}
</style></head>
<body>
<header><b>{stream_id}</b><span>refreshing every {refresh_ms} ms</span>
<code>{jpeg_path}</code><span id="err"></span></header>
<div id="wrap"><img id="f" alt="live frame"></div>
<script>
 var img = document.getElementById('f'), err = document.getElementById('err');
 var busy = false;
 function tick() {{
   if (busy) return;
   busy = true;
   var next = new Image();
   next.onload = function () {{ busy = false; err.textContent = ''; img.src = next.src; }};
   next.onerror = function () {{ busy = false; err.textContent = 'frame unavailable'; }};
   next.src = '{jpeg_path}?t=' + Date.now();
 }}
 tick();
 setInterval(tick, {refresh_ms});
</script>
</body></html>
"""


class FrameStore:
    """Holds the most recent decoded frame (RGB) for preview and snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb = None

    def put(self, rgb: np.ndarray) -> None:
        with self._lock:
            self._rgb = np.array(rgb, dtype=np.uint8, copy=True)

    def get(self):
        with self._lock:
            return None if self._rgb is None else self._rgb.copy()


#: PIL quality -> ffmpeg mjpeg ``-q:v`` (2 best, 31 worst). Only used when
#: Pillow's JPEG encoder is unusable; see :func:`_pillow_jpeg_works`.
_FFMPEG_Q = {85: 3, 70: 6, 55: 9, 40: 13, 25: 20}

_pillow_jpeg: "bool | None" = None


def _pillow_jpeg_works() -> bool:
    """Can this Pillow actually WRITE a JPEG?

    On the reCamera Pro firmware it cannot: the bundled Pillow 9.4.0 was
    compiled against libjpeg 9 while the image ships libjpeg 8, so every save
    dies with ``Wrong JPEG library version: library is 80, caller expects 90``
    -> ``OSError: encoder error -2``. ``PIL.features`` reports "JPEG support ok,
    compiled for 9.0" and is therefore useless as a check -- the only reliable
    probe is to encode something. Decoding, resizing and PNG all work, so this
    is scoped to the encoder alone and evaluated once per process.
    """
    global _pillow_jpeg
    if _pillow_jpeg is None:
        try:
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, format="JPEG", quality=80)
            _pillow_jpeg = buf.tell() > 0
        except Exception:
            _pillow_jpeg = False
    return _pillow_jpeg


def _encode_jpeg_ffmpeg(rgb: np.ndarray, quality: int):
    """JPEG-encode RGB through ffmpeg, for images Pillow refuses to write.

    ffmpeg is on the device because the frame source already needs it, and it
    links its own JPEG encoder, so it is unaffected by the libjpeg mismatch.
    """
    h, w = rgb.shape[:2]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
        "-frames:v", "1", "-q:v", str(_FFMPEG_Q.get(quality, 6)),
        "-f", "image2", "-vcodec", "mjpeg", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, input=np.ascontiguousarray(rgb).tobytes(),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20,
        )
    except Exception:
        return None
    return proc.stdout or None


def encode_jpeg(rgb: np.ndarray, max_bytes: int = SNAPSHOT_MAX_BYTES):
    """Encode RGB to JPEG under ``max_bytes``, dropping quality then resolution.

    Returns raw JPEG bytes -- the snapshot topic carries binary, not base64 --
    or None if even the smallest attempt overshoots.
    """
    from PIL import Image

    use_pillow = _pillow_jpeg_works()
    base = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    for downscale in (1.0, 0.75, 0.5, 0.35, 0.25):
        if downscale == 1.0:
            current = base
        else:
            current = base.resize(
                (max(16, int(base.width * downscale)), max(16, int(base.height * downscale))),
                Image.BILINEAR,
            )
        for quality in (85, 70, 55, 40, 25):
            if use_pillow:
                buf = io.BytesIO()
                current.save(buf, format="JPEG", quality=quality)
                if buf.tell() <= max_bytes:
                    return buf.getvalue()
                continue
            data = _encode_jpeg_ffmpeg(np.asarray(current, dtype=np.uint8), quality)
            if data is not None and len(data) <= max_bytes:
                return data
    return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "esk-recamera-preview"
    protocol_version = "HTTP/1.1"

    store: FrameStore
    stream_id: str
    debug_source: object

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, body: bytes, code: int = 200) -> None:
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _live_page(self) -> None:
        jpeg_path = f"/preview/{self.stream_id}.jpg"
        body = LIVE_PAGE.format(
            stream_id=self.stream_id, jpeg_path=jpeg_path, refresh_ms=LIVE_REFRESH_MS
        ).encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        import json

        path = self.path.split("?", 1)[0]
        if path in ("/", "/live", f"/live/{self.stream_id}"):
            self._live_page()
            return
        if path == "/healthz":
            self._json(b'{"ok":true}')
            return
        if path == "/debug/decode":
            source = self.debug_source() if callable(self.debug_source) else self.debug_source
            self._json(
                json.dumps(
                    {
                        "decode": getattr(source, "decode_path", None),
                        "backend_class": type(source).__name__,
                        "decoder_cmd": getattr(source, "decoder_cmd", None),
                        "decoder_report": getattr(source, "decoder_report", None),
                        "source_size": getattr(source, "source_size", None),
                    }
                ).encode("utf-8")
            )
            return
        if path not in ("/preview.jpg", f"/preview/{self.stream_id}.jpg"):
            self.send_error(404, "no such preview")
            return
        frame = self.store.get()
        if frame is None:
            self.send_error(503, "no frame decoded yet")
            return
        body = encode_jpeg(frame)
        if body is None:
            self.send_error(500, "jpeg encode failed")
            return
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # silence per-request logging
        return


def start_preview_server(
    store: FrameStore, stream_id: str, bind: str, port: int, debug_source=None
) -> ThreadingHTTPServer:
    handler = type(
        "PreviewHandler",
        (_Handler,),
        {
            "store": store,
            "stream_id": stream_id,
            # staticmethod, not the bare callable: a plain function in a class
            # dict becomes a bound method on attribute access, so the accessor
            # would be handed the request handler as its first argument.
            "debug_source": staticmethod(debug_source)
            if callable(debug_source)
            else debug_source,
        },
    )
    server = ThreadingHTTPServer((bind, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="preview").start()
    return server

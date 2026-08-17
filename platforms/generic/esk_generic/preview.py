"""Latest-frame store, JPEG size capping and the preview HTTP endpoint.

Two routes matter to consumers:

``/preview.jpg`` (and ``/preview/<stream_id>.jpg``)
    One current frame as JPEG. This is what ``status.streams[].preview_url``
    advertises, and what the hub's single-frame proxy fetches.
``/live`` (and ``/``)
    A minimal HTML page that reloads that JPEG on a timer. This is what
    ``status.streams[].live_url`` advertises, so the workbench "live view" button
    has somewhere to open. It is deliberately a refreshing still, not a video
    stream: no extra encode path, and it works in any browser.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
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
    """Holds the most recent decoded frame for preview and snapshot replies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def encode_jpeg(
    frame: np.ndarray, max_bytes: int = SNAPSHOT_MAX_BYTES
) -> bytes | None:
    """Encode to JPEG under ``max_bytes``, dropping quality then resolution.

    Returns raw JPEG bytes (the snapshot topic carries binary, not base64) or
    None if even the smallest attempt overshoots.
    """
    current = frame
    for downscale in (1.0, 0.75, 0.5, 0.35, 0.25):
        if downscale != 1.0:
            current = cv2.resize(
                frame,
                (
                    max(16, int(frame.shape[1] * downscale)),
                    max(16, int(frame.shape[0] * downscale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        for quality in (85, 70, 55, 40, 25):
            ok, buf = cv2.imencode(
                ".jpg", current, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if ok and buf.nbytes <= max_bytes:
                return buf.tobytes()
    return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "esk-generic-preview"
    protocol_version = "HTTP/1.1"

    # Injected by start_preview_server.
    store: FrameStore
    stream_id: str

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

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
        path = self.path.split("?", 1)[0]
        if path in ("/", "/live", f"/live/{self.stream_id}"):
            self._live_page()
            return
        if path in ("/healthz",):
            body = b'{"ok":true}'
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
    store: FrameStore, stream_id: str, bind: str, port: int
) -> ThreadingHTTPServer:
    handler = type("PreviewHandler", (_Handler,), {"store": store, "stream_id": stream_id})
    server = ThreadingHTTPServer((bind, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="preview").start()
    return server

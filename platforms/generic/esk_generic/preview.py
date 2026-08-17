"""Latest-frame store, JPEG size capping and the preview HTTP endpoint."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

SNAPSHOT_MAX_BYTES = 200 * 1024


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

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path in ("/healthz",):
            body = b'{"ok":true}'
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path not in ("/", "/preview.jpg", f"/preview/{self.stream_id}.jpg"):
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

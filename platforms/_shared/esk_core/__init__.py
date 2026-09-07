"""Runtime pieces every platform detector needs, in one copy.

``platforms/*/`` are separate uv projects with separate images on purpose — a
Jetson image must not carry RKNN wheels — but three things in them are about the
MQTT contract rather than about the accelerator, and a per-platform copy of each
is a per-platform place to get it wrong:

* :mod:`esk_core.control` — the ``cmd/control`` decision table;
* :mod:`esk_core.streams` — one capture loop per stream, with the per-stream
  state that must not be shared between cameras;
* :mod:`esk_core.supervisor` — the N-workers-one-process bookkeeping that turns
  those into a status message, a snapshot route and an ack.

Everything platform-specific stays behind :class:`esk_core.streams.StreamBackend`:
opening a source, reading a frame, running the model, and saying which decode
path it got. The base never imports cv2, TensorRT, RKNN or HailoRT.

Import path: the images copy this directory next to the platform package under
``/app``, so ``import esk_core`` resolves with no path manipulation. Outside a
container each platform package bootstraps it in its own ``__init__``; see
``platforms/_shared/README.md`` for why that is a shim and not a dependency.
"""

from .control import ACK_SCHEMA, CommandError, ControlHandler
from .config_streams import persist_streams, stream_configs
from .streams import (
    OPEN_TIMEOUT_S, FatalSourceError, SourceLost, StreamBackend, StreamConfig,
    StreamWorker, detection_items,
)
from .supervisor import StreamSupervisor

__all__ = [
    "ACK_SCHEMA", "CommandError", "ControlHandler",
    "OPEN_TIMEOUT_S", "FatalSourceError", "SourceLost", "StreamBackend",
    "StreamConfig", "StreamWorker", "detection_items",
    "StreamSupervisor", "persist_streams", "stream_configs",
]

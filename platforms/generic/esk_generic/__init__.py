"""Generic CPU/ONNX detector for edge-security-kit."""

# esk_core lives in platforms/_shared/ and is copied next to this package inside
# the image (WORKDIR /app), where a plain `import esk_core` resolves. Outside a
# container -- `uv run` from this directory, pytest -- it is one directory up and
# across, so put it on the path here rather than in every module that needs it.
# It is a path shim and not a uv dependency on purpose: making it one would mean
# regenerating four lockfiles and copying a second pyproject into four builder
# stages, to share three files that carry no third-party imports of their own.
import os as _os
import sys as _sys

_shared = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "_shared",
)
if _os.path.isdir(_os.path.join(_shared, "esk_core")) and _shared not in _sys.path:
    _sys.path.insert(0, _shared)

__all__ = ["config", "detector", "letterbox", "preview", "tracker", "yolo"]

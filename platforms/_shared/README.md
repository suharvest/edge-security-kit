# Shared runtime base (`esk_core`)

`platforms/*/` stay separate uv projects with separate images on purpose — a
Jetson image must not carry RKNN wheels. But three things in them are about the
MQTT contract rather than about the accelerator, and a per-platform copy of each
is a per-platform place to get it wrong:

| Module | Holds |
|---|---|
| `control.py` | the `cmd/control` decision table: which commands are honoured, what the ack says, and that a redelivered `request_id` applies once |
| `streams.py` | one capture loop per stream, and the per-stream state that must not be shared between cameras |
| `supervisor.py` | N-workers-one-process bookkeeping: add/remove a stream, retune one, route a snapshot, report them all in the status message |
| `config_streams.py` | the two config forms folded into one, and the atomic write-back |

Everything platform-specific stays behind `StreamBackend`: opening a source,
reading a frame, running the model, and saying which decode path it got. The
base imports no cv2, TensorRT, RKNN or HailoRT.

## Why per-stream state is per-stream

`track_id` is per-stream by contract, so one tracker across two cameras hands
the same id to two different people. The confidence threshold is what an
operator retunes per camera, so a single value on the model session moves every
camera when one slider moves. Both are invisible in a one-camera run, which is
why `tests/test_esk_core.py` asserts them directly.

Inference *is* shared, under a lock. A second accelerator context multiplies
memory and, on the Rockchip boards, NPU core placement.

## Why it is a path shim and not a dependency

The images copy this directory next to the platform package under `/app`, where
a plain `import esk_core` resolves with no path manipulation. Outside a
container each platform package puts it on `sys.path` in its own `__init__`.

Making it a real uv dependency would mean regenerating four lockfiles and
copying a second `pyproject.toml` into four builder stages, to share four
modules that have no third-party imports of their own. If `esk_core` ever grows
a dependency, that trade flips and it should become a workspace package.

## What is not migrated

`platforms/generic/` still carries its own `streams.py` and `control.py`. It was
the first implementation and the one these modules were extracted from; the
three accelerated platforms now share this copy, and folding generic in is a
follow-up rather than something this change did.

## Tests

```bash
cd platforms/rknn && uv run --with pyyaml pytest ../_shared/tests tests -q
```

Each platform's `tests/test_control_wiring.py` checks only the wiring — that the
detector subscribes to `cmd/control`, is a `StreamSupervisor`, has a backend
with the methods the capture loop calls, and takes a per-call threshold. A
detector that failed any of those would turn the console's four controls into a
timeout on that board and nothing else would notice.

"""`backend` has to survive the trip from MQTT to /api/devices.

The detector reports which runtime it actually opened -- `hailort-4.21.0-hailo`,
`rknn-2.3.2`, `onnxruntime-1.28.0-cpu` -- inside the status payload's `health`
block. The registry used to drop it while keeping `decode` and
`fallback_active` from the same block, so the one field that answers "is this
board running the runtime I think it is" could only be had by tapping MQTT
directly. That is the field an operator wants after swapping an image.
"""
from __future__ import annotations

from conftest import DEVICE, status


def with_health(payload: dict, **health) -> dict:
    payload["health"] = {"fps": 15.0, "decode": "hw", "fallback_active": False, **health}
    return payload


async def test_backend_reaches_the_device_record(hub):
    await hub.on_status(with_health(status(), backend="hailort-4.21.0-hailo"))
    entry = next(d for d in hub.registry.list_devices() if d["device_id"] == DEVICE)
    assert entry["backend"] == "hailort-4.21.0-hailo"


async def test_backend_survives_a_goodbye_with_no_health_block(hub):
    """An LWT payload carries no health. Blanking the field on the way down
    would lose it exactly when someone is asking why the device went away."""
    await hub.on_status(with_health(status(), backend="hailort-4.21.0-hailo"))
    await hub.on_status(status(online=False))
    entry = next(d for d in hub.registry.list_devices() if d["device_id"] == DEVICE)
    assert entry["online"] is False
    assert entry["backend"] == "hailort-4.21.0-hailo"


async def test_a_changed_backend_is_a_material_change(hub):
    """A runtime swapped underneath a device changes what its numbers mean,
    so it is pushed rather than left for the next poll to notice."""
    await hub.on_status(with_health(status(), backend="hailort-4.21.0-hailo"))
    before = hub.registry._materially_changed(
        {"online": True, "backend": "hailort-4.21.0-hailo", "streams": []},
        {"online": True, "backend": "hailort-4.22.0-hailo", "streams": []},
    )
    assert before is True


async def test_absent_backend_is_none_not_a_crash(hub):
    """Older detectors predate the field; they must still register."""
    await hub.on_status(status())
    entry = next(d for d in hub.registry.list_devices() if d["device_id"] == DEVICE)
    assert entry["backend"] is None

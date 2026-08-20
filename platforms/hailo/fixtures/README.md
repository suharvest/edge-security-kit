# Captured payloads

Taken verbatim off the fixture broker (`127.0.0.1:1884`) during the hardware run
on the reference Raspberry Pi 5 + Hailo-8 on 2026-08-20, per
`contracts/MQTT.md` ("Conformance"). Nothing here is hand-written or copied from
another platform.

| File | Captured from |
|---|---|
| `rpi5-hailo8-detection.json` | live `detections/cam-0`, 1280×720 H.264 @ 5 fps, `decode: sw`, `backend: hailort-4.21.0-hailo` |
| `rpi5-hailo8-status.json` | the retained `status` topic from the same session |
| `rpi5-hailo8-status-lwt.json` | the LWT the broker delivered after `SIGKILL` (no DISCONNECT): `online: false`, no `streams` |
| `rpi5-hailo8-evidence-annotated.jpg` | `--annotate`, boxes reconstructed from the **published** `frame_norm` values |

`tests/test_fixtures_schema.py` validates all three JSON payloads (12 tests,
previously skipped). `contracts/validate_payload.py` passes on each.

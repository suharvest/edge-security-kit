# edge-security hub

Implementation of [`docs/HUB_SPEC.md`](../docs/HUB_SPEC.md) §1-§9.

```bash
uv sync
uv run pytest -q                       # unit tests, no broker needed
uv run python -m edge_hub              # start hub (MQTT_HOST/MQTT_PORT env)
uv run python tools/fake_detector.py   # synthetic detections for end-to-end self-test
```

Layout:

| Path | HUB_SPEC section |
|---|---|
| `edge_hub/mqtt_ingest.py` | §1 ingest, schema validation, snapshot magic-byte/size check |
| `edge_hub/device_registry.py` | §1 status retain + LWT |
| `edge_hub/rules/` | §2 geometry / zone / line / state / engine |
| `edge_hub/alert_manager.py` | §3 cooldown, state machine, snapshot lifecycle |
| `edge_hub/storage.py` | §6 SQLite DDL + §4 atomic config write |
| `edge_hub/http_api.py` | §4 REST, §5 WS, §7 auth, static hosting |

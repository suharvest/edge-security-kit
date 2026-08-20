# Captured payloads

Empty on purpose. `contracts/MQTT.md` ("Conformance") wants a representative
payload per message kind, taken verbatim off the broker during a real run — not
hand-written and not copied from another platform. No such run has happened yet
on this platform; see the top-level `README.md` section "Not yet verified on
hardware" for what is blocking it.

`tests/test_fixtures_schema.py` skips while this directory holds no JSON and
activates as soon as it does.

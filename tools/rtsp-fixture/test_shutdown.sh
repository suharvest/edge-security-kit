#!/usr/bin/env bash
# Prove the graceful-shutdown status fix: a detector that exits cleanly must
# leave a retained status with online:false, and the hub must show it offline.
# Uses its own device_id and preview port so the two running detectors are
# untouched.
set -uo pipefail
ESK_ROOT=${ESK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FIX=${ESK_FIXTURE_DIR:-$HOME/edge-security-fixture}
GEN=${ESK_GENERIC_DIR:-$ESK_ROOT/platforms/generic}
HUB=${ESK_HUB_DIR:-$ESK_ROOT/hub}
DATA=${ESK_HUB_DATA:-$HOME/edge-security-hub-data}
DEV=generic-shutdown-test
MOSQ="docker exec edge-sec-mosquitto"

cat > "$FIX/detector-shutdown-test.yaml" <<EOF
device_id: $DEV
stream_id: cam-9
source: rtsp://127.0.0.1:8555/edge-sec-truth
rtsp_transport: tcp
decode_primary: sw
model: $GEN/models/yolov8n.onnx
conf_threshold: 0.35
iou_threshold: 0.45
intra_threads: 2
mqtt_host: 127.0.0.1
mqtt_port: 1884
status_interval_s: 5
preview_enabled: false
preview_port: 8097
EOF

echo "===== unit tests for platforms/generic after the fix ====="
cd "$GEN" && uv run pytest -q 2>&1 | tail -3

echo
echo "===== running $DEV for 20 s, then letting it exit cleanly ====="
cd "$GEN"
uv run python -m esk_generic --config "$FIX/detector-shutdown-test.yaml" --seconds 20 \
  > "$FIX/e2e-logs/shutdown-test.log" 2>&1
echo "exit=$?"
tail -3 "$FIX/e2e-logs/shutdown-test.log"

sleep 3
echo
echo "===== retained status on sensecraft/security/$DEV/status ====="
timeout 10 $MOSQ mosquitto_sub -h 127.0.0.1 -p 1884 -C 1 \
  -t "sensecraft/security/$DEV/status" 2>&1

echo
echo "===== GET /api/devices entry for $DEV ====="
curl -sS -b "$FIX/e2e-out/cookies.txt" http://127.0.0.1:18080/api/devices \
  | python3 -c "
import json,sys
for d in json.load(sys.stdin)['devices']:
    if d['device_id'] in ('$DEV','generic-truth','generic-adl'):
        print(json.dumps({k: d[k] for k in ('device_id','online','last_seen_ms','streams')}))
"

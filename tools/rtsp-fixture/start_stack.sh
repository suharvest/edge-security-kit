#!/usr/bin/env bash
# Bring up hub + two generic detectors on the test host. Idempotent-ish: kills only the
# processes this script starts (matched on their own data dir / config paths).
set -uo pipefail

# Roots are overridable; defaults match the acceptance run's layout.
ESK_ROOT=${ESK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FIX=${ESK_FIXTURE_DIR:-$HOME/edge-security-fixture}
GEN=${ESK_GENERIC_DIR:-$ESK_ROOT/platforms/generic}
HUB=${ESK_HUB_DIR:-$ESK_ROOT/hub}
# The built frontend. /nonexistent here means the hub serves only its placeholder
# page, so a stack restarted per the README came up with no UI at all.
WEB=${ESK_HUB_WEB_DIR:-$ESK_ROOT/web/dist}
DATA=${ESK_HUB_DATA:-$HOME/edge-security-hub-data}
LOGS=$FIX/e2e-logs

mkdir -p "$DATA" "$LOGS"
[ -d "$WEB" ] || echo "WARNING: no frontend at $WEB -- the hub will serve its placeholder page" >&2

echo "== stopping any previous run of THIS stack"
pkill -f "edge_hub --data-dir $DATA" && echo "  hub stopped"
pkill -f "esk_generic --config $FIX/detector-" && echo "  detectors stopped"
sleep 2

# ---------------- detector configs ----------------
cat > "$FIX/detector-truth.yaml" <<EOF
device_id: generic-truth
stream_id: cam-0
source: rtsp://127.0.0.1:8555/edge-sec-truth
rtsp_transport: tcp
decode_primary: sw
model: $GEN/models/yolov8n.onnx
conf_threshold: 0.35
iou_threshold: 0.45
track_iou_threshold: 0.2
track_max_lost_s: 0.75
# Bounded ONNX threads. Unset, two detectors oversubscribe the host and the
# frame gap grows past track_max_lost_s -> new track ids -> duplicate zone_enter
# (platforms/generic/README.md).
intra_threads: 6
mqtt_host: 127.0.0.1
mqtt_port: 1884
status_interval_s: 10
preview_enabled: true
preview_bind: 0.0.0.0
preview_port: 8099
preview_advertise_host: 127.0.0.1
EOF

cat > "$FIX/detector-adl.yaml" <<EOF
device_id: generic-adl
stream_id: cam-1
source: rtsp://127.0.0.1:8555/edge-sec-720p
rtsp_transport: tcp
decode_primary: sw
model: $GEN/models/yolov8n.onnx
conf_threshold: 0.35
iou_threshold: 0.45
track_iou_threshold: 0.2
track_max_lost_s: 0.75
# Bounded ONNX threads. Unset, two detectors oversubscribe the host and the
# frame gap grows past track_max_lost_s -> new track ids -> duplicate zone_enter
# (platforms/generic/README.md).
intra_threads: 6
mqtt_host: 127.0.0.1
mqtt_port: 1884
status_interval_s: 10
preview_enabled: true
preview_bind: 0.0.0.0
preview_port: 8098
preview_advertise_host: 127.0.0.1
EOF

# ---------------- hub ----------------
echo "== starting hub on :18080"
cd "$HUB"
MQTT_HOST=127.0.0.1 MQTT_PORT=1884 HUB_HTTP_PORT=18080 \
  HUB_ADMIN_PASSWORD='e2e-truth-run-2026' HUB_WEB_DIR="$WEB" \
  setsid nohup uv run python -m edge_hub --data-dir "$DATA" \
  > "$LOGS/hub.log" 2>&1 < /dev/null &
sleep 8

# ---------------- detectors ----------------
echo "== starting detectors"
cd "$GEN"
setsid nohup uv run python -m esk_generic --config "$FIX/detector-truth.yaml" \
  > "$LOGS/detector-truth.log" 2>&1 < /dev/null &
sleep 2
setsid nohup uv run python -m esk_generic --config "$FIX/detector-adl.yaml" \
  > "$LOGS/detector-adl.log" 2>&1 < /dev/null &
sleep 12

echo "== hub log"
tail -20 "$LOGS/hub.log"
echo "== detector-truth log"
tail -8 "$LOGS/detector-truth.log"
echo "== detector-adl log"
tail -8 "$LOGS/detector-adl.log"
echo "== health"
curl -sS http://127.0.0.1:18080/api/health; echo
echo "== processes"
pgrep -af 'edge_hub|esk_generic' | sed 's/ --/\n    --/g'

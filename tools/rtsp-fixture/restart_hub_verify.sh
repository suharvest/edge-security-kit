#!/usr/bin/env bash
# Restart the hub to pick up the device_registry fix. Doubles as the HUB_SPEC
# §10 restart-persistence check: alerts and dispositions must survive.
set -uo pipefail
ESK_ROOT=${ESK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FIX=${ESK_FIXTURE_DIR:-$HOME/edge-security-fixture}
GEN=${ESK_GENERIC_DIR:-$ESK_ROOT/platforms/generic}
HUB=${ESK_HUB_DIR:-$ESK_ROOT/hub}
DATA=${ESK_HUB_DATA:-$HOME/edge-security-hub-data}
LOGS=$FIX/e2e-logs
C="$FIX/e2e-out/cookies.txt"

echo "===== BEFORE restart ====="
BEFORE=$(curl -sS -b "$C" 'http://127.0.0.1:18080/api/alerts?limit=1')
echo "$BEFORE" | python3 -c "import json,sys; d=json.load(sys.stdin); print('count_header:', d['count'], 'max_id:', d['alerts'][0]['id'] if d['alerts'] else None)"
# ack one alert so a disposition has to survive too
TOPID=$(echo "$BEFORE" | python3 -c "import json,sys; print(json.load(sys.stdin)['alerts'][0]['id'])")
echo "acking alert $TOPID:"
curl -sS -b "$C" -X POST "http://127.0.0.1:18080/api/alerts/$TOPID/ack"; echo

echo
echo "===== restarting hub ====="
pkill -f "edge_hub --data-dir $DATA"
sleep 3
cd "$HUB"
MQTT_HOST=127.0.0.1 MQTT_PORT=1884 HUB_HTTP_PORT=18080 \
  HUB_ADMIN_PASSWORD='e2e-truth-run-2026' HUB_WEB_DIR=/nonexistent \
  setsid nohup uv run python -m edge_hub --data-dir "$DATA" \
  >> "$LOGS/hub.log" 2>&1 < /dev/null &
sleep 12
tail -4 "$LOGS/hub.log"

echo
echo "===== AFTER restart: login again (sessions are in-memory by design) ====="
rm -f "$C"
curl -sS -c "$C" -X POST -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"e2e-truth-run-2026"}' \
  http://127.0.0.1:18080/api/auth/login; echo

echo
echo "===== alerts survived? ====="
curl -sS -b "$C" 'http://127.0.0.1:18080/api/alerts?limit=1' | python3 -c "
import json,sys; d=json.load(sys.stdin)
a=d['alerts'][0]
print('count_header:', d['count'], 'max_id:', a['id'], 'state:', a['state'], 'acted_by:', a['acted_by'], 'snapshot_state:', a['snapshot_state'])"
echo "total rows in DB:"
curl -sS -b "$C" 'http://127.0.0.1:18080/api/alerts?limit=500' | python3 -c "
import json,sys; print(len(json.load(sys.stdin)['alerts']))"

echo
echo "===== /api/devices after restart (offline device must not report running/fps) ====="
curl -sS -b "$C" http://127.0.0.1:18080/api/devices | python3 -m json.tool

echo
echo "===== /api/health ====="
curl -sS http://127.0.0.1:18080/api/health; echo

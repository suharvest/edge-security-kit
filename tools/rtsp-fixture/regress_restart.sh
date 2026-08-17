#!/usr/bin/env bash
# Regression test for the event_id-collision bug: restart the hub while the
# detectors keep running (unchanged session_id) and confirm alerting resumes.
set -uo pipefail
ESK_ROOT=${ESK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FIX=${ESK_FIXTURE_DIR:-$HOME/edge-security-fixture}
GEN=${ESK_GENERIC_DIR:-$ESK_ROOT/platforms/generic}
HUB=${ESK_HUB_DIR:-$ESK_ROOT/hub}
DATA=${ESK_HUB_DATA:-$HOME/edge-security-hub-data}
LOGS=$FIX/e2e-logs
C="$FIX/e2e-out/cookies.txt"

echo "===== detector sessions (must be unchanged across the hub restart) ====="
grep -h "mqtt connected" "$LOGS/detector-truth.log" "$LOGS/detector-adl.log" | tail -2
curl -sS -b "$C" http://127.0.0.1:18080/api/devices | python3 -c "
import json,sys
for d in json.load(sys.stdin)['devices']:
    print(' ', d['device_id'], 'session_id=', d['session_id'])"

echo
echo "===== highest stored event_id per stream BEFORE restart ====="
curl -sS -b "$C" 'http://127.0.0.1:18080/api/alerts?limit=500' | python3 -c "
import json,sys
best={}
for a in json.load(sys.stdin)['alerts']:
    k=(a['device_id'],a['stream_id'])
    if k not in best: best[k]=a['event_id']
for k,v in best.items(): print(' ', k, '->', v)"

echo
echo "===== restarting hub ====="
pkill -f "edge_hub --data-dir $DATA"; sleep 3
cd "$HUB"
MQTT_HOST=127.0.0.1 MQTT_PORT=1884 HUB_HTTP_PORT=18080 \
  HUB_ADMIN_PASSWORD='e2e-truth-run-2026' HUB_WEB_DIR=/nonexistent \
  setsid nohup uv run python -m edge_hub --data-dir "$DATA" \
  >> "$LOGS/hub-restart2.log" 2>&1 < /dev/null &
sleep 12
rm -f "$C"
curl -sS -c "$C" -X POST -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"e2e-truth-run-2026"}' \
  http://127.0.0.1:18080/api/auth/login >/dev/null

echo "waiting 70 s for at least two trajectory cycles..."
sleep 70

echo
echo "===== UNIQUE-constraint failures after the fix ====="
grep -c "UNIQUE constraint failed" "$LOGS/hub-restart2.log" || echo 0
echo "===== handler_errors / mqtt state ====="
curl -sS http://127.0.0.1:18080/api/health; echo

echo
echo "===== new alerts after the restart ====="
curl -sS -b "$C" 'http://127.0.0.1:18080/api/alerts?limit=500' | python3 -c "
import json,sys
a=json.load(sys.stdin)['alerts']
new=[x for x in a if x['id']>350]
print('alerts with id > 350 (pre-restart high-water):', len(new))
for x in sorted(new, key=lambda y:y['id'])[:8]:
    print(' ', x['id'], x['event_id'], x['event_type'], x['rule_name'],
          x.get('direction'), 'snap=', x['snapshot_state'])"

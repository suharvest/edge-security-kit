#!/usr/bin/env bash
# Republish the truth stream at 5 fps and restart both detectors with a bounded
# ONNX thread pool. Only touches processes matched on comm==ffmpeg plus this
# stack's own config paths, so nothing belonging to the fall-detection fixture
# or another tenant is in scope.
set -uo pipefail
# Roots are overridable; defaults match the acceptance run's layout.
ESK_ROOT=${ESK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FIX=${ESK_FIXTURE_DIR:-$HOME/edge-security-fixture}
GEN=${ESK_GENERIC_DIR:-$ESK_ROOT/platforms/generic}
HUB=${ESK_HUB_DIR:-$ESK_ROOT/hub}
DATA=${ESK_HUB_DATA:-$HOME/edge-security-hub-data}
LOGS=$FIX/e2e-logs
mkdir -p "$LOGS"

echo "== killing ONLY the edge-sec-truth ffmpeg publisher"
for p in $(ps -eo pid,comm,args --no-headers | awk '$2=="ffmpeg" && /edge-sec-truth/ {print $1}'); do
    echo "  kill $p"; kill "$p"
done
sleep 2
echo "  surviving ffmpeg publishers:"
ps -eo pid,comm,args --no-headers | awk '$2=="ffmpeg"' | grep -o 'rtsp://[^ ]*' | sort

echo "== republishing truth.mp4 at 5 fps"
cat > "$FIX/publish_truth.sh" <<'EOF'
#!/usr/bin/env bash
# Same encode profile as publish.sh, but at the source's own 5 fps. publish.sh
# forces fps=15, which would duplicate frames back up to a rate the detector
# cannot consume.
set -euo pipefail
exec ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$1" -an \
    -vf "fps=5" \
    -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline \
    -level 3.1 -b:v 2000k -maxrate 2000k -bufsize 2000k \
    -g 10 -keyint_min 10 -bf 0 \
    -x264-params 'repeat-headers=1:scenecut=0' -pix_fmt yuv420p \
    -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:8555/edge-sec-truth"
EOF
chmod +x "$FIX/publish_truth.sh"
cd "$FIX"
setsid nohup ./publish_truth.sh truth.mp4 > publish-truth.log 2>&1 < /dev/null &
sleep 6
cat publish-truth.log

echo "== bounding ONNX threads in both detector configs"
for f in detector-truth.yaml detector-adl.yaml; do
    grep -q '^intra_threads:' "$FIX/$f" || echo 'intra_threads: 6' >> "$FIX/$f"
    grep -q '^track_max_lost_s:' "$FIX/$f" || echo 'track_max_lost_s: 0.75' >> "$FIX/$f"
done
grep -E 'intra_threads|track_max_lost' "$FIX"/detector-*.yaml

echo "== restarting detectors"
pkill -f "esk_generic --config $FIX/detector-"
sleep 3
cd "$GEN"
setsid nohup uv run python -m esk_generic --config "$FIX/detector-truth.yaml" \
  > "$LOGS/detector-truth.log" 2>&1 < /dev/null &
sleep 2
setsid nohup uv run python -m esk_generic --config "$FIX/detector-adl.yaml" \
  > "$LOGS/detector-adl.log" 2>&1 < /dev/null &
sleep 25

echo "== ffprobe truth (5 fps expected)"
timeout 25 ffprobe -hide_banner -rtsp_transport tcp -i rtsp://127.0.0.1:8555/edge-sec-truth 2>&1 | tail -4
echo "== ffprobe adl"
timeout 25 ffprobe -hide_banner -rtsp_transport tcp -i rtsp://127.0.0.1:8555/edge-sec-720p 2>&1 | tail -4
echo "== detector logs"
tail -4 "$LOGS/detector-truth.log"; echo ---; tail -4 "$LOGS/detector-adl.log"
echo "== devices"
curl -sS -b "$FIX/e2e-out/cookies.txt" http://127.0.0.1:18080/api/devices | python3 -m json.tool | grep -E 'device_id|fps|state|decode'

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

#!/usr/bin/env bash
# Publish a looping H.264 test stream into the fixture's mediamtx.
#
# The default profile is 1280x720 on purpose: a non-square source is what
# exposes a missing or wrong letterbox inverse in the detector, which a square
# fixture would hide.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 VIDEO [PATH=edge-sec-720p] [RTSP_HOST=127.0.0.1] [SIZE=1280x720]" >&2
    exit 2
fi

video=$1
path=${2:-edge-sec-720p}
host=${3:-127.0.0.1}
size=${4:-1280x720}
width=${size%x*}
height=${size#*x}

exec ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$video" \
    -an \
    -vf "fps=15,scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:-1:-1:color=black" \
    -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline \
    -level 3.1 -b:v 2000k -maxrate 2000k -bufsize 2000k \
    -g 30 -keyint_min 30 -bf 0 \
    -x264-params 'repeat-headers=1:scenecut=0' -pix_fmt yuv420p \
    -f rtsp -rtsp_transport tcp "rtsp://${host}:8555/${path}"

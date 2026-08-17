#!/bin/sh
# Bring up an RTSP source for the RK detector, on the board itself.
#
# Two standalone containers rather than a compose project, and every name and
# port is distinct from anything else that may already be running on the board
# (the fall-detection rig owns mediamtx on 8554 and the default RTMP/HLS/WebRTC
# ports). Nothing here stops, removes or reconfigures a container it did not
# create.
#
# Usage: start_rk_fixture.sh [VIDEO] [FPS]
set -eu

VIDEO=${1:-/home/radxa/edge-security-rknn/fixture/truth.mp4}
FPS=${2:-5}
CONF=${ESK_RK_MEDIAMTX_CONF:-/home/radxa/edge-security-rknn/fixture/mediamtx-rk.yml}
IMAGE=${ESK_RK_MEDIAMTX_IMAGE:-docker.m.daocloud.io/bluenviron/mediamtx:latest-ffmpeg}
SERVER=esk-rk-rtsp-server
PUB=esk-rk-rtsp-pub
PORT=${ESK_RK_RTSP_PORT:-8556}
PATH_NAME=edge-sec-truth

for name in "$SERVER" "$PUB"; do
  if [ -n "$(docker ps -aq -f "name=^${name}$")" ]; then
    echo "removing our own previous $name"
    docker rm -f "$name" >/dev/null
  fi
done

docker run -d --name "$SERVER" --restart unless-stopped --network host \
  -v "$CONF":/mediamtx.yml:ro "$IMAGE" >/dev/null
echo "started $SERVER on rtsp://127.0.0.1:$PORT/$PATH_NAME"

# The clip is already at its target rate; -re paces it to wall clock and
# -stream_loop -1 makes it endless. -c copy keeps the original H.264 bitstream,
# which matters: re-encoding here would hide whether the board can hardware
# decode the stream the fixture actually produces.
docker run -d --name "$PUB" --restart unless-stopped --network host \
  -v "$VIDEO":/video.mp4:ro --entrypoint ffmpeg "$IMAGE" \
  -re -stream_loop -1 -i /video.mp4 -c copy \
  -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:$PORT/$PATH_NAME" >/dev/null
echo "started $PUB publishing $VIDEO at its native rate (~${FPS} fps)"

sleep 3
docker ps --filter "name=esk-rk-rtsp" --format 'table {{.Names}}\t{{.Status}}'

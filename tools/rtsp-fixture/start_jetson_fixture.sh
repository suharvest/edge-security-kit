#!/bin/sh
# Bring up an RTSP source for the Jetson detector, on the board itself.
#
# On-board rather than across the network on purpose: the acceptance run asserts
# event instants to +/- 0.5 s, and a Tailscale hop between the publisher and the
# decoder adds jitter that the assertion cannot tell apart from a detector that
# is late.
#
# Two standalone containers rather than a compose project, and every name and
# port is distinct from the defaults. Nothing here stops, removes or
# reconfigures a container it did not create.
#
# Usage: start_jetson_fixture.sh [VIDEO] [FPS]
set -eu

VIDEO=${1:-/home/harvest/edge-security-jetson/fixture/truth.mp4}
FPS=${2:-5}
CONF=${ESK_JETSON_MEDIAMTX_CONF:-/home/harvest/edge-security-jetson/fixture/mediamtx-jetson.yml}
IMAGE=${ESK_JETSON_MEDIAMTX_IMAGE:-docker.m.daocloud.io/bluenviron/mediamtx:latest-ffmpeg}
SERVER=esk-jetson-rtsp-server
PUB=esk-jetson-rtsp-pub
PORT=${ESK_JETSON_RTSP_PORT:-8556}
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
# decode the stream the fixture actually produces, and NVDEC support is the
# whole point of this platform.
docker run -d --name "$PUB" --restart unless-stopped --network host \
  -v "$VIDEO":/video.mp4:ro --entrypoint ffmpeg "$IMAGE" \
  -re -stream_loop -1 -i /video.mp4 -c copy \
  -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:$PORT/$PATH_NAME" >/dev/null
echo "started $PUB publishing $VIDEO at its native rate (~${FPS} fps)"

sleep 3
docker ps --filter "name=esk-jetson-rtsp" --format 'table {{.Names}}\t{{.Status}}'

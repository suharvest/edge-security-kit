# RK3588 NPU person detector. arm64 only.
#
# Build from the repository root:
#   docker build -f platforms/rknn/Dockerfile \
#     -t <registry>/edge-security-detector-rk3588:0.1.0 .
#
# What this image deliberately does NOT contain
# ---------------------------------------------
# librknnrt.so, rknn_toolkit_lite2 and the GStreamer MPP plugin are all absent.
# Every one of them is a user-space half of a kernel interface: the RKNN user
# library has to match the board's NPU driver, and libgstrockchipmpp.so has to
# match the board's MPP/RGA libraries. Baking a copy in produces an image that
# works on the board it was built against and fails on the next kernel. The
# deploy step stages the board's own copies next to the compose file and mounts
# them:
#   /usr/lib/librknnrt.so   bind-mounted from the board
#   /rknnsite               the board's rknnlite (+ ruamel, psutil), on PYTHONPATH
#   /gstmpp                 the board's libgstrockchipmpp.so, h264parse
#                           (libgstvideoparsersbad.so) and their libs, on
#                           GST_PLUGIN_PATH and LD_LIBRARY_PATH
#
# What that forces about the base image
# -------------------------------------
# * Python must be Debian bookworm's 3.11. The board's rknnlite ships compiled
#   extensions tagged cpython-311-aarch64-linux-gnu; a 3.12 interpreter cannot
#   import them. python:3.11-slim is not equivalent — its /usr/local
#   interpreter would have to load Debian's python3-gi against a second
#   libpython. One interpreter, the distro's, avoids both problems.
# * GStreamer must be bookworm's 1.22, because the staged MPP plugin is linked
#   against exactly that.
# PEP 668 marks the distro interpreter as externally managed, so the wheels go
# into a venv created with --system-site-packages — the same arrangement the
# board itself uses, and the reason python3-gi stays importable.
#
# The weights are baked in at /app/models/. They are gitignored; fetch them
# into the build context first (platforms/rknn/models/README.md) — the build
# asserts the sha256 from platforms/rknn/models/SHA256SUMS, because a model
# converted with the wrong toolkit version loads and then produces garbage.
ARG DEBIAN_TAG=bookworm-slim
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

FROM debian:${DEBIAN_TAG} AS builder
ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY platforms/rknn/pyproject.toml platforms/rknn/uv.lock /src/rknn/
# The dependency set comes from uv.lock rather than being re-resolved, so the
# image matches what the tests ran against. rknn-toolkit-lite2 and PyGObject are
# not in the lock on purpose (see pyproject.toml) and stay board-supplied.
RUN pip install --no-cache-dir --break-system-packages uv \
 && python3 -m venv --system-site-packages /opt/venv \
 && cd /src/rknn \
 && uv export --frozen --no-dev --no-emit-project --format requirements-txt > /tmp/req.txt \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/req.txt \
 && find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name 'tests' -prune -exec rm -rf {} +
# The wheels are deliberately NOT stripped. bookworm's binutils 2.40 rewrites
# the 64 KiB-aligned program headers these aarch64 wheels use and the result
# fails to load: "libscipy_openblas64_-....so: ELF load command address/offset
# not page-aligned".

FROM debian:${DEBIAN_TAG}

LABEL org.opencontainers.image.title="edge-security-detector-rk3588" \
      org.opencontainers.image.description="RK3588 NPU person detector for edge-security-kit: RTSP in over the Rockchip MPP hardware decoder, sensecraft.detection/1 over MQTT out, with a JPEG preview endpoint. The RKNN runtime and the MPP plugin are mounted from the board, not bundled." \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/Seeed-Studio/edge-security-kit" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Seeed Studio"

# python3-gi + the three typelibs esk_rknn asks for (Gst, GstRtsp, GstVideo);
# plugins-base carries appsink/capsfilter, plugins-good carries rtspsrc and
# rtph264depay. h264parse is NOT installed: plugins-bad would drag in a second
# copy of the parser and shadow the board's staged libgstvideoparsersbad.so.
# libgomp1 is the OpenMP runtime the OpenCV wheel links against.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-gi \
      gir1.2-glib-2.0 \
      gir1.2-gstreamer-1.0 \
      gir1.2-gst-plugins-base-1.0 \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# XDG_CACHE_HOME is where GStreamer writes its plugin registry. The default
# would be a root-owned $HOME that a read-only or non-root run cannot write, and
# a registry it fails to write is a registry it rebuilds on every start.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    XDG_CACHE_HOME=/tmp

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY platforms/rknn/esk_rknn/ /app/esk_rknn/
COPY platforms/rknn/models/yolov8n_zoo_int8.rk3588.rknn \
     platforms/rknn/models/yolov8n_zoo_fp16.rk3588.rknn \
     platforms/rknn/models/SHA256SUMS \
     /app/models/

RUN cd /app/models \
 && grep -E 'yolov8n_zoo_(int8|fp16)\.rk3588\.rknn$' SHA256SUMS | sha256sum -c - \
 && rm -f SHA256SUMS

# Sanity-check what the image itself owns. rknnlite and mppvideodec are
# intentionally missing here and are not checked — they arrive from the board.
RUN python3 -c "import gi; gi.require_version('Gst','1.0'); gi.require_version('GstRtsp','1.0'); gi.require_version('GstVideo','1.0'); from gi.repository import Gst, GstRtsp, GstVideo; Gst.init(None); assert Gst.ElementFactory.make('rtspsrc','s'), 'rtspsrc missing'; assert Gst.ElementFactory.make('rtph264depay','d'), 'rtph264depay missing'; assert Gst.ElementFactory.make('appsink','a'), 'appsink missing'; import numpy, cv2, yaml, paho.mqtt.client; print('gst', Gst.version_string(), '| numpy', numpy.__version__, '| cv2', cv2.__version__)"

EXPOSE 8099

# /healthz answers 200 as soon as the preview server is up; it does not assert
# that a frame has decoded, which is deliberate — a camera that is briefly
# unreachable must not make the container restart-loop. Ask /debug/decode for
# which decoder GStreamer actually instantiated.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8099/healthz', timeout=3).status==200 else 1)"

ENTRYPOINT ["python3", "-m", "esk_rknn"]
CMD ["--config", "/app/config/detector.yaml"]

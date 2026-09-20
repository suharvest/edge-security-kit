# Hailo-8 / 8L person detector for Raspberry Pi 5. arm64 only.
#
# Build from the repository root:
#   docker build -f platforms/hailo/Dockerfile \
#     -t <registry>/edge-security-detector-hailo:0.1.0 .
#
# What this image deliberately does NOT contain
# ---------------------------------------------
# libhailort.so and the `hailo_platform` Python package are both absent. Each is
# the user-space half of a kernel interface: HailoRT refuses to open a device
# whose PCIe driver reports a different version, so an image with 4.21.0 baked
# in stops working the moment the board's driver is updated. Both are also
# distributed through Hailo's developer zone under a click-through licence
# rather than a public index, which is a second reason not to redistribute them
# inside an image. The deploy step stages the board's own copies:
#   /hailosite            the board's hailo_platform + hailort, on PYTHONPATH
#   /usr/lib/libhailort.so bind-mounted from the board
#
# What that forces about the base image
# -------------------------------------
# Python must be Debian bookworm's 3.11, because the board's HailoRT wheel ships
# extensions tagged cp311. Same constraint, same reason, as platforms/rknn.
# PEP 668 marks the distro interpreter externally managed, so the wheels go into
# a venv created with --system-site-packages.
#
# Decode is software and that is the primary path, not a fallback
# --------------------------------------------------------------
# The Raspberry Pi 5 has no H.264 decoder — VideoCore VII decodes HEVC only — so
# FFmpeg decodes on the CPU and `health.decode` reports `sw` with
# `fallback_active` false. There is deliberately no require_hw_decode switch and
# no GStreamer in this image: unlike the Rockchip platform there is no hardware
# path that could be silently dropped. Budget CPU accordingly — the measured
# 8.7-13.0% of one core buys decode as well as inference.
#
# The HEF is baked in at /app/models/. It is gitignored; build it into the
# context first (platforms/hailo/models/README.md) — the build asserts its
# sha256, because a HEF compiled by a Dataflow Compiler that does not match the
# runtime loads and then produces wrong numbers rather than failing.
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
COPY platforms/hailo/pyproject.toml platforms/hailo/uv.lock /src/hailo/
# Resolved from uv.lock rather than re-resolved, so the image carries what the
# tests ran against. hailort is not in the lock on purpose — it is board-supplied.
RUN pip install --no-cache-dir --break-system-packages uv \
 && python3 -m venv --system-site-packages /opt/venv \
 && cd /src/hailo \
 && uv export --frozen --no-dev --no-emit-project --format requirements-txt > /tmp/req.txt \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/req.txt \
 && find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name 'tests' -prune -exec rm -rf {} +
# The wheels are deliberately NOT stripped. bookworm's binutils rewrites the
# 64 KiB-aligned program headers these aarch64 wheels use, and the result fails
# to load at import with "ELF load command address/offset not page-aligned".

FROM debian:${DEBIAN_TAG}

# Stamped into the image metadata; override with --build-arg VERSION=<tag> so the
# label always matches the tag actually pushed.
ARG VERSION=0.2.0
LABEL org.opencontainers.image.title="edge-security-detector-hailo" \
      org.opencontainers.image.description="Hailo-8 / 8L person detector for edge-security-kit on Raspberry Pi 5: RTSP in over FFmpeg software decode, sensecraft.detection/1 over MQTT out, with a JPEG preview endpoint. HailoRT and libhailort are mounted from the board, not bundled." \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/suharvest/edge-security-kit" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Seeed Studio"

# libgomp1 is the OpenMP runtime the OpenCV wheel links against. FFmpeg's
# shared libraries come with the opencv-python-headless wheel, so no system
# ffmpeg package is needed for cv2.CAP_FFMPEG to open an RTSP URL.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    XDG_CACHE_HOME=/tmp

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY platforms/hailo/esk_hailo/ /app/esk_hailo/
# The runtime pieces shared by every platform (the cmd/control decision
# table, the per-stream capture loop, the N-workers bookkeeping). Copied
# next to the package rather than installed: it has no third-party imports
# of its own, so a plain `import esk_core` from WORKDIR is the whole
# mechanism. See platforms/_shared/README.md.
COPY platforms/_shared/esk_core/ /app/esk_core/
COPY platforms/hailo/models/yolov8n.hef platforms/hailo/models/SHA256SUMS /app/models/

RUN cd /app/models \
 && grep -E 'yolov8n\.hef$' SHA256SUMS | sha256sum -c - \
 && rm -f SHA256SUMS

# Checks only what the image itself owns. hailo_platform is intentionally
# missing here and is not checked — it arrives from the board at /hailosite.
#
# The numpy major-version assertion is load-bearing. `_pyhailort` is built
# against the numpy 1.x C ABI, and under numpy 2 the array passed to
# `set_buffer()` reads as size 0 on the C++ side: every frame is rejected with
# "Input buffer size 0 is different than expected 1228800" at the first
# inference, with nothing in the message naming numpy. The hailort wheel
# carries `Requires-Dist: numpy<2`, but this image does not install hailort —
# it is board-supplied — so that constraint does not reach the resolver here
# and has to be restated in pyproject.toml. This line fails the build rather
# than shipping an image that deploys cleanly and cannot infer.
RUN python3 -c "import numpy, cv2, yaml, paho.mqtt.client; \
    assert numpy.__version__.split('.')[0] == '1', \
        'numpy %s: _pyhailort needs the 1.x ABI, see pyproject.toml' % numpy.__version__; \
    print('numpy', numpy.__version__, '| cv2', cv2.__version__)"

EXPOSE 8099

# /healthz answers 200 once the preview server is up; it deliberately does not
# assert that a frame has decoded, so a briefly unreachable camera cannot make
# the container restart-loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8099/healthz', timeout=3).status==200 else 1)"

ENTRYPOINT ["python3", "-m", "esk_hailo"]
CMD ["--config", "/app/config/detector.yaml"]

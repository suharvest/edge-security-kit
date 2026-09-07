# Generic CPU/ONNX person detector. arm64 or x86_64, no accelerator SDK.
#
# Build from the repository root:
#   docker build -f platforms/generic/Dockerfile \
#     -t <registry>/edge-security-detector-cpu:0.1.0 .
#
# The weights are baked in at /app/models/yolov8n.onnx, so the deployed compose
# needs no model download and no checksum plumbing. They are gitignored, so
# fetch them into the build context first (platforms/generic/models/README.md):
#   curl -L -o platforms/generic/models/yolov8n.onnx \
#     https://raw.githubusercontent.com/Zhang-zu-hao/Industrial-security-demo/master/models/yolov8n.onnx
# md5 eda19d19a8aa73d54411a6fc1d091c7f — the build asserts it below, because a
# silently different export changes the published coordinates.
ARG PYTHON_TAG=3.12-slim-bookworm
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

FROM python:${PYTHON_TAG} AS builder
ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /src
COPY platforms/generic/pyproject.toml platforms/generic/uv.lock /src/generic/
RUN pip install --no-cache-dir uv \
 && cd /src/generic \
 && uv export --frozen --no-dev --no-emit-project --format requirements-txt > /tmp/req.txt \
 && pip install --no-cache-dir --prefix=/install -r /tmp/req.txt \
 && find /install -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && find /install -type d -name 'tests' -prune -exec rm -rf {} +
# The wheels are deliberately NOT stripped. bookworm's binutils 2.40 rewrites
# the 64 KiB-aligned program headers these aarch64 wheels use and the result
# fails to load: "libscipy_openblas64_-....so: ELF load command address/offset
# not page-aligned". The ~40 MB it would save is not worth an image whose numpy
# only fails at runtime.

FROM python:${PYTHON_TAG}

LABEL org.opencontainers.image.title="edge-security-detector-cpu" \
      org.opencontainers.image.description="CPU/ONNX Runtime person detector for edge-security-kit: RTSP in, sensecraft.detection/1 over MQTT out, with a JPEG preview endpoint" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/suharvest/edge-security-kit" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Seeed Studio"

# libgomp1 is the OpenMP runtime onnxruntime and opencv link against;
# libglib2.0-0 is still a link-time dependency of the headless OpenCV wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /install /usr/local
COPY platforms/generic/esk_generic/ /app/esk_generic/
# The runtime pieces shared by every platform (the cmd/control decision
# table, the per-stream capture loop, the N-workers bookkeeping). Copied
# next to the package rather than installed: it has no third-party imports
# of its own, so a plain `import esk_core` from WORKDIR is the whole
# mechanism. See platforms/_shared/README.md.
COPY platforms/_shared/esk_core/ /app/esk_core/
COPY platforms/generic/models/yolov8n.onnx /app/models/yolov8n.onnx

RUN python -c "import hashlib,sys; h=hashlib.md5(open('/app/models/yolov8n.onnx','rb').read()).hexdigest(); sys.exit(0 if h=='eda19d19a8aa73d54411a6fc1d091c7f' else 'unexpected model md5: '+h)"

# Import everything the detector imports. A wheel that unpacked but cannot load
# its own shared objects is otherwise only discovered when a camera is attached.
RUN python -c "import numpy, cv2, onnxruntime, yaml, paho.mqtt.client; import esk_generic; print('numpy', numpy.__version__, '| cv2', cv2.__version__, '| onnxruntime', onnxruntime.__version__, '| providers', onnxruntime.get_available_providers())"

EXPOSE 8099

# /healthz answers 200 as soon as the preview server is up; it does not assert
# that a frame has decoded, which is deliberate — a camera that is briefly
# unreachable must not make the container restart-loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8099/healthz', timeout=3).status==200 else 1)"

ENTRYPOINT ["python", "-m", "esk_generic"]
CMD ["--config", "/app/config/detector.yaml"]

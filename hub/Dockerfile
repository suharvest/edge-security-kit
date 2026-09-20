# HUB_SPEC §8: pure-CPU arm64 (or x86) single container, no numpy/cv2.
#
# Build from the repository root:
#   docker build -f hub/Dockerfile -t <registry>/edge-security-hub:0.1.0 .
#
# Two stages so the wheel builder, pip's own metadata and uv never reach the
# runtime layer. The dependency set is taken from hub/uv.lock rather than
# re-resolved, so the image matches what the tests ran against.
ARG PYTHON_TAG=3.12-slim-bookworm
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

FROM python:${PYTHON_TAG} AS builder
ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /src
COPY hub/pyproject.toml hub/uv.lock hub/README.md /src/hub/
RUN pip install --no-cache-dir uv \
 && cd /src/hub \
 && uv export --frozen --no-dev --no-emit-project --format requirements-txt > /tmp/req.txt \
 && pip install --no-cache-dir --prefix=/install -r /tmp/req.txt \
 && find /install -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && find /install -type d -name 'tests' -prune -exec rm -rf {} +

FROM python:${PYTHON_TAG}

# Stamped into the image metadata; override with --build-arg VERSION=<tag> so the
# label always matches the tag actually pushed.
ARG VERSION=0.2.0
LABEL org.opencontainers.image.title="edge-security-hub" \
      org.opencontainers.image.description="Aggregation hub for edge-security-kit: MQTT ingest, zone/line/loitering rule engine, alert lifecycle, REST/WS API and the alert workbench" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/suharvest/edge-security-kit" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Seeed Studio"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HUB_DATA_DIR=/data \
    HUB_WEB_DIR=/app/web/dist \
    HUB_CONTRACT_SCHEMA=/app/contracts/mqtt-detection.schema.json \
    PYTHONPATH=/app/hub

WORKDIR /app

COPY --from=builder /install /usr/local

COPY contracts/ /app/contracts/
COPY hub/edge_hub/ /app/hub/edge_hub/
COPY hub/tools/ /app/hub/tools/
# The built frontend is optional: the hub serves a placeholder without it.
COPY web/dist/ /app/web/dist/

RUN python -c "import aiohttp, aiomqtt, jsonschema, bcrypt, paho.mqtt.client; import edge_hub; print('aiohttp', aiohttp.__version__, '| jsonschema', jsonschema.__version__)"

VOLUME ["/data"]
EXPOSE 8090

# No privileged flags, no added capabilities (HUB_SPEC §7).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/api/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["python", "-m", "edge_hub"]

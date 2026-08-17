# HUB_SPEC §8: pure-CPU arm64 (or x86) single container, no numpy/cv2.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HUB_DATA_DIR=/data \
    HUB_WEB_DIR=/app/web/dist \
    HUB_CONTRACT_SCHEMA=/app/contracts/mqtt-detection.schema.json

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies first so a code change does not re-resolve the environment.
COPY hub/pyproject.toml hub/uv.lock /app/hub/
RUN cd /app/hub && uv export --no-dev --no-emit-project --format requirements-txt > /tmp/req.txt \
    && uv pip install --system -r /tmp/req.txt

COPY contracts/ /app/contracts/
COPY hub/edge_hub/ /app/hub/edge_hub/
COPY hub/tools/ /app/hub/tools/
# The built frontend is optional: the hub serves a placeholder without it.
COPY web/dist/ /app/web/dist/

ENV PYTHONPATH=/app/hub
VOLUME ["/data"]
EXPOSE 8090

# No privileged flags, no added capabilities (HUB_SPEC §7).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/api/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["python", "-m", "edge_hub"]

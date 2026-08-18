# Jetson Orin NVDEC + TensorRT person detector. arm64 (L4T) only.
#
# Build from the repository root:
#   docker build -f platforms/jetson/Dockerfile \
#     -t <registry>/edge-security-detector-jetson:0.1.0 .
#
# Why a plain ubuntu:22.04 base and not l4t-cuda
# ----------------------------------------------
# L4T r36.x userspace *is* Ubuntu 22.04 (glibc 2.35, Python 3.10, GStreamer
# 1.20.3), so a stock Ubuntu base is ABI-compatible with everything the board
# supplies and the image needs no CUDA, no TensorRT and no L4T multimedia stack
# of its own. That is the difference between a ~600 MB image and a multi-GB one,
# and it matters on a Jetson whose rootfs is usually nearly full.
#
# What this image deliberately does NOT contain
# ---------------------------------------------
# libnvinfer, libcudart, the TensorRT Python bindings, nvv4l2decoder and
# nvvidconv are all absent. Every one of them is the user-space half of
# something version-locked to the board: the TensorRT Python bindings must match
# the installed libnvinfer, libnvinfer must match the driver, and the L4T
# GStreamer plugins must match the kernel's NVDEC interface. Baking a copy in
# produces an image that works on the JetPack it was built against and fails on
# the next one.
#
# They arrive at runtime instead, and the compose file is where that is wired:
#   runtime: nvidia   the container runtime's CSV mode injects libcuda, the
#                     /usr/lib/aarch64-linux-gnu/nvidia tree (with its
#                     ld.so.conf) and the two L4T GStreamer plugins
#                     libgstnvvideo4linux2.so / libgstnvvidconv.so as individual
#                     file mounts into the container's own paths
#   /host-trt         the board's libnvinfer / libnvinfer_plugin /
#                     libnvonnxparser, bind-mounted per file
#   /host-python      the board's tensorrt Python package, on PYTHONPATH
#   /usr/local/cuda   the board's CUDA runtime, for libcudart.so.12
#
# Host libraries are exposed under PRIVATE paths and reached through
# LD_LIBRARY_PATH / PYTHONPATH. They must never be mounted over the container's
# own /usr/lib/aarch64-linux-gnu or over /usr/lib/python3.10/dist-packages:
# doing that hides the image's own libraries and the first symptom is numpy
# failing to import with "libblas.so.3: cannot open shared object file", which
# takes cv2 down with it.
#
# The engine is not in here either
# --------------------------------
# A serialized TensorRT engine is valid only for the exact TensorRT version and
# GPU architecture that built it; copying one in makes deserialize_cuda_engine
# return None with no other diagnostic. The ONNX is baked in at
# /app/models/yolov8n.onnx so the deploy step can copy it out and build the
# engine on the target with the board's own trtexec (~307 s on an Orin NX), and
# the resulting .engine is bind-mounted back in.
#
# The weights are gitignored; fetch them into the build context first
# (platforms/jetson/models/README.md):
#   curl -L -o platforms/jetson/models/yolov8n.onnx \
#     https://raw.githubusercontent.com/Zhang-zu-hao/Industrial-security-demo/master/models/yolov8n.onnx
# md5 eda19d19a8aa73d54411a6fc1d091c7f — the build asserts it below, because a
# silently different export changes the published coordinates.
ARG UBUNTU_TAG=22.04
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

FROM ubuntu:${UBUNTU_TAG} AS builder
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
COPY platforms/jetson/pyproject.toml platforms/jetson/uv.lock /src/jetson/
# The dependency set comes from uv.lock rather than being re-resolved, so the
# image matches what the tests ran against. tensorrt and PyGObject are not in
# the lock on purpose (see pyproject.toml): the first is board-supplied, the
# second is the distro's python3-gi, installed into the runtime stage below.
# --system-site-packages is what lets both resolve from outside the venv.
RUN pip install --no-cache-dir uv \
 && python3 -m venv --system-site-packages /opt/venv \
 && cd /src/jetson \
 && uv export --frozen --no-dev --no-emit-project --format requirements-txt > /tmp/req.txt \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/req.txt \
 && find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name 'tests' -prune -exec rm -rf {} +
# The wheels are deliberately NOT stripped. jammy's binutils rewrites the 64
# KiB-aligned program headers these aarch64 wheels use and the result fails to
# load: "libscipy_openblas64_-....so: ELF load command address/offset not
# page-aligned". The ~40 MB it would save is not worth an image whose numpy only
# fails at runtime.

FROM ubuntu:${UBUNTU_TAG}

LABEL org.opencontainers.image.title="edge-security-detector-jetson" \
      org.opencontainers.image.description="Jetson Orin person detector for edge-security-kit: RTSP in over NVDEC, YOLOv8n on TensorRT, sensecraft.detection/1 over MQTT out, with a JPEG preview endpoint. CUDA, TensorRT and the L4T GStreamer plugins come from the board, not from this image." \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/suharvest/edge-security-kit" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Seeed Studio"

# python3-gi + the three typelibs esk_jetson asks for (Gst, GstRtsp, GstVideo);
# plugins-base carries appsink/capsfilter, plugins-good carries rtspsrc and
# rtph264depay, plugins-bad carries h264parse. Unlike the RK3588 image, the
# parser is installed here rather than staged from the board: on L4T only the
# two nv* plugins are special, and the container runtime mounts those in.
# libv4l-0 and libegl1 are link-time dependencies of libgstnvvideo4linux2.so;
# without them the mounted plugin loads as a broken .so and nvv4l2decoder
# silently does not exist. libgomp1 is the OpenMP runtime the OpenCV wheel
# links against.
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
      gstreamer1.0-plugins-bad \
      libv4l-0 \
      libegl1 \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# XDG_CACHE_HOME / GST_REGISTRY: GStreamer writes its plugin registry there. The
# default would be a root-owned $HOME that a read-only or non-root run cannot
# write, and a registry it fails to write is a registry it rebuilds on every
# start — which on this platform also means re-scanning the injected L4T
# plugins on every start.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    XDG_CACHE_HOME=/tmp \
    GST_REGISTRY=/tmp/gstreamer-1.0-registry.bin

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY platforms/jetson/esk_jetson/ /app/esk_jetson/
COPY platforms/jetson/tools/build_engine.sh /app/tools/build_engine.sh
COPY platforms/jetson/models/yolov8n.onnx /app/models/yolov8n.onnx

RUN chmod +x /app/tools/build_engine.sh \
 && python3 -c "import hashlib,sys; h=hashlib.md5(open('/app/models/yolov8n.onnx','rb').read()).hexdigest(); sys.exit(0 if h=='eda19d19a8aa73d54411a6fc1d091c7f' else 'unexpected model md5: '+h)"

# Sanity-check what the image itself owns. tensorrt, cuda-python's libcudart and
# nvv4l2decoder are intentionally missing here and are not checked — they arrive
# from the board at runtime. A wheel that unpacked but cannot load its own
# shared objects is otherwise only discovered when a camera is attached.
RUN python3 -c "import gi; gi.require_version('Gst','1.0'); gi.require_version('GstRtsp','1.0'); gi.require_version('GstVideo','1.0'); from gi.repository import Gst, GstRtsp, GstVideo; Gst.init(None); assert Gst.ElementFactory.make('rtspsrc','s'), 'rtspsrc missing'; assert Gst.ElementFactory.make('rtph264depay','d'), 'rtph264depay missing'; assert Gst.ElementFactory.make('h264parse','p'), 'h264parse missing'; assert Gst.ElementFactory.make('appsink','a'), 'appsink missing'; import numpy, cv2, yaml, paho.mqtt.client, cuda; import esk_jetson; print('gst', Gst.version_string(), '| numpy', numpy.__version__, '| cv2', cv2.__version__)"

EXPOSE 8099

# /healthz answers 200 as soon as the preview server is up; it does not assert
# that a frame has decoded, which is deliberate — a camera that is briefly
# unreachable must not make the container restart-loop. Ask /debug/decode for
# which decoder GStreamer actually instantiated.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8099/healthz', timeout=3).status==200 else 1)"

ENTRYPOINT ["python3", "-m", "esk_jetson"]
CMD ["--config", "/app/config/detector.yaml"]

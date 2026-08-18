#!/usr/bin/env bash
# Build a TensorRT engine from a YOLOv8 ONNX export, ON THE TARGET DEVICE.
#
# A serialized TensorRT engine is not portable: it is tied to the exact
# TensorRT version, the GPU architecture and, in practice, the driver. Building
# elsewhere and copying the result produces a null deserialize with no other
# diagnostic, so this script is meant to be run on the Jetson itself and its
# output belongs in models/, which is gitignored.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL.onnx OUTPUT.engine [trtexec args...]" >&2
  exit 2
fi

ONNX_PATH=$1
ENGINE_PATH=$2
shift 2
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PRECISION=${PRECISION:-fp16}

[[ -f "$ONNX_PATH" ]] || { echo "ONNX model not found: $ONNX_PATH" >&2; exit 1; }
[[ -x "$TRTEXEC" ]] || { echo "trtexec not executable: $TRTEXEC" >&2; exit 1; }
mkdir -p "$(dirname "$ENGINE_PATH")"

echo "== TensorRT builder =="
# trtexec 10.3 exits 1 on --version, so it cannot gate the build.
"$TRTEXEC" --version 2>&1 | head -3 || true
echo "ONNX:      $ONNX_PATH"
echo "Engine:    $ENGINE_PATH"
echo "Precision: $PRECISION"

# Ultralytics exports come in two flavours and the difference is not visible in
# the filename. A dynamic export needs an explicit optimization profile, or
# trtexec picks the parser's placeholder shape and produces an engine that
# either fails tactic selection or is unusable at runtime. A static export
# refuses a profile outright:
#
#   [E] Static model does not take explicit shapes since the shape of inference
#       tensors will be determined by the model itself
#
# Rather than make the operator know which export they have, the build is tried
# with a profile and retried without one on exactly that error.
TRT_INPUT_NAME=${TRT_INPUT_NAME:-images}
TRT_INPUT_SIZE=${TRT_INPUT_SIZE:-640}
has_explicit_shapes=false
for argument in "$@"; do
  case "$argument" in
    --minShapes=*|--optShapes=*|--maxShapes=*|--shapes=*) has_explicit_shapes=true ;;
  esac
done

trtexec_args=(
  "--onnx=$ONNX_PATH"
  "--saveEngine=$ENGINE_PATH"
  "--skipInference"
  "--builderOptimizationLevel=3"
  "--timingCacheFile=${ENGINE_PATH}.timing.cache"
)
case "$PRECISION" in
  fp32) ;;
  fp16) trtexec_args+=("--fp16") ;;
  int8)
    # int8 needs a calibration cache; --fp16 stays on so layers the calibrator
    # cannot quantize fall back to fp16 rather than fp32.
    trtexec_args+=("--fp16" "--int8")
    [[ -n "${TRT_CALIB_CACHE:-}" ]] && trtexec_args+=("--calib=${TRT_CALIB_CACHE}")
    ;;
  *) echo "unknown PRECISION: $PRECISION (fp32|fp16|int8)" >&2; exit 2 ;;
esac
if [[ "$has_explicit_shapes" == false ]]; then
  trtexec_args+=(
    "--minShapes=${TRT_INPUT_NAME}:1x3x${TRT_INPUT_SIZE}x${TRT_INPUT_SIZE}"
    "--optShapes=${TRT_INPUT_NAME}:1x3x${TRT_INPUT_SIZE}x${TRT_INPUT_SIZE}"
    "--maxShapes=${TRT_INPUT_NAME}:1x3x${TRT_INPUT_SIZE}x${TRT_INPUT_SIZE}"
  )
fi
trtexec_args+=("$@")

build_log=$(mktemp)
trap 'rm -f "$build_log"' EXIT
started=$(date +%s)
if ! "$TRTEXEC" "${trtexec_args[@]}" 2>&1 | tee "$build_log"; then
  if grep -q "Static model does not take explicit shapes" "$build_log"; then
    echo "== static ONNX export: retrying without an optimization profile =="
    retry_args=()
    for argument in "${trtexec_args[@]}"; do
      case "$argument" in
        --minShapes=*|--optShapes=*|--maxShapes=*) ;;
        *) retry_args+=("$argument") ;;
      esac
    done
    "$TRTEXEC" "${retry_args[@]}"
  else
    echo "trtexec failed; see the output above" >&2
    exit 1
  fi
fi
elapsed=$(( $(date +%s) - started ))

test -s "$ENGINE_PATH"
# The timing cache is a build accelerator, not a runtime artifact. On a device
# with a few GB free it is not worth keeping around by default.
if [[ "${KEEP_TIMING_CACHE:-0}" != "1" ]]; then
  rm -f "${ENGINE_PATH}.timing.cache"
fi
echo "Engine written: $ENGINE_PATH"
echo "Build seconds:  $elapsed"
sha256sum "$ENGINE_PATH"

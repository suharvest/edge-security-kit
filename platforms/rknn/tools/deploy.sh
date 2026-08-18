#!/bin/sh
# Push the detector to an RK3588 or RK3576 board and bring it up.
#
# Deliberately additive: it creates one directory tree under $HOME on the board
# and starts one process matched on this project's own config path. It never
# stops, removes or reconfigures anything it did not create, and it installs
# nothing system-wide -- the venv borrows rknn-toolkit-lite2 and PyGObject from
# the OS rather than replacing them.
set -eu

usage() {
  cat <<'EOF'
Usage: deploy.sh --device DEVICE --config FILE [options]
  --device DEVICE   Fleet target (default: radxa)
  --config FILE     detector config to install as config/config.yaml
  --model FILE      .rknn to push (default: ../models/yolov8n_fp16.rk3588.rknn)
  --remote DIR      install root on the board (default: $HOME/edge-security-rknn)
  --no-up           push and validate, but do not start the detector
  --dry-run         print every action without changing remote state
EOF
}

device=radxa config='' model='' remote='' up=yes dry_run=no
while [ "$#" -gt 0 ]; do
  case "$1" in
    --device) device=$2; shift 2 ;;
    --config) config=$2; shift 2 ;;
    --model) model=$2; shift 2 ;;
    --remote) remote=$2; shift 2 ;;
    --no-up) up=no; shift ;;
    --dry-run) dry_run=yes; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$config" ] || { echo "--config is required" >&2; exit 2; }

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
platform_dir=$(CDPATH='' cd -- "$script_dir/.." && pwd)
model=${model:-$platform_dir/models/yolov8n_fp16.rk3588.rknn}
remote=${remote:-/home/radxa/edge-security-rknn}
fleet=${FLEET_BIN:-$HOME/.rpty/bin/fleet}

[ -f "$model" ] || { echo "model not found: $model (run prepare_model.sh)" >&2; exit 4; }

run() { if [ "$dry_run" = yes ]; then echo "DRY-RUN: $*"; else "$@"; fi; }

run "$fleet" exec "$device" -- mkdir -p "$remote/esk_rknn" "$remote/models" "$remote/config"
for f in "$platform_dir"/esk_rknn/*.py; do
  run "$fleet" push "$device" "$f" "$remote/esk_rknn/$(basename "$f")"
done
run "$fleet" push "$device" "$platform_dir/pyproject.toml" "$remote/pyproject.toml"
run "$fleet" push "$device" "$config" "$remote/config/config.yaml"

# Only push the model when the board does not already have this exact one:
# it is 8 MB and the boards this runs on are usually short of space.
if [ "$dry_run" = yes ]; then
  echo "DRY-RUN: compare remote SHA256 and push $model if it differs"
else
  if command -v sha256sum >/dev/null 2>&1; then local_sha=$(sha256sum "$model" | awk '{print $1}')
  else local_sha=$(shasum -a 256 "$model" | awk '{print $1}'); fi
  remote_sum=$("$fleet" exec "$device" -- sha256sum "$remote/models/$(basename "$model")" 2>/dev/null || true)
  case "$remote_sum" in
    "$local_sha "*) echo "remote model cache hit ($local_sha)" ;;
    *) "$fleet" push "$device" "$model" "$remote/models/$(basename "$model")" ;;
  esac
fi

# --system-site-packages is required, not a preference: rknn-toolkit-lite2 must
# be the board's copy so it matches the kernel NPU driver, and PyGObject must
# match the system GStreamer or the MPP plugin is invisible and decode silently
# drops to CPU.
run "$fleet" exec --timeout 600 "$device" -- sh -c \
  "cd $remote && export PATH=\$HOME/.local/bin:\$PATH && [ -d .venv ] || uv venv --system-site-packages --python /usr/bin/python3 .venv && uv pip install --python .venv/bin/python 'opencv-python-headless>=4.9' 'paho-mqtt>=1.6,<2.0' 'pyyaml>=6.0'"

run "$fleet" exec --timeout 300 "$device" -- sh -c \
  "cd $remote && ./.venv/bin/python -m esk_rknn --config config/config.yaml --validate"

if [ "$up" = yes ]; then
  run "$fleet" exec --timeout 300 "$device" -- sh -c \
    "pkill -f 'esk_rknn --config $remote/config/config.yaml' 2>/dev/null; sleep 1; cd $remote && nohup ./.venv/bin/python -m esk_rknn --config config/config.yaml > $remote/detector.log 2>&1 & sleep 8; tail -5 $remote/detector.log"
fi
echo "deployed to $device:$remote"

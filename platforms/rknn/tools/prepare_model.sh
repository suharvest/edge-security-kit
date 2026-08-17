#!/bin/sh
# Build platforms/rknn/models/yolov8n_fp16.rk3588.rknn from the same ONNX the
# generic platform uses, and record its SHA256.
#
# RKNN Toolkit 2 is x86_64-only, so on any other host this delegates to an
# x86_64 Fleet device (--builder-host). The toolkit version is pinned to the
# version string inside the board's librknnrt (2.3.2 on radxa), not to the
# filename of the .so, which lies.
set -eu

usage() {
  cat <<'EOF'
Usage: prepare_model.sh --onnx FILE [options]
  --onnx FILE            source YOLOv8n ONNX (expected md5 eda19d19a8aa73d54411a6fc1d091c7f)
  --platform rk3588      target SoC (default rk3588)
  --output-dir DIR       default: ../models relative to this script
  --builder-host DEVICE  x86_64 Fleet device to convert on when this host is not x86_64
  --toolkit-version V    rknn-toolkit2 version (default 2.3.2)
  --dry-run              print the actions only
EOF
}

onnx='' platform=rk3588 output_dir='' builder_host='' toolkit=2.3.2 dry_run=no
while [ "$#" -gt 0 ]; do
  case "$1" in
    --onnx) onnx=$2; shift 2 ;;
    --platform) platform=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --builder-host) builder_host=$2; shift 2 ;;
    --toolkit-version) toolkit=$2; shift 2 ;;
    --dry-run) dry_run=yes; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$onnx" ] || { echo "--onnx is required" >&2; exit 2; }
case "$platform" in rk3576|rk3588) ;; *) echo "--platform must be rk3576 or rk3588" >&2; exit 2 ;; esac

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
output_dir=${output_dir:-$script_dir/../models}
name=yolov8n_fp16.$platform.rknn
output=$output_dir/$name
manifest=$output_dir/SHA256SUMS
fleet=${FLEET_BIN:-$HOME/.rpty/bin/fleet}

sha_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
record_sha() {
  tmp=$manifest.tmp.$$
  mkdir -p "$output_dir"
  if [ -f "$manifest" ]; then awk -v n="$name" '$2!=n && $2!=("*" n)' "$manifest" > "$tmp"
  else : > "$tmp"; fi
  echo "$1  $name" >> "$tmp"; mv "$tmp" "$manifest"
}

arch=${RKNN_TEST_ARCH:-$(uname -m)}
if [ "$arch" != x86_64 ] && [ "$arch" != amd64 ]; then
  [ -n "$builder_host" ] || {
    echo "RKNN Toolkit 2 conversion is x86_64-only; this host is $arch." >&2
    echo "Re-run with --builder-host <x86_64 Fleet device>." >&2
    exit 5
  }
  work=/tmp/esk-rknn-builder-$platform
  if [ "$dry_run" = yes ]; then
    echo "DRY-RUN: delegate conversion to $builder_host in $work"
    exit 0
  fi
  "$fleet" exec "$builder_host" -- mkdir -p "$work"
  "$fleet" push "$builder_host" "$script_dir/convert_yolov8_rknn.py" "$work/convert_yolov8_rknn.py"
  "$fleet" push "$builder_host" "$onnx" "$work/$(basename "$onnx")"
  # --isolated keeps the toolkit out of the builder's global environment, and
  # unsetting the proxy variables keeps a large wheel download off a metered
  # tunnel; the pinned index is a domestic mirror.
  "$fleet" exec --timeout 3600 "$builder_host" -- sh -c "cd $work && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy UV_PYTHON_PREFERENCE=only-system uv run --isolated --python 3.12 --with rknn-toolkit2==$toolkit --with 'setuptools<81' --with 'onnx==1.16.1' python convert_yolov8_rknn.py --onnx $work/$(basename "$onnx") --out $work/$name --platform $platform"
  mkdir -p "$output_dir"
  "$fleet" pull "$builder_host" "$work/$name" "$output"
  actual=$(sha_file "$output"); record_sha "$actual"
  echo "prepared through $builder_host: $output ($actual)"
  exit 0
fi

if [ "$dry_run" = yes ]; then echo "DRY-RUN: convert $onnx -> $output"; exit 0; fi
mkdir -p "$output_dir"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  UV_PYTHON_PREFERENCE=only-system \
  uv run --isolated --python 3.12 \
    --with "rknn-toolkit2==$toolkit" --with 'setuptools<81' --with 'onnx==1.16.1' \
    python "$script_dir/convert_yolov8_rknn.py" \
      --onnx "$onnx" --out "$output" --platform "$platform"
actual=$(sha_file "$output"); record_sha "$actual"
echo "prepared: $output ($actual)"

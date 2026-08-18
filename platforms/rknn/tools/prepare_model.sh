#!/bin/sh
# Build a platforms/rknn/models/*.rknn from a YOLOv8n ONNX, and record its SHA256.
#
# RKNN Toolkit 2 is x86_64-only, so on any other host this delegates to an
# x86_64 Fleet device (--builder-host). The toolkit version is pinned to the
# version string inside the board's librknnrt (2.3.2 as shipped), not to the
# filename of the .so, which lies.
#
# Three models are in play; see ../README.md for what each is for.
#
#   A  --graph ultralytics --dtype fp16   yolov8n_fp16.rk3588.rknn
#   B  --graph zoo         --dtype fp16   yolov8n_zoo_fp16.rk3588.rknn
#   C  --graph zoo         --dtype int8   yolov8n_zoo_int8.rk3588.rknn
#
# int8 additionally needs --calib-dir: a directory of images on the *builder*,
# turned into the path list RKNN's build(dataset=...) reads. Build it with
# ../calibration/build_calibration.py and copy it over; a wrong calibration set
# is a silent accuracy loss, not an error.
set -eu

usage() {
  cat <<'EOF'
Usage: prepare_model.sh --onnx FILE [options]
  --onnx FILE            source YOLOv8n ONNX
  --graph ultralytics|zoo  which export the ONNX is (default ultralytics)
  --dtype fp16|int8      build precision (default fp16)
  --calib-dir DIR        directory of calibration images ON THE BUILDER (int8 only)
  --platform rk3588      target SoC (default rk3588)
  --output-dir DIR       default: ../models relative to this script
  --builder-host DEVICE  x86_64 Fleet device to convert on when this host is not x86_64
  --builder-workdir DIR  scratch dir on the builder (default /tmp/esk-rknn-builder-<platform>)
  --builder-env "K=V .."  extra env for the builder (e.g. UV_CACHE_DIR on a host whose / is full)
  --toolkit-version V    rknn-toolkit2 version (default 2.3.2)
  --dry-run              print the actions only
EOF
}

onnx='' graph=ultralytics dtype=fp16 calib_dir='' platform=rk3588
output_dir='' builder_host='' builder_workdir='' builder_env='' toolkit=2.3.2 dry_run=no
while [ "$#" -gt 0 ]; do
  case "$1" in
    --onnx) onnx=$2; shift 2 ;;
    --graph) graph=$2; shift 2 ;;
    --dtype) dtype=$2; shift 2 ;;
    --calib-dir) calib_dir=$2; shift 2 ;;
    --platform) platform=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --builder-host) builder_host=$2; shift 2 ;;
    --builder-workdir) builder_workdir=$2; shift 2 ;;
    --builder-env) builder_env=$2; shift 2 ;;
    --toolkit-version) toolkit=$2; shift 2 ;;
    --dry-run) dry_run=yes; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$onnx" ] || { echo "--onnx is required" >&2; exit 2; }
case "$platform" in rk3576|rk3588) ;; *) echo "--platform must be rk3576 or rk3588" >&2; exit 2 ;; esac
case "$graph" in ultralytics|zoo) ;; *) echo "--graph must be ultralytics or zoo" >&2; exit 2 ;; esac
case "$dtype" in fp16|int8) ;; *) echo "--dtype must be fp16 or int8" >&2; exit 2 ;; esac
if [ "$dtype" = int8 ] && [ -z "$calib_dir" ]; then
  echo "--dtype int8 requires --calib-dir (see ../calibration/)" >&2; exit 2
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
output_dir=${output_dir:-$script_dir/../models}
case "$graph" in
  ultralytics) stem=yolov8n ;;
  zoo)         stem=yolov8n_zoo ;;
esac
name=${stem}_${dtype}.$platform.rknn
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
  work=${builder_workdir:-/tmp/esk-rknn-builder-$platform}
  if [ "$dry_run" = yes ]; then
    echo "DRY-RUN: delegate $graph/$dtype conversion to $builder_host in $work"
    exit 0
  fi
  "$fleet" exec "$builder_host" -- mkdir -p "$work"
  "$fleet" push "$builder_host" "$script_dir/convert_yolov8_rknn.py" "$work/convert_yolov8_rknn.py"
  "$fleet" push "$builder_host" "$onnx" "$work/$(basename "$onnx")"
  dataset_arg=''
  if [ "$dtype" = int8 ]; then
    dataset_arg="--dataset $work/dataset.txt"
  fi
  # The whole remote job is written to a file and executed as one argument.
  # `fleet exec ... -- sh -c "a && b"` is not safe: the command is re-joined on
  # the way over, the quoting around the -c argument is lost, and `sh -c cd`
  # then runs on its own -- the rest of the pipeline executes in $HOME with no
  # error. A pushed script has no quoting to lose.
  #
  # --isolated keeps the toolkit out of the builder's global environment, and
  # unsetting the proxy variables keeps a large wheel download off a metered
  # tunnel; the pinned index is a domestic mirror.
  runner=$(mktemp)
  {
    echo '#!/bin/sh'
    echo 'set -eu'
    echo "cd $work"
    # The calibration list has to hold builder-side absolute paths, so it is
    # generated there. AppleDouble files survive a tar written on macOS and the
    # toolkit will happily try to decode one, so they are cut.
    if [ "$dtype" = int8 ]; then
      echo "find '$calib_dir' -type f \\( -name '*.jpg' -o -name '*.png' \\) ! -name '._*' | sort > $work/dataset.txt"
      echo "echo \"calibration images: \$(wc -l < $work/dataset.txt)\""
    fi
    echo "exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \\"
    echo "  UV_PYTHON_PREFERENCE=only-system $builder_env \\"
    echo "  uv run --isolated --python 3.12 --with rknn-toolkit2==$toolkit --with 'setuptools<81' --with 'onnx==1.16.1' \\"
    echo "  python $work/convert_yolov8_rknn.py --onnx $work/$(basename "$onnx") --out $work/$name \\"
    echo "    --platform $platform --dtype $dtype $dataset_arg"
  } > "$runner"
  "$fleet" push "$builder_host" "$runner" "$work/run_convert.sh"
  rm -f "$runner"
  "$fleet" exec --timeout 3600 "$builder_host" -- sh "$work/run_convert.sh"
  mkdir -p "$output_dir"
  "$fleet" pull "$builder_host" "$work/$name" "$output"
  actual=$(sha_file "$output"); record_sha "$actual"
  echo "prepared through $builder_host: $output ($actual)"
  exit 0
fi

if [ "$dry_run" = yes ]; then echo "DRY-RUN: convert $onnx -> $output"; exit 0; fi
mkdir -p "$output_dir"
dataset_arg=''
if [ "$dtype" = int8 ]; then
  find "$calib_dir" -type f \( -name '*.jpg' -o -name '*.png' \) ! -name '._*' | sort > "$output_dir/dataset.txt"
  dataset_arg="--dataset $output_dir/dataset.txt"
fi
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  UV_PYTHON_PREFERENCE=only-system \
  uv run --isolated --python 3.12 \
    --with "rknn-toolkit2==$toolkit" --with 'setuptools<81' --with 'onnx==1.16.1' \
    python "$script_dir/convert_yolov8_rknn.py" \
      --onnx "$onnx" --out "$output" --platform "$platform" --dtype "$dtype" $dataset_arg
actual=$(sha_file "$output"); record_sha "$actual"
echo "prepared: $output ($actual)"

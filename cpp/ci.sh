#!/usr/bin/env bash
# CI: build bench_ort against ONNX Runtime 1.30.0 and require bit-identical outputs to Python ORT.
# Headers: conda-forge onnxruntime-cpp 1.30.0 (sha256 pinned). Library: the onnxruntime 1.30.0 pip
# wheel (the conda library needs a newer libstdc++ than Ubuntu 22.04 ships). Needs fetched assets.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${PY:-python}"
work="${WORK:-$(mktemp -d)}"
pkg=onnxruntime-cpp-1.30.0-h987d1f6_1_cpu
sha=6f042fb15ce643fe954c1eefef3bc3c02a2f302a1b2c76d4bbea8eedaca75e3e

if [ ! -f "$work/$pkg.conda" ]; then
  curl -sSfL -o "$work/$pkg.conda" "https://conda.anaconda.org/conda-forge/linux-64/$pkg.conda"
fi
echo "$sha  $work/$pkg.conda" | sha256sum -c -
mkdir -p "$work/x" "$work/ort"
"$py" -m zipfile -e "$work/$pkg.conda" "$work/x/"
zstd -dc "$work/x/pkg-$pkg.tar.zst" | tar -x -C "$work/ort" include
lib="$("$py" -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')"
mkdir -p "$work/lib"
ln -sf "$lib/libonnxruntime.so.1.30.0" "$work/lib/libonnxruntime.so.1"

ORT_INCLUDE="$work/ort/include/onnxruntime/core/session" ORT_LIB="$work/lib" "$here/build.sh"
"$py" "$here/dump_obs.py" runs/ppo_v2_chunk1.pt runs/policy_v2.onnx
"$here/bench_ort" runs/policy_v2.onnx "$here/obs.bin" "$here/ref.bin" 2000

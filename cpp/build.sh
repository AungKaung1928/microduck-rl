#!/usr/bin/env bash
# Build bench_ort against an ONNX Runtime with the same version as the Python wheel.
#   ORT_INCLUDE  dir holding onnxruntime_cxx_api.h (e.g. include/onnxruntime/core/session of the
#                conda-forge onnxruntime-cpp package of the same version)
#   ORT_LIB      dir holding libonnxruntime.so.1 (the Python wheel's onnxruntime/capi works,
#                symlink libonnxruntime.so.1 -> libonnxruntime.so.1.<ver>)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${ORT_INCLUDE:?set ORT_INCLUDE}" "${ORT_LIB:?set ORT_LIB}"
g++ -O2 -std=c++17 -Wall -Wextra -I "$ORT_INCLUDE" "$here/bench_ort.cpp" \
  -L "$ORT_LIB" -l:libonnxruntime.so.1 -Wl,-rpath,"$ORT_LIB" -o "$here/bench_ort"
echo "built $here/bench_ort"

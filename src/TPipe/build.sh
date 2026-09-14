#!/usr/bin/env bash
# Build (and optionally run) every example. No dependencies beyond a C++17
# compiler - the whole point is that this runs on your laptop.
set -euo pipefail
cd "$(dirname "$0")"

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra}

mkdir -p bin
for src in examples/*.cpp; do
  name=$(basename "${src}" .cpp)
  echo "building ${name}"
  ${CXX} ${CXXFLAGS} -Iinclude -Iexamples "${src}" -o "bin/${name}"
done

if [[ "${1:-}" == "--run" ]]; then
  for bin in bin/*; do
    [[ -f "${bin}" && -x "${bin}" ]] || continue
    echo
    echo "===================== $(basename "${bin}") ====================="
    "${bin}"
  done
fi

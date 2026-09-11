#!/usr/bin/env bash
# Fallback one-shot build without CMake (Debian/Ubuntu LLVM packages).
set -euo pipefail
LLVM_VER=${LLVM_VER:-18}
CFG=llvm-config-${LLVM_VER}
clang++-${LLVM_VER} -std=c++17 -fno-rtti \
  $(${CFG} --cxxflags) \
  toy.cpp \
  $(${CFG} --ldflags) -lMLIR $(${CFG} --libs core support) \
  -o toy
echo "built ./toy"

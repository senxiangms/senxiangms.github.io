#!/usr/bin/env bash
# Fallback build without CMake (Debian/Ubuntu LLVM packages): run mlir-tblgen
# by hand, then compile. This is exactly what the CMake rules above automate.
set -euo pipefail
LLVM_VER=${LLVM_VER:-18}
TBLGEN=mlir-tblgen-${LLVM_VER}
CFG=llvm-config-${LLVM_VER}
INC=$(${CFG} --includedir)
OUT=${OUT:-gen}

mkdir -p "${OUT}"
${TBLGEN} -gen-op-decls      -I "${INC}" ToyOps.td -o "${OUT}/ToyOps.h.inc"
${TBLGEN} -gen-op-defs       -I "${INC}" ToyOps.td -o "${OUT}/ToyOps.cpp.inc"
${TBLGEN} -gen-dialect-decls -dialect=toy -I "${INC}" ToyOps.td \
  -o "${OUT}/ToyOpsDialect.h.inc"
${TBLGEN} -gen-dialect-defs  -dialect=toy -I "${INC}" ToyOps.td \
  -o "${OUT}/ToyOpsDialect.cpp.inc"
echo "generated ${OUT}/*.inc"

clang++-${LLVM_VER} -std=c++17 -fno-rtti \
  $(${CFG} --cxxflags) -I. -I"${OUT}" \
  toy.cpp ToyDialect.cpp \
  $(${CFG} --ldflags) -Wl,-rpath,$(${CFG} --libdir) \
  -lMLIR $(${CFG} --libs core support) \
  -o toy
echo "built ./toy"

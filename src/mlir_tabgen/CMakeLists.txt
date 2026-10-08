cmake_minimum_required(VERSION 3.20)
project(toy-dialect-ods LANGUAGES CXX C)

# Point at an MLIR install, e.g.
#   -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# The .inc files land in the build tree, so it has to be on the include path.
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#===----------------------------------------------------------------------===#
# Run mlir-tblgen over ToyOps.td
#===----------------------------------------------------------------------===#

# Every mlir_tablegen() below reads this file; set it before the calls.
set(LLVM_TARGET_DEFINITIONS ToyOps.td)
mlir_tablegen(ToyOps.h.inc -gen-op-decls)
mlir_tablegen(ToyOps.cpp.inc -gen-op-defs)
mlir_tablegen(ToyOpsDialect.h.inc -gen-dialect-decls -dialect=toy)
mlir_tablegen(ToyOpsDialect.cpp.inc -gen-dialect-defs -dialect=toy)
# One target that owns the four generated files; anything that includes them
# must depend on it, or a parallel build races the generator.
add_public_tablegen_target(ToyOpsIncGen)

#===----------------------------------------------------------------------===#
# The executable
#===----------------------------------------------------------------------===#

add_llvm_executable(toy toy.cpp ToyDialect.cpp DEPENDS ToyOpsIncGen)
llvm_update_compile_flags(toy)
target_link_libraries(toy
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
mlir_check_all_link_libraries(toy)

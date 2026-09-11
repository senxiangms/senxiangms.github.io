cmake_minimum_required(VERSION 3.20)
project(toy-dialect LANGUAGES CXX C)

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
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_llvm_executable(toy toy.cpp)
llvm_update_compile_flags(toy)
target_link_libraries(toy
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
)
mlir_check_all_link_libraries(toy)

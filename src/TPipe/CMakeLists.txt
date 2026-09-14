cmake_minimum_required(VERSION 3.16)
project(mini-ascendc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mini_ascendc INTERFACE)
target_include_directories(mini_ascendc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()
foreach(example add_simple add_double_buffer sync_check)
  add_executable(${example} examples/${example}.cpp)
  target_link_libraries(${example} PRIVATE mini_ascendc)
  target_include_directories(${example} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
  target_compile_options(${example} PRIVATE -Wall -Wextra)
  # Every example verifies its own result and returns non-zero on a mismatch or
  # an unexpected synchronisation report.
  add_test(NAME ${example} COMMAND ${example})
endforeach()

cmake_minimum_required(VERSION 3.22)

project(membench LANGUAGES CUDA CXX)

# CUDA tool-chain provided by SCALE
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------------------------
# Google Benchmark (pulled only once, even if other targets do the same)
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
FetchContent_MakeAvailable(googlebenchmark)

# -------------------------------------------------------------------
# Source files for this benchmark
add_executable(membench
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_compile_options(membench PRIVATE -Wall)
target_link_libraries(membench PRIVATE CUDA::cudart benchmark::benchmark)

cmake_minimum_required(VERSION 3.22)

# ── project & languages ────────────────────────────────────────────
project(membench LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── include path for hip_to_cuda.h (repo root) ─────────────────────
# CMAKE_CURRENT_LIST_DIR = .../benchs/memcpy
include_directories(${CMAKE_CURRENT_LIST_DIR}/../..)   # => /workspace

# ── Google Benchmark fetch (one-time; cached by CMake) ─────────────
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
FetchContent_MakeAvailable(googlebenchmark)

# ── source files for this benchmark ────────────────────────────────
add_executable(membench
    main.cpp
    src/membench.hpp
    src/helper.hpp
)

target_compile_options(membench PRIVATE -Wall)
target_link_libraries(membench PRIVATE CUDA::cudart benchmark::benchmark)

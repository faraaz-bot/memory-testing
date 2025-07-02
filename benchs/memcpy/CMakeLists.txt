cmake_minimum_required(VERSION 3.24)

# ──────────────────────────────────────────────────────────────
#  Project & language setup
# ──────────────────────────────────────────────────────────────
project(membench LANGUAGES CXX CUDA)

# Make sure the CUDA language is really on
enable_language(CUDA)

# Tell CMake where the shim CUDA toolkit lives (set by `scaleenv`)
set(CUDAToolkit_ROOT $ENV{CUDA_HOME})
find_package(CUDAToolkit REQUIRED)        # creates CUDA::cudart, … targets

# C++ / CUDA dialect
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CUDA_STANDARD       17)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ──────────────────────────────────────────────────────────────
#  Google Benchmark dependency
# ──────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.1
)
FetchContent_MakeAvailable(googlebenchmark)

# ──────────────────────────────────────────────────────────────
#  Sources
#   – only the .cpp goes here; headers are pulled in automatically
#   – mark it as CUDA so nvcc handles kernels sprinkled in headers
# ──────────────────────────────────────────────────────────────
add_executable(membench
    main.cpp
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

# ──────────────────────────────────────────────────────────────
#  Linking
# ──────────────────────────────────────────────────────────────
target_link_libraries(membench
    PRIVATE
        CUDA::cudart                # CUDA runtime from the shim
        benchmark::benchmark        # Google Benchmark
        Threads::Threads
)

# ──────────────────────────────────────────────────────────────
#  Optional unit-test target
# ──────────────────────────────────────────────────────────────
enable_testing()

add_executable(test-membench
    test/membench-test.cpp
)
set_source_files_properties(test/membench-test.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(test-membench
    PRIVATE
        CUDA::cudart
        benchmark::benchmark
        Threads::Threads
)

add_test(NAME membench-test COMMAND test-membench)

# ────────────────────────────────────────────────────────────────────
# CMakeLists.txt  (CUDA / SCALE edition)
# ────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.22)

project(membench LANGUAGES CUDA CXX)
find_package(GTest QUIET)
find_package(TBB QUIET)

# Make sure the CUDA toolkit shim injected by SCALE is visible
find_package(CUDA REQUIRED)           # comes from scaleenv
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)   # SCALE maps “native” → gfx*

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Include path for our shim and project headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Fetch Google Benchmark -----------------------------------------------------
include(FetchContent)
option(BENCHMARK_ENABLE_TESTING OFF)
option(BENCHMARK_ENABLE_INSTALL OFF)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
set(HAVE_STD_REGEX ON)
set(RUN_HAVE_STD_REGEX 1)
FetchContent_MakeAvailable(googlebenchmark)

# ── Fetch Google Test ----------------------------------------------------------
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

# ── Main executable ------------------------------------------------------------
add_executable(membench
    main.cpp
    src/membench.hpp
)

# Any .cu or .cpp that still contains kernel launches must be marked CUDA
# (If you renamed HIP kernels to .cu, CMake auto-detects them.)
set_source_files_properties(
    main.cpp
    src/membench.hpp
    PROPERTIES LANGUAGE CUDA
)

target_compile_options(membench PRIVATE -Wall)
target_link_libraries(membench
    PRIVATE
        CUDA::cudart          # SCALE’s runtime stub
        benchmark::benchmark
)

# ── Unit-test executable -------------------------------------------------------
add_executable(test-membench test/membench-test.cpp)
target_compile_options(test-membench PRIVATE -Wall)
target_link_libraries(test-membench
    PRIVATE
        CUDA::cudart
        GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test-membench)

# ── MPI variant (unchanged except hip→CUDA links) ------------------------------
option(ENABLE_MPI       "Use MPI"       OFF)
option(ENABLE_CRAY_MPI  "Use Cray MPI"  OFF)

if(ENABLE_MPI OR ENABLE_CRAY_MPI)
    find_package(MPI REQUIRED)
    add_executable(mpi-membench
        mpi-main.cpp
        src/mpi-membench.hpp
    )
    target_compile_options(mpi-membench PRIVATE -Wall)
    add_compile_definitions(MPI_ENABLED)

    if(ENABLE_MPI)
        target_link_libraries(mpi-membench
            PRIVATE
                CUDA::cudart
                benchmark::benchmark
                MPI::MPI_CXX
        )
    elseif(ENABLE_CRAY_MPI)
        get_filename_component(MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
        target_link_directories(mpi-membench
            PRIVATE
                ${MPI_LIBDIR}/../../../../gtl/lib
        )
        target_link_libraries(mpi-membench
            PRIVATE
                CUDA::cudart
                benchmark::benchmark
                mpi_gtl_hsa
                MPI::MPI_CXX
        )
    endif()
endif()

# ── Use SCALE’s clang++ by default for C++ host code ---------------------------
# (Optional: remove if scaleenv already sets CC/CXX)
set(CMAKE_CXX_COMPILER "/opt/scale/bin/amdclang++")

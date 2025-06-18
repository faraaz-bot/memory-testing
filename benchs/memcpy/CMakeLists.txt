cmake_minimum_required(VERSION 3.19)

project(membench LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED ENV{ROCM_PATH})
    set(ENV{ROCM_PATH} "/opt/rocm")
endif()

if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(ENV{CMAKE_PREFIX_PATH} "/opt/rocm")
endif()

# list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/.local)
# list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH})

find_package(hip REQUIRED)

include_directories(include)
include(FetchContent)
option(ENABLE_MPI "Use MPI" OFF)
option(ENABLE_CRAY_MPI "Use Cray MPI" OFF)

# Google Benchmark (https://github.com/google/benchmark)
message(STATUS "Fetching Google Benchmark ....")
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library" OFF)
option(BENCHMARK_ENABLE_INSTALL "Enable installation of benchmark" OFF)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
set(HAVE_STD_REGEX ON)
set(RUN_HAVE_STD_REGEX 1)
FetchContent_MakeAvailable(googlebenchmark)


# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(membench main.cpp src/membench.hpp)
set_source_files_properties(main.cpp mpi-main.cpp src/membench.hpp src/mpi-membench.hpp src/helper.hpp src/mpi-helper.hpp PROPERTIES LANGUAGE HIP)

target_compile_options(membench PRIVATE -Wall)
target_link_libraries(membench hip::host hip::device)
target_link_libraries(membench benchmark::benchmark)

add_executable(test-membench test/membench-test.cpp)
target_link_libraries(test-membench hip::host hip::device GTest::gtest_main)
include(GoogleTest)

if(ENABLE_MPI OR ENABLE_CRAY_MPI)
    find_package(MPI REQUIRED)
    add_executable(mpi-membench mpi-main.cpp src/mpi-membench.hpp)
    add_compile_definitions(MPI_ENABLED) 
endif()

if(ENABLE_MPI)
    target_link_libraries(mpi-membench hip::host hip::device benchmark::benchmark MPI::MPI_CXX)
    message(STATUS "MPI available and enabled")
elseif(ENABLE_CRAY_MPI)
    get_filename_component(MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
    target_link_directories(mpi-membench
        PRIVATE
        ${MPI_LIBDIR}/../../../../gtl/lib)
    target_link_libraries(mpi-membench hip::host hip::device benchmark::benchmark mpi_gtl_hsa MPI::MPI_CXX)
    message(STATUS "Cray MPI available and enabled")
else()
    message(STATUS "Not using MPI")
endif()

set(CMAKE_CXX_COMPILER "/opt/rocm/bin/amdclang++")

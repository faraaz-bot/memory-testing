cmake_minimum_required( VERSION 3.19 )

project(membench LANGUAGES CXX HIP)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

if (NOT DEFINED ENV{ROCM_PATH})
    set (ENV{ROCM_PATH} "/opt/rocm")

endif()

if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set (ENV{CMAKE_PREFIX_PATH} "/opt/rocm")
endif()

# list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH})
find_package( hip REQUIRED )

include_directories(include)
include(FetchContent)
# find_package( MPI REQUIRED )
# option(CRAY_MPI_ENABLE "Cray MPI" OFF)




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

add_executable( membench main.cpp src/mem-bench.cpp)

set_source_files_properties(mem-bench.cpp mem-bench.hpp PROPERTIES LANGUAGE HIP)
target_link_libraries( membench hip::host hip::device )
target_link_libraries( membench benchmark::benchmark)

# if ( CRAY_MPI_ENABLE )
#     get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
#     target_link_directories( membench
#         PRIVATE
#         ${MPI_LIBDIR}/../../../../gtl/lib )
#     target_link_libraries( membench hip::host hip::device mpi_gtl_hsa MPI::MPI_CXX )
# else()
#     target_link_libraries( membench hip::host hip::device MPI::MPI_CXX)    
# endif()

cmake_minimum_required(VERSION 3.19)

project(mpi-errors LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED ENV{ROCM_PATH})
    set (ENV{ROCM_PATH} "/opt/rocm")
endif()

if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set (ENV{CMAKE_PREFIX_PATH} "/opt/rocm")
endif()

find_package(hip REQUIRED)
find_package(MPI REQUIRED)

include_directories(include)
include(FetchContent)
option(ENABLE_CRAY_MPI "Use Cray MPI (by default use other MPI library)" OFF)

add_executable(test main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

target_compile_options(test PRIVATE -Wall)

if(NOT ENABLE_CRAY_MPI)
    target_link_libraries(test hip::host hip::device MPI::MPI_CXX)
    message(STATUS "MPI available and enabled")
else()
    get_filename_component(MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
    target_link_directories(test
        PRIVATE
        ${MPI_LIBDIR}/../../../../gtl/lib)
    target_link_libraries(test hip::host hip::device mpi_gtl_hsa MPI::MPI_CXX)
    message(STATUS "Cray MPI available and enabled")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


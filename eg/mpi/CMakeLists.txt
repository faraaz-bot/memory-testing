cmake_minimum_required( VERSION 3.16 )

project(mpihip_eg LANGUAGES CXX)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

find_package( MPI REQUIRED )

find_package( HIP REQUIRED )

add_executable( gpu-aware gpu-aware.cpp )

target_link_libraries( gpu-aware hip::host MPI::MPI_CXX )

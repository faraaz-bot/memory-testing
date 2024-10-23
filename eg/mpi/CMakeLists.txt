cmake_minimum_required( VERSION 3.22 )

project(mpihip_eg LANGUAGES CXX HIP)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

find_package( hip REQUIRED )

find_package( MPI REQUIRED )

add_executable( gpu-aware gpu-aware.cpp )

set_source_files_properties(gpu-aware.cpp PROPERTIES LANGUAGE HIP)


target_link_libraries( gpu-aware hip::host hip::device  MPI::MPI_CXX )

cmake_minimum_required( VERSION 3.22 )

project(membench LANGUAGES CXX HIP)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH})
find_package( hip REQUIRED )

# find_package( MPI REQUIRED )
# option(CRAY_MPI_ENABLE "Cray MPI" OFF)

add_executable( membench mem-bench.cpp mem-bench.hpp )

set_source_files_properties(mem-bench.cpp mem-bench.hpp PROPERTIES LANGUAGE HIP)
target_link_libraries( membench hip::host hip::device )

# if ( CRAY_MPI_ENABLE )
#     get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
#     target_link_directories( membench
#         PRIVATE
#         ${MPI_LIBDIR}/../../../../gtl/lib )
#     target_link_libraries( membench hip::host hip::device mpi_gtl_hsa MPI::MPI_CXX )
# else()
#     target_link_libraries( membench hip::host hip::device MPI::MPI_CXX)    
# endif()

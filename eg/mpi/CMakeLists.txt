cmake_minimum_required( VERSION 3.22 )

project(mpihip_eg LANGUAGES CXX HIP)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

find_package( hip REQUIRED )

find_package( MPI REQUIRED )

add_executable( gpu-aware gpu-aware.cpp )

set_source_files_properties(gpu-aware.cpp PROPERTIES LANGUAGE HIP)

message("ok, how about: ${MPI_CXX_LIB_NAMES}")

# get_cmake_property(_variableNames VARIABLES)
# foreach(_variableName ${_variableNames})
#   message(STATUS "${_variableName} : ${${_variableName}}")
# endforeach()

if ( MPI_CXX_LIB_NAMES STREQUAL "mpi_cray")
  message("cray mpi it is!")
  #export MPICH_GPU_SUPPORT_ENABLED=1
  add_link_options( gpu-aware "-L${CRAYLIBS_X86_64}")
  message("CRAYLIBS_X86_64: ${CRAYLIBS_X86_64}")
  message("manual L: /opt/cray/pe/mpich/8.1.28/gtl/lib")
  message("${MPI_LIBRARY}")
  get_filename_component(BASEDIR ${MPI_LIBRARY} DIRECTORY)
  message("BASEDIR: ${BASEDIR}")
  target_link_directories(gpu-aware PRIVATE ${BASEDIR}/../../../../gtl/lib)
  #target_link_directories(gpu-aware PRIVATE /opt/cray/pe/mpich/8.1.28/gtl/lib)
  target_link_libraries( gpu-aware mpi_gtl_hsa)
endif()

target_link_libraries( gpu-aware hip::host hip::device MPI::MPI_CXX)

cmake_minimum_required( VERSION 3.22 )

project(mpihip_eg LANGUAGES CXX HIP)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

find_package( hip REQUIRED )

find_package( MPI REQUIRED )
option(CRAY_MPI_ENABLE "Cray MPI" OFF)

set( hip_list gpu-aware isendrcv alltoall alltoallv)
set( host_list isendrcv_cpu mpihello )


foreach( sample ${hip_list} )
  add_executable( ${sample} ${sample}.cpp )

  set_source_files_properties(${sample}.cpp PROPERTIES LANGUAGE HIP)
  #set_property( TARGET ${sample} PROPERTY HIP_ARCHITECTURES gfx90a)

  if ( CRAY_MPI_ENABLE )
    get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
    target_link_directories( ${sample}
	PRIVATE
	${MPI_LIBDIR}/../../../../gtl/lib )
    target_link_libraries( ${sample} hip::host hip::device mpi_gtl_hsa MPI::MPI_CXX )
  else()
    target_link_libraries( ${sample} hip::host hip::device MPI::MPI_CXX)    
  endif()
endforeach( )


foreach( sample ${host_list} )
  add_executable( ${sample} ${sample}.cpp )

  if ( CRAY_MPI_ENABLE )
    get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
    target_link_directories( ${sample}
	PRIVATE
	${MPI_LIBDIR}/../../../../gtl/lib )
    target_link_libraries( ${sample} mpi_gtl_hsa MPI::MPI_CXX )
  else()
    target_link_libraries( ${sample} MPI::MPI_CXX )
  endif()
endforeach( )

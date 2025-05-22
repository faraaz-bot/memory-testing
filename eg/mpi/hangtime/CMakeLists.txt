cmake_minimum_required( VERSION 3.22 )


project(mpihang LANGUAGES CXX)

set( CMAKE_CXX_STANDARD 17 )


find_package( MPI REQUIRED )

set( targets a2ahang puteg )

foreach( target ${targets} )
  add_executable( ${target} ${target}.cpp )
  target_link_libraries( ${target} MPI::MPI_CXX)    


endforeach()

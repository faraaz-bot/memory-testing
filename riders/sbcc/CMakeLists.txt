cmake_minimum_required(VERSION 3.10)

project( sbcc_rider LANGUAGES CXX )

option( SBCC_RUNTIME_COMPILE "Runtime-compile the SBCC kernel with hipRTC"  OFF )

add_executable( sbcc-rider sbcc-rider.cpp)
set_property(TARGET sbcc-rider PROPERTY CXX_STANDARD 17 )

if( SBCC_RUNTIME_COMPILE )
  target_compile_definitions( sbcc-rider PUBLIC SBCC_RUNTIME_COMPILE=1 )
endif()
target_compile_options( sbcc-rider PRIVATE -fopenmp )
target_link_libraries( sbcc-rider PRIVATE -fopenmp )

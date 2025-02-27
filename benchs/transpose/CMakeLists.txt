cmake_minimum_required(VERSION 3.10)


project( transpose_rider LANGUAGES CXX )

option( ROCFFT_SRC "rocFFT source dir" )

option( USE_LDS "LDS version or naive version"  ON )
option( ROW_MAJOR "Make LDS version row-major"  OFF )

option( TRANSPOSE_RUNTIME_COMPILE "Runtime-compile the transpose kernel with hipRTC"  OFF )

find_package( Boost COMPONENTS program_options REQUIRED)

add_executable( transpose-rider transpose-rider.cpp)
set_property(TARGET transpose-rider PROPERTY CXX_STANDARD 17 )


IF( ROW_MAJOR )
  target_compile_definitions( transpose-rider PUBLIC ROW_MAJOR )
endif()

  
IF( USE_LDS )
  target_compile_definitions( transpose-rider PUBLIC USE_LDS=1 )
else()
  target_compile_definitions( transpose-rider PUBLIC USE_LDS=0 )
endif( USE_LDS)

if( TRANSPOSE_RUNTIME_COMPILE )
  target_compile_definitions( transpose-rider PUBLIC TRANSPOSE_RUNTIME_COMPILE=1 )
endif()

target_include_directories( transpose-rider
    PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    )

target_link_libraries( transpose-rider
    PRIVATE
    ${Boost_LIBRARIES}
    )

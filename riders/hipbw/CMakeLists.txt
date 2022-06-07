cmake_minimum_required(VERSION 3.10)


project( hipbw LANGUAGES CXX )

option( ROCFFT_SRC "rocFFT source dir" )

include_directories( "${ROCFFT_SRC}" )

find_package( Boost COMPONENTS program_options REQUIRED)

add_executable( hipbw-rider hipbw-rider.cpp)
set_property(TARGET hipbw-rider PROPERTY CXX_STANDARD 17 )

target_include_directories( hipbw-rider
    PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    )

target_link_libraries( hipbw-rider
    PRIVATE
    ${Boost_LIBRARIES}
    )

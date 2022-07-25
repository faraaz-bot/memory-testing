cmake_minimum_required(VERSION 3.10)


project( transpose_rider LANGUAGES CXX )

option( ROCFFT_SRC "rocFFT source dir" )

include_directories( "${ROCFFT_SRC}" )

find_package( Boost COMPONENTS program_options REQUIRED)

add_executable( transpose-rider transpose-rider.cpp)
set_property(TARGET transpose-rider PROPERTY CXX_STANDARD 14 )

target_include_directories( transpose-rider
    PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    )

target_link_libraries( transpose-rider
    PRIVATE
    ${Boost_LIBRARIES}
    )

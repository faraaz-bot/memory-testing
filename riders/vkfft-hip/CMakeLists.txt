cmake_minimum_required( VERSION 3.19 )

project( vkfft-hip-clients LANGUAGES CXX )

add_executable( vkfft-hip-rider vkfft-hip-rider.cpp )
set_property( TARGET vkfft-hip-rider PROPERTY CXX_STANDARD 17 )

find_package( Boost COMPONENTS program_options REQUIRED)


list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip)

add_compile_definitions(VKFFT_BACKEND=2)

target_include_directories( vkfft-hip-rider PUBLIC
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
  ${VKFFT_SRC_DIR}/include
  ${VKFFT_SRC_DIR}/vkFFT
  ${ROCFFT_SRC_DIR}/clients
  ${ROCFFT_SRC_DIR}/shared)

add_library( VkFFT INTERFACE )
target_compile_definitions( VkFFT INTERFACE -DVKFFT_BACKEND=2 )
target_link_libraries( vkfft-hip-rider
  PUBLIC hip::host
  VkFFT
  Boost::program_options )



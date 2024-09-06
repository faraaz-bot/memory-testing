cmake_minimum_required(VERSION 3.16)
project(rtceg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)


add_executable(rtc rtc.cpp)

find_package( HIP REQUIRED)

find_package( Boost COMPONENTS program_options REQUIRED)

find_package(OpenMP REQUIRED)
target_link_libraries(rtc PUBLIC OpenMP::OpenMP_CXX)


target_include_directories( rtc
  PRIVATE
  $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>)


target_link_libraries( rtc
  PRIVATE
  Boost::program_options
  hip::host
  )
  

cmake_minimum_required(VERSION 3.17)
project(cache_query LANGUAGES CXX HIP)
list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH})
find_package(HIP REQUIRED)
find_package(hsa-runtime64 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cache_query cache_query.cpp)
target_link_libraries(cache_query hip::device hip::host hsa-runtime64::hsa-runtime64)

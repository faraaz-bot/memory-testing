cmake_minimum_required(VERSION 3.16)
project(cuda_eg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(HIP REQUIRED)
find_package(HIPFFT REQUIRED)
add_executable(dual dual.cpp cuda_side.cpp hip_side.cpp)

get_target_property(dual hip::hipfft INCLUDE_DIRECTORIES)

message("sadf ${hipfft_INCLUDE_DIRS}")

# Question: why do we need the /hipfft at the end here?
target_include_directories(dual PRIVATE ${hipfft_INCLUDE_DIRS}/hipfft)

target_link_libraries(dual CUDA::cudart CUDA::cufft hip::host hip::hipfft)

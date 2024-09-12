cmake_minimum_required( VERSION 3.21 )

project( hip_simple_kernels LANGUAGES CXX HIP )
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

add_executable( aotasan  asan.cpp )
set_source_files_properties(asan.cpp PROPERTIES LANGUAGE HIP)

add_executable( rtcasan rtcasan.cpp )

find_package( Boost COMPONENTS program_options REQUIRED)

find_package( hip REQUIRED )

# cmake doesn't add the xnack option to the architectures (and seems
# to add an architecture for each device, which doesn't make sense),
# so clean that up:
set(newarch "")
list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)
foreach(arch ${CMAKE_HIP_ARCHITECTURES})
  message(STATUS "arch: ${arch}")
  if(NOT arch MATCHES "xnack" ) 
    message(STATUS "no xnack in ${arch}")
    string(APPEND arch ":xnack+")
    message(STATUS "new arch: ${arch}")
    list(APPEND newarch ${arch})
  endif()
endforeach()
set(CMAKE_HIP_ARCHITECTURES ${newarch})

message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
set_property(TARGET aotasan PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})

target_compile_options(aotasan PRIVATE -fsanitize=address)
target_link_options(aotasan PRIVATE -fsanitize=address -shared-libasan)
target_link_libraries( aotasan hip::device hip::host Boost::program_options  )

target_compile_options(rtcasan PRIVATE -fsanitize=address )
target_link_options(rtcasan PRIVATE  -fsanitize=address -shared-libasan)
target_link_libraries( rtcasan hip::host Boost::program_options)

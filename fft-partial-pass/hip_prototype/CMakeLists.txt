# #############################################################################
# Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

cmake_minimum_required(VERSION 3.16)

# This should appear before the project command, because it does not
# use FORCE
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH
  "Install path prefix, prepended onto install directories")

set(CMAKE_BUILD_TYPE Debug)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

project(partial-pass LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT HIP_FOUND)
  find_package(HIP REQUIRED)
endif()

if(NOT ROCM_FOUND)
  find_package(ROCM 0.7.3 REQUIRED)
endif()

include(ROCMInstallTargets)

set(partial-pass_source
  run.cpp
  rocfft/kargs.cpp
)

set(partial-pass_includes
  rocfft/arithmetic.h
  rocfft/kargs.h
  rocfft/butterfly_constant.h
  rocfft/callback.h
  rocfft/common.h
  rocfft/memory_gfx.h
  rocfft/rocfft_complex.h
  rocfft/gpubuf.h
  rocfft/real2complex_device.h
  rocfft/rocfft_butterfly_template.h
  rocfft/twiddle_factors.h
  CS_3D_RC_64_64_64/sbrr_64.h
  CS_3D_RC_64_64_64/sbcc_64.h
  partial_pass_RR_CC_64_64_64/sbrr_pp_64.h
  partial_pass_RR_CC_64_64_64/sbcc_pp_64.h
  compute_twiddles.h
  launch_3D_FFT.h
  launch_common.h
)

add_executable(partial-pass ${partial-pass_source} ${partial-pass_includes})

include(ExternalProject)

set(rocfft-test_include_dirs
  ${ROCM_CLANG_ROOT}/include
)

target_compile_options(partial-pass PRIVATE ${WARNING_FLAGS} -Wno-cpp)

# set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")
# set(CMAKE_C_FLAGS_DEBUG "-ggdb -O0")

target_include_directories(partial-pass
  PRIVATE
  ${partial-pass_include_dirs}
)

target_link_libraries(partial-pass
  PRIVATE
  hip::device
  hip::host
)

set_target_properties(partial-pass PROPERTIES
  DEBUG_POSTFIX "-d"
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set(OUT_DIR "/bin")
string(CONCAT OUT_DIR "${PROJECT_BINARY_DIR}" ${OUT_DIR})

set_target_properties(partial-pass
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
  ${OUT_DIR})

# rocm_install(TARGETS partial-pass COMPONENT tests)

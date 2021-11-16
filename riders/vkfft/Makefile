#
# vkfft-rider makefile
#
# After cloning VkFFT, build it using cmake, then
#
# make VKFFT_HOME=/path/to/vkfft/source/dir ROCFFT_HOME=/path/to/rocfft/source/dir
#

VKFFT_HOME ?= $(HOME)/VkFFT
ROCFFT_HOME ?= $(HOME)/rocFFT

INCLUDES = \
 -I$(VKFFT_HOME)/vkFFT \
 -I$(VKFFT_HOME)/benchmark_scripts/vkFFT_scripts/include \
 -I$(VKFFT_HOME)/glslang-master/glslang/Include \
 -I$(ROCFFT_HOME)/clients \
 -I$(ROCFFT_HOME)/library/include

LIBRARIES = \
  -lboost_program_options \
  -lvulkan \
  -L$(VKFFT_HOME)/build/glslang-master/glslang -lglslang -lMachineIndependent -lGenericCodeGen \
  -L$(VKFFT_HOME)/build/glslang-master/OGLCompilersDLL -lOGLCompiler \
  -L$(VKFFT_HOME)/build/glslang-master/glslang/OSDependent/Unix -lOSDependent \
  -L$(VKFFT_HOME)/build/glslang-master/SPIRV -lSPIRV \
  -lvulkan

all: vkfft-rider

vkfft-rider: vkfft-rider.o utils_VkFFT.o
	hipcc -std=c++17 -o vkfft-rider vkfft-rider.o utils_VkFFT.o $(LIBRARIES)

vkfft-rider.o: vkfft-rider.cpp
	hipcc -std=c++17 $(INCLUDES) vkfft-rider.cpp -c -o vkfft-rider.o

utils_VkFFT.o:
	hipcc -std=c++17 $(INCLUDES) $(VKFFT_HOME)/benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp -c -o utils_VkFFT.o

#
# vkfft-rider makefile
#
# After cloning VkFFT, build it using cmake, install to VKFFT_HOME, then
#
# make VKFFT_HOME=/path/to/vkfft/install/dir ROCFFT_SRC=/path/to/rocfft/source/dir VKFFT_SRC=/path/to/vkFFT_source_dir
#

VKFFT_SRC ?= $(HOME)/VkFFT
VKFFT_ROOT ?= $(HOME)/VkFFT
ROCFFT_SRC ?= $(HOME)/rocFFT

INCLUDES = \
 -I$(VKFFT_SRC)/vkFFT \
 -I$(VKFFT_SRC)/benchmark_scripts/vkFFT_scripts/include \
 -I$(VKFFT_SRC)/glslang-master/glslang/Include \
 -I$(ROCFFT_SRC)/clients \
 -I$(ROCFFT_SRC)/library/include

LIBRARIES = \
  -lboost_program_options \
  -lvulkan \
  -L$(VKFFT_HOME)/lib -lglslang -lMachineIndependent -lGenericCodeGen \
  -L$(VKFFT_HOME)/lib -lOGLCompiler \
  -L$(VKFFT_HOME)/lib -lOSDependent \
  -L$(VKFFT_HOME)/lib -lSPIRV \
  -lvulkan

all: vkfft-rider

vkfft-rider: vkfft-rider.o utils_VkFFT.o
	hipcc -std=c++17 -o vkfft-rider vkfft-rider.o utils_VkFFT.o $(LIBRARIES)

vkfft-rider.o: vkfft-rider.cpp
	hipcc -std=c++17 $(INCLUDES) vkfft-rider.cpp -c -o vkfft-rider.o

utils_VkFFT.o:
	hipcc -std=c++17 $(INCLUDES) $(VKFFT_SRC)/benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp -c -o utils_VkFFT.o

clean:  FORCE
	rm -rf vkfft-rider vkfft-rider.o utils_VkFFT.o

FORCE:

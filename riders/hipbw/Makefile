# make ROCFFT_SRC=~/repo/rocfft8 RUNTIME=AMD
#
# export HIP_PLATFORM=nvidia
# make ROCFFT_SRC=~/repo/rocfft8 RUNTIME=NVIDIA
#
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib

CXX ?= /opt/rocm/hip/bin/hipcc

RUNTIME ?= AMD

ROCFFT_SRC ?= $(HOME)/rocFFT


#INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm-4.5.0/llvm/include -I/opt/rocm-4.5.0/hip/include/


SRCS=hipbw-rider.cpp
OBJS=$(SRCS:.cpp=.o)

ALL = hipbw-rider


ifeq ($(strip $(RUNTIME)),AMD)
	INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm/hip/include/ -I/opt/rocm/llvm/include/
	CXXFLAGS=-std=c++17 -D__HIP_PLATFORM_AMD__ -Ofast $(INCLUDES) -x hip
	AMD=no
endif
ifeq ($(strip $(RUNTIME)),NVIDIA)
	INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm/hip/include/  -I/usr/local/cuda/include
	CXXFLAGS= -std=c++17 -D__HIP_PLATFORM_NVCC__ -O3 $(INCLUDES)
endif

LIBRARIES=-L$(HOME)/boost -lboost_program_options -L/opt/rocm/hip/lib -lamdhip64 -lhiprtc-builtins
LDFLAGS=--driver-mode=g++ -lgcc_s -lgcc   $(LIBRARIES) -lclang_rt.builtins-x86_64 -lclang_rt.builtins-x86_64
LDFLAGS=--driver-mode=g++ -L"/opt/rocm-4.5.0/hip/lib" -lgcc_s -lgcc -lpthread -lm -lrt -Wl,--enable-new-dtags -Wl,-rpath=/opt/rocm-4.5.0/hip/lib:/opt/rocm-4.5.0/lib -lamdhip64 -std=c++17 --driver-mode=g++ -L/home/mal/boost -lboost_program_options -L/opt/rocm/hip/lib -lamdhip64 -lhiprtc-builtins -o "hipbw-rider" -L/opt/rocm-4.5.0/llvm/bin/../lib/clang/13.0.0/lib/linux -lclang_rt.builtins-x86_64

all: $(ALL)

hipbw-rider: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -f *.o $(ALL)


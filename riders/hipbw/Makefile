# make ROCFFT_SRC=~/repo/rocfft8 RUNTIME=AMD
#
# export HIP_PLATFORM=nvidia
# make ROCFFT_SRC=~/repo/rocfft8 RUNTIME=NVIDIA
#
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib

CXX=/opt/rocm/hip/bin/hipcc

RUNTIME ?= AMDs

ROCFFT_SRC ?= $(HOME)/rocFFT


#INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm-4.5.0/llvm/include -I/opt/rocm-4.5.0/hip/include/


SRCS=hipbw-rider.cpp
OBJS=$(SRCS:.cpp=.o)

ALL = hipbw-rider


ifeq ($(strip $(RUNTIME)),AMD)
	INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm/hip/include/
	CXXFLAGS= -std=c++17 -D__HIP_PLATFORM_AMD__ -Ofast $(INCLUDES)
	AMD=no
endif
ifeq ($(strip $(RUNTIME)),NVIDIA)
	INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm/hip/include/  -I/usr/local/cuda/include
	CXXFLAGS= -std=c++17 -D__HIP_PLATFORM_NVCC__ -O3 $(INCLUDES)
endif

LIBRARIES=-L$(HOME)/boost -lboost_program_options -L/opt/rocm/hip/lib -lamdhip64 -lhiprtc-builtins
LDFLAGS=$(LIBRARIES)

all: $(ALL)

hipbw-rider: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -f *.o $(ALL)


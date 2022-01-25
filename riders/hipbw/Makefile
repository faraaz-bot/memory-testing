# make ROCFFT_SRC=~/repo/rocfft8 CXX=clang++
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/hip/lib

CXX=/opt/rocm/hip/bin/hipcc


ROCFFT_SRC ?= $(HOME)/rocFFT

INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm/hip/include/
#INCLUDES=-I$(ROCFFT_SRC) -I/opt/rocm-4.5.0/llvm/include -I/opt/rocm-4.5.0/hip/include/

LIBRARIES=-L$(HOME)/boost -lboost_program_options -L/opt/rocm/hip/lib -lamdhip64 -lhiprtc-builtins

SRCS=hipbw-rider.cpp
OBJS=$(SRCS:.cpp=.o)

ALL = hipbw-rider

CXXFLAGS= -std=c++17 -D__HIP_PLATFORM_AMD__ -Ofast $(INCLUDES)
LDFLAGS=$(LIBRARIES)

all: $(ALL)

hipbw-rider: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -f *.o $(ALL)


#
# Simple Makefile for the rocFFT TFLOPS test
#
# Builds the MPI and non-MPI versions by default.
#

ROCM     = /opt/rocm-4.3.0-7053
CXX      = hipcc
CXXFLAGS = -std=c++14 -Wall
OMPI     = /usr/lib/x86_64-linux-gnu/openmpi

LIBFFT ?= -L$(ROCM)/lib -lrocfft
LIBMPI ?= -I$(OMPI)/include -pthread -L$(OMPI)/lib -lmpi_cxx -lmpi -lopen-pal

all: rocfft-tflops rocfft-tflops-mpi

rocfft-tflops: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBFFT)

rocfft-tflops-mpi: main.cpp
	$(CXX) $(CXXFLAGS) -DUSE_MPI $< -o $@ $(LIBFFT) $(LIBMPI)

rocfft-tflops-verify: main.cpp
	$(CXX) $(CXXFLAGS) -DUSE_FFTW $< -o $@ $(LIBFFT) -lfftw3 -lfftw3f

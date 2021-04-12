#
# Simple Makefile for the rocFFT TFLOPS test
#
# Builds the MPI and non-MPI versions by default.
#

CXX      = hipcc
CXXFLAGS = -std=c++14 -Wall

LIBFFT ?= -L/opt/rocm/lib -lrocfft
LIBMPI ?= -lmpi

all: rocfft-tflops rocfft-tflops-mpi

rocfft-tflops: main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBFFT)

rocfft-tflops-mpi: main.cpp
	$(CXX) $(CXXFLAGS) -DUSE_MPI $< -o $@ $(LIBFFT) $(LIBMPI)

rocfft-tflops-verify: main.cpp
	$(CXX) $(CXXFLAGS) -DUSE_FFTW $< -o $@ $(LIBFFT) -lfftw3 -lfftw3f

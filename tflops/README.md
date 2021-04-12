# rocFFT TFLOPS

ROCFFT-TFLOPS is a simple program to measure rocFFT TFLOPS.

A simple 1D-batched transform is distributed across multiple MPI
ranks.  Using two MPI ranks per MI200 allows the benchmark to use both
MCMs on a single MI200.

Each rank uses GPU event timers to measure elapsed kernel time and
compute TFLOPS according to

  TFLOPS = nbatch * 5 * n * log_2(n) / time

The reported TFLOPS is the sum of TFLOPS across the MPI ranks.

## Build

Use `make`.  You can edit the Makefile or set the CXX, CXXFLAGS etc
variables on the command line.

## Run

Default transform length, batch size, and precision:

    mpiexec -n 2 ./rocfft-tflops-mpi

Full options are:

    mpiexec -n 2 ./rocfft-tflops-mpi LENGTH NBATCH PRECISION
    
where `PRECISION` is 0 for double, and 1 for single.

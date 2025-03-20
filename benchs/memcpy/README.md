# Multi-GPU Memory Transfer Benchmark
Benchmarking tool for comparing speed of various memory copy methods
between multiple gpus. Currently will do out-of-place operations on 
square matrices only.

Implementations to Run:
- hipMemcpy2D(Async)
- Copy Kernel
- MPI alltoall(v)
- RCCL

# Build & Usage
mkdir build\
cd build\
cmake ..\
make -j\
`./membench`                # using default matrix size, number of gpus\
`./membench -n 128 -g 4`    # using 128 x 128 matrix, for 4 gpus \
`./membench -h`             # explore other options like precision, verbosity, etc.\

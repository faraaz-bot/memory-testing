# Multi-GPU Memory Transfer Benchmark
Benchmarking tool for comparing speed of various memory copy methods
between multiple gpus. Currently will do out-of-place operations on 
square matrices only.

Implementations to Run:
- hipMemcpy2D(+Async)
- Copy Kernel
- MPI alltoall(+v)
- RCCL

# Build & Usage
The runner.py tool can be used to invoke a built membench executable for varying sizes/GPUs
or scaling modes, and graph the relevant data. It currently does not support MPI benchmarks.

The following shows examples of running the membench executable directly:
`mkdir build` \
`cd build` \
`cmake ..` \
`make -j` \
`./membench`                                 # using default matrix size, number of gpus \
`./membench -n 128 -g 4`                     # using 128 x 128 matrix, for 4 gpus \
`./membench -f memcpy2D naiveCopy+Transpose` # specify filter on benchmarks to run \
`./membench -h`                              # explore further options \

Due to how MPI is typically ran, the method for utilizing multiple GPU devices
will not work properly for non-MPI benchmarks, so the executables are split up
with their respective MPI and non-MPI benchmarks. `membench` and `mpi-membench`
are the respective make build targets/executables.
`cmake -DENABLE_MPI=true ..` # tell CMake to look for MPI and add `mpi-membench` target
`cmake -DENABLE_CRAY_MPI=true ..` # similar, but for specifically CRAY MPI

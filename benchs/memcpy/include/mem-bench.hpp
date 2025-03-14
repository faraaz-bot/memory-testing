#include "../../../eg/argv/CLI11.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdio.h>
#include <vector>

#define HIP_CHECK(cmd)                                                                  \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)



enum copy_operation
{
    memcpy_async,
    mpi_alltoall,
    copy_kernel,
    all,
};



/**
 * Benchmarking tool for comparing speed of various memory copy methods
 * between multiple gpus. Currently will do out-of-place operations on 
 * square matrices only.
 *
 * TODO list:
 * - Implement basic implementations for each method
 * - Optimize stuff after
 *     - Experiment with async, LDS optimizations, bank conflicts
 *     - Toggling SDMA
 * - Perform local transpose on data as well
 *
 * - Display/write output timings/other metrics, allow ntrials
 *     - Add Google Benchmark
*/

// (1.1) hipMemcpy2D between two devices
void run_memcpy(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs);

// (1.2) hipMemcpy2D between two devices, using streams
void run_memcpy_async(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs, const std::vector<hipStream_t>& streams);

// (2) Copy kernel
__global__ void copy(const int N, const float* input, float* output);

// (3.1) MPI alltoall
// (3.2) MPI alltoallv

// (4) RCCL alltoall

/* Helpers for verifying correctness */

// Combine ngpu # of gpubuf partitions back in an N x N matrix on the host
// Assumes hostbuf_result has enough memory allocated for it
void assemble_output_to_host(const int N, const std::vector<float*>& gpubufs, float* hostbuf_result);

// Helper just to print N consecutive values in gpubuf
__global__ void print(const int N, const float* input);

// Helper just to print N consecutive values in gpubuf
__global__ void print2d(const int N, const int M, const float* input);

// Check equality of matrices
bool is_same_matrix(const int N, const std::vector<float>& input1, const std::vector<float>& input2);

// Helper to print initial host matrix and transposed matrix
void print_host_2d(const int N, const int M, const std::vector<float>& input);
// Reference impl (out-of-place)
void host_transpose(const int N, const std::vector<float>& input, std::vector<float>& output);
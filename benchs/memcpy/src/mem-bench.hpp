#include <hip/hip_runtime.h>
#include "helper.hpp"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <vector>
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

/* Helpers for verifying correctness */

// Helper kernel just to print N consecutive values in gpubuf
template<typename Tfloat>
__global__ void print(const int N, const Tfloat* input)
{
    printf("[ ");
    for(int i = 0; i < N; i++)
        printf("%.6f ", input[i]);
    printf("]\n");
}

// Helper kernel just to print N consecutive values in gpubuf
template<typename Tfloat>
__global__ void print2d(const int N, const int M, const Tfloat* input)
{
    printf("[\n");
    for(int i = 0; i < N; i++)
    {
        printf("\t[ ");
        for(int j = 0; j < M; j++)
        {
            auto idx = i + N * j;
            printf("%.6f ", input[idx]);
        }
        printf("]\n");
    }
    printf(" ]\n");
}

// Helper to print initial host matrix and transposed matrix
template<typename Tfloat>
void print_host_2d(const int N, const int M, const std::vector<Tfloat>& input)
{
    std::cout << "[\n";
    for(int i = 0; i < N; i++)
    {
        std::cout << "  [ ";
        for(int j = 0; j < M; j++)
        {
            auto idx = i * N + j;
            std::cout << std::setw(4) << input[idx] << " ";
        }
        std::cout << " ]\n";
    }
    std::cout << "]" << std::endl;
}

// Combine ngpu # of gpubuf partitions back in an N x N matrix on the host
// Assumes hostbuf_result has enough memory allocated for it
template<typename Tfloat>
void assemble_output_to_host(const int N, const std::vector<Tfloat*>& gpubufs, Tfloat* hostbuf_result)
{
    const size_t ngpus = gpubufs.size();
    const size_t buf_size = N * N / ngpus;
    for(auto i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipMemcpy(hostbuf_result + i * buf_size, gpubufs[i], buf_size * sizeof(Tfloat), hipMemcpyDeviceToHost));
    }
}

// Check equality of matrices
template<typename Tfloat>
bool is_same_matrix(const int N, const std::vector<Tfloat>& input1, const std::vector<Tfloat>& input2)
{
    for (auto i = 0; i < N*N; i++)
        if(input1[i] != input2[i]) return false;

    return true;
}

// Reference impl on CPU (out-of-place)
template<typename Tfloat>
void host_transpose(const int N, const std::vector<Tfloat>& input, std::vector<Tfloat>& output)
{
    output.reserve(N*N);
#pragma omp parallel for
    for(size_t i = 0; i < N; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            // Get curr index and send data to opposing location across diagonal
            auto idx1 = i * N + j;
            auto idx2 = j * N + i;
            output[idx2] = input[idx1];
        }
    }
}


/* Setup and Teardown helpers */
// Allocate and initialize gpu buffers, streams + distribute host input to gpu buffers
template<typename Tfloat>
void setup(size_t N, size_t ngpus, std::vector<Tfloat*>& gpubufs_input, std::vector<Tfloat*>& gpubufs_output, const std::vector<Tfloat>& host_input, std::vector<hipStream_t>& streams)
{
    const size_t buf_height = N / ngpus;
    const size_t buf_size = N * buf_height; // Number of elements in buf
    const size_t pitch_bytes = N * sizeof(Tfloat); // Size of a column in bytes incl. padding (which is 0)
    
    std::cout << "buf_height = " << buf_height << "\nbuf_size = " << buf_size << "\npitch_bytes = " << pitch_bytes << std::endl;
    for(size_t i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));

        HIP_CHECK(hipMalloc(&gpubufs_input[i], sizeof(Tfloat) * buf_size)); 
        HIP_CHECK(hipMemcpy(gpubufs_input[i], host_input.data() + i * buf_size, buf_size * sizeof(Tfloat), hipMemcpyHostToDevice));
        // HIP_CHECK(hipMemcpy2D(gpubufs_input[i], pitch_bytes, input.data() + i * buf_size, pitch_bytes, N, buf_height, hipMemcpyHostToDevice));
        
        HIP_CHECK(hipMalloc(&gpubufs_output[i], sizeof(Tfloat) * buf_size));
        HIP_CHECK(hipMemset(gpubufs_output[i], 0, sizeof(Tfloat) * buf_size));

        // Assign streams to current gpu
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }
}

// Clear data in 
// template<typename Tfloat>
// void reset()
// {}
//
// template<typename Tfloat>
// void teardown()
// {}

/* Implementations */

// (1.1) hipMemcpy2D between two devices
template<typename Tfloat>
void run_memcpy(const int N, const std::vector<Tfloat*>& in_bufs, std::vector<Tfloat*>& out_bufs)
{
    const size_t ngpus = in_bufs.size();
    const size_t sub_block_size = N / ngpus; // Length of block in each transfer
    const size_t bytes_to_copy_per_row = sub_block_size * sizeof(float); // Bytes per row in transfer
    const size_t pitch_bytes = N * sizeof(float); // Width of buf

    float ms;
    hipEvent_t start, end;
    for(auto i = 0; i < ngpus; i++) // src GPU
    {
        for(auto j = 0; j < ngpus; j++) // Offset within GPU, AKA dst GPU
        {
            HIP_CHECK(hipMemcpy2D(out_bufs[j] + (i * sub_block_size), pitch_bytes, in_bufs[i] + (j * sub_block_size), pitch_bytes, bytes_to_copy_per_row, sub_block_size, hipMemcpyDeviceToDevice));
        }
    }
    return;
}

// (1.2) hipMemcpy2D between two devices, using streams
template<typename Tfloat>
void run_memcpy_async(const int N, const std::vector<Tfloat*>& in_bufs, std::vector<Tfloat*>& out_bufs, const std::vector<hipStream_t>& streams)
{

    return;
}

// (2) Copy kernel
template<typename Tfloat>
__global__ void copy(const int N, const Tfloat* input, Tfloat* output);

// (3.1) MPI alltoall
// (3.2) MPI alltoallv

// (4) RCCL alltoall


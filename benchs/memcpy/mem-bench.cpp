#include "mem-bench.hpp"
#include "../../eg/argv/CLI11.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
// #include <mpi.h>
#include <random>
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
 *     - Experiment with async, hipDeviceEnablePeerAccess, LDS optimization, bank conflicts
 * - Perform local transpose on data as well
 * - Display/write output timings/other metrics
*/

// (1.1) hipMemcpy2D between two devices
// void run_memcpy(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs)
// {
//     const size_t buf_height = N / in_bufs.size();
//     const size_t buf_size = N * buf_height;
//
//     for(auto i = 0; i < N; i++)
//     {
//         for(auto j = 0; j < N; j++)
//         {
//             // Copy from in[i] to out[j]
//             HIP_CHECK(hipMemcpy2D(out_bufs[j],  , in_bufs[i]));
//         }
//     }
//     return;
// }

// (1.2) hipMemcpy2D between two devices, using streams
void run_memcpy_async(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs, const std::vector<hipStream_t>& streams)
{
    return;
}

// (2) Copy kernel
__global__ void copy(const int N, const float* input, float* output) 
{
   return; 
}

// (3.1) MPI alltoall
// (3.2) MPI alltoallv

// (4) RCCL alltoall

/* Helpers for verifying correctness */

// Combine ngpu # of gpubuf partitions back in an N x N matrix on the host
// void assemble_output_to_host(const int N, const std::vector<float*>& gpubufs, float* hostbuf_result)
// {
//     const size_t buf_height = N / in_bufs.size();
//     const size_t buf_size = N * buf_height;
//
//     for(auto i = 0; i < N; i++)
//     {
//         HIP_CHECK(hipMemcpy2D(hostbuf_result + i * buf_size, pitch_bytes, gpubufs_input[i], pitch_bytes, N, buf_height, hipMemcpyDeiceToHost));
//     }
//
// }

// Helper just to print N consecutive values in gpubuf
__global__ void print(const int N, const float* input)
{
    printf("[ ");
    for(int i = 0; i < N; i++)
        printf("%.f ", input[i]);
    printf("]\n");
}

// TODO
// Helper just to print N consecutive values in gpubuf
__global__ void print2d(const int N, const int M, const float* input)
{
    printf("[ ");
    for(int i = 0; i < N; i++)
        printf("%.f ", input[i]);
    printf("]");
}

// Check equality of matrices
bool is_same_matrix(const int N, float* input1, float* input2)
{
    for (auto i = 0; i < N*N; i++)
        if(input1[i] != input2[i]) return false;

    return true;
}

// Helper to print initial host matrix and transposed matrix
void print_host_2d(const int N, const int M, const std::vector<float>& input)
{
    std::cout << "[\n";
    for(int i = 0; i < N; i++)
    {
        std::cout << "\t[ ";
        for(int j = 0; j < M; j++)
        {
            auto idx = i * N + j;
            std::cout << std::setw(4) << input[idx] << " ";
        }
        std::cout << " ]\n";
    }
    std::cout << "]" << std::endl;
}

// Reference impl (out-of-place)
void host_transpose(const int N, const std::vector<float>& input, std::vector<float>& output)
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

int main(int argc, char* argv[])
{
    CLI::App app{"Memcpy bench"};

    size_t N;
    size_t ngpus;
    app.add_option("-n, --length", N, "Length of input square matrix")->default_val(1000U);
    app.add_option("-g, --ngpus", ngpus, "Number of gpus")->default_val(4U);
    // Could restrict which methods to compare

    std::cout << "Comparing on " << N << " x " << N << " size matrix, across " << ngpus << " gpus.\n";

    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    // Generate random input
    // Can consider adding in option to use rocRAND for faster device generation
    std::random_device                    rd;
    std::mt19937                          m_engine(rd()); // Mersenne Twister, rd as seed
    std::uniform_real_distribution<float> dist{-0.5, 0.5};

    std::vector<float> input(N*N);
    for(size_t i = 0; i < N*N; ++i)
        input[i] = dist(m_engine);

    std::cout << "Generated Input Matrix:\n";
    print_host_2d(N, N, input);

    std::vector<float> reference_matrix(N*N);
    host_transpose(N, input, reference_matrix);
    std::cout << "Host Transposed Matrix:\n";
    print_host_2d(N,N,reference_matrix);

    // Split input and transfer it
    // Assume inputs are evenly divisible :)
    std::vector<float*> gpubufs_input(ngpus);
    std::vector<float*> gpubufs_output(ngpus);
    std::vector<hipStream_t> streams(ngpus);

    const size_t buf_height = N / ngpus;
    const size_t buf_size = N * buf_height;
    const size_t pitch_bytes = N * sizeof(float); // Size of a column in bytes incl. padding (which is 0)

    // Allocate and init bufs, streams
    for(size_t i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));

        HIP_CHECK(hipMalloc(&gpubufs_input[i], sizeof(float) * buf_size)); 
        HIP_CHECK(hipMemcpy2D(gpubufs_input[i], pitch_bytes, input.data() + i * buf_size, pitch_bytes, N, buf_height, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&gpubufs_output[i], sizeof(float) * buf_size));
        std::cout << "GPU Buffer " << i << ":\n";
        print<<<1,1>>>(buf_size, gpubufs_input[i]);

        // Assign streams to current gpu
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    // -- Run stuff --

    
    // std::vector<float> h_assembled_output(N*N);
    // run_memcpy(N, gpubufs_input, gpubufs_output);
    // assemble_output_to_host(gpubufs_output, h_assembled_output.data());
    // bool res = is_same_matrix(reference_matrix, h_assembled_output.data());

    // Copy kernel
    // MPI alltoall
    // RCCL alltoall
    
    // Free up buffers, streams
    for(auto i = 0; i < ngpus; i++){
        HIP_CHECK(hipFree(gpubufs_input[i]));
        HIP_CHECK(hipFree(gpubufs_output[i]));
        HIP_CHECK(hipStreamDestroy(streams[i])); 
    }
}

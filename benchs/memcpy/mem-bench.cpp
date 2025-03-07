#include "mem-bench.hpp"
#include "../../eg/argv/CLI11.hpp"

#include <hip/hip_runtime.h>
#include <iostream>
#include <mpi.h>
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
 * - Perform local transpose on data as well
 * - Display/write output timings/other metrics
*/

// (1.1) hipMemcpy2D between two devices
void run_memcpy(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs)
{
    return;
}

// (1.2) hipMemcpy2D between two devices, using streams
void run_memcpy_async(const int N, const std::vector<float*>& in_bufs, std::vector<float*>& out_bufs, const std::vector<hipStream_t>& streams)
{
    return;
}

// (2) Copy kernel
__global__ void copy(const int N, const float* input, float* output) 
{
    
}

// (3) MPI alltoall

// (4) RCCL alltoall


// Check equality of matrices
bool is_same_matrix(const int N, float* input1, float* input2)
{
    for (auto i = 0; i < N; i++)
    {
        if(input1[i] != input2[i]) return false;
    }
    return true;
}

// Reference impl (out-of-place)
void host_transpose(const std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output)
{
    const size_t N = input[0].size();
    output.reserve(N*N);
    for(size_t i = 0; i < N; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            output[j][i] = input[i][j];
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

    std::vector<std::vector<float>> input(N, std::vector<float>(N));
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j < N; ++j)
        input[i][j] = dist(m_engine);

    // Split input and transfer it
    // Assume inputs are evenly divisible :)
    std::vector<float*> gpubufs_input(ngpus);
    std::vector<float*> gpubufs_output(ngpus);
    std::vector<hipStream_t> streams(ngpus);

    const size_t buf_height = N / ngpus;
    const size_t buf_size = N * buf_height;
    // Allocate and init bufs, streams
    for(size_t i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipMalloc(&gpubufs_input[i], sizeof(float) * buf_size)); 
        HIP_CHECK(hipMemcpy2D(&gpubufs_input[i], 1, &(input.data() + ngpus * buf_size), 1, N, buf_height, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&gpubufs_output[i], sizeof(float) * buf_size));
    }

    // -- Run stuff --
    std::vector<std::vector<float>> reference_matrix(N, std::vector<float>(N));
    host_transpose(input, reference_matrix);
    
    bool res;
    run_memcpy(N, gpubufs_input, gpubufs_output);
    res = is_same_matrix();

    // Copy kernel
    // MPI alltoall
    // RCCL alltoall
    

}

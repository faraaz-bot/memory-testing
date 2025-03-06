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

// (1) hipMemcpy2D between two devices
void run_memcpy(const int N, const std::vector<float*>&)
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
bool is_same_matrix()

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

    std::vector<std::vector<float>> input(N);
    for(size_t i = 0; i < N; ++i)
        for(size_t j = 0; j < N; ++j)
        input[i][j] = dist(m_engine);

    // Split input and transfer it

    // -- Run stuff --
    // hipMemcpy2D


    // Copy kernel
    // MPI alltoall
    // RCCL alltoall
    

    // Check for correctness of result(s)
}

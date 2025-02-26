#include "mem-bench.hpp"
#include "../../eg/argv/CLI11.hpp"

#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <random>
#include <stdio.h>
#include <vector>

/**
 * Benchmarking tool for comparing speed of various memory copy methods
*/

// hipMemcpyAsync

// hipMemcpyAsync2D

// MPI alltoall

// Copy kernel

int main(int argc, char* argv[])
{
    CLI::App app{"Memcpy bench"};

    // Declare the supported options. Some option pointers are declared to track passed opts.
    // app.add_flag("--version", "Print queryable version information from the rocfft library")
    //     ->each([](const std::string&) {
    //         char v[256];
    //         rocfft_get_version_string(v, 256);
    //         std::cout << "version " << v << std::endl;
    //         return EXIT_SUCCESS;
    //     });

    // CLI::Option* opt_token
    //     = app.add_option("--token", token, "Token to read FFT params from")->default_val("");

    // std::vector<size_t> lengths(3);
    const size_t N = 1000;

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
    std::random_device                    rd;
    std::mt19937                          m_engine(rd()); // Mersenne Twister, rd as seed
    std::uniform_real_distribution<float> dist{-0.5, 0.5};

    std::vector<float> input(N);
    for(size_t i = 0; i < N; i++)
        input[i] = dist(m_engine);

    // Run stuff
}

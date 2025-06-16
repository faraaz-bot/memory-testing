#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <set>

#include "../../../eg/argv/CLI11.hpp"
#include "../src/membench.hpp"

// struct test_params
// {
// }

class MembenchTest : public ::testing::TestWithParam<std::tuple<size_t, precision>>
{
protected:
    // void SetUp() override {}
    // void TearDown() override {}
    // bool verifyBlockTranspose()
    // {
    //     return true;
    // }
};

TEST_P(MembenchTest, Memcpy2D)
{
    size_t    n = std::get<size_t>(GetParam());
    precision p = std::get<precision>(GetParam());
    // Switch on p...
}

// Provide list of lengths + types to run with
INSTANTIATE_TEST_SUITE_P(
    MembenchTests,
    MembenchTest,
    ::testing::Combine(
        ::testing::Values(
            8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u),
        ::testing::Values(precision::p_single, precision::p_double)));

int main(int argc, char* argv[])
{
    // Note: also edit map in add_benchmarks() if editing this set
    std::set<std::string> valid_benchmarks = {"all",
                                              "memcpy2D",
                                              "memcpy2DAsync",
                                              "naiveCopy",
                                              "localTranspose",
                                              "naiveCopy+Transpose",
                                              "memcpy2D+Transpose",
                                              "memcpy2DAsync+Transpose"};
    // Parse args
    CLI::App app{"Memcpy bench"};

    std::string run_bench_helper
        = "Benchmarks to run, i.e: --runBenchmark memcpy2D "
          "memcpy2DAsync\n\nAvailable Benchmarks:\n------------------------\n";

    for(const auto& x : valid_benchmarks)
        run_bench_helper += x + "\n";

    run_bench_helper += "------------------------\n";

    benchmark_context     ctx;
    size_t                trials;
    std::set<std::string> param_enabled_benchmarks;
    app.add_option("-n, --length", ctx.N, "Length of input square matrix")->default_val(8U);
    app.add_option("-g, --ngpus", ctx.ngpus, "Number of gpus")
        ->default_val(4U)
        ->check(CLI::PositiveNumber);
    app.add_option("-v, --verbose",
                   ctx.verbose,
                   "Adjust output verbosity level\n1) Basic benchmark details\n2) Matrix data\n3) "
                   "Initial buffer data")
        ->default_val(0);

    app.add_flag(
        "-c, --verify", ctx.verify_results, "Toggle correctness checks performed after each trial");
    app.add_option(
           "-t, --trials", trials, "The amount of minimum trials to run per function (default 20)")
        ->default_val(20);
    app.add_option("-f, --filter", param_enabled_benchmarks, run_bench_helper)->default_val("all");

    precision p;
    generator gen;
    double    min_val;
    double    max_val;
    app.add_option("-p, --precision", p, "Data precision: single (default), double")
        ->default_val("single");
    app.add_option("-i, --inputGen",
                   gen,
                   "Data generation type:\n0) random (default)\n1) ordered (linear sequence)")
        ->default_val(0);
    app.add_option("--min", min_val, "Minimum value to use if generating random input")
        ->default_val(-1.0);
    app.add_option("--max", max_val, "Maximum value to use if generating random input")
        ->default_val(1.0);

    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    // Check if inputs are valid for benchmark
    if(!(is_power_of_two(ctx.N)) || !(is_power_of_two(ctx.ngpus)))
        throw std::runtime_error("N and ngpus should both be powers of two");
    const size_t N = ctx.N;

    if(ctx.verbose)
        std::cout << "Comparing on " << N << " x " << N << " size matrix, across " << ctx.ngpus
                  << " gpus." << std::endl;

    // Enable peer to peer memory access between GPUs
    for(size_t i = 0; i < ctx.ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        for(size_t j = 0; j < ctx.ngpus; j++)
        {
            int can_access_peer;
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, i, j));
            if(can_access_peer)
                HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
        }
    }

    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();
    return retval;
}

#include <benchmark/benchmark.h>
#include <limits.h>
#include <random>
#include <string>
#include <vector>

#include "../../eg/argv/CLI11.hpp"
#include "src/mem-bench.hpp"

template <typename T>
void run_benchmark(benchmark::State& state, const size_t trials, const size_t N)
{
    std::vector<std::vector<T>> temp(N, std::vector<T>(N, 0));

    std::random_device rd;
    std::mt19937       m_engine(rd()); // Mersenne Twister, rd as seed

    T mini = std::numeric_limits<T>::min();
    T maxi = std::numeric_limits<T>::max();

    std::uniform_real_distribution<double> dist{static_cast<double>(mini),
                                                static_cast<double>(maxi)};

    hipStream_t stream;
    hipEvent_t  start, stop;

    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t __ = 0; __ < trials; __++)
        {

            for(size_t i = 0; i < N; i++)
            {
                for(size_t j = 0; j < N; j++)
                    temp[i][j] = static_cast<T>(dist(m_engine));
            }
        }

        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        state.SetIterationTime(elapsed / 1000.f);
    }

    state.counters["Throughput (GB/S)"]
        = static_cast<double>((trials * state.iterations() * N * N * sizeof(T)))
          / static_cast<double>((1024 * 1024 * 1024));

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
}

int main(int argc, char* argv[])
{
    CLI::App app{"Memcpy bench"};

    size_t    N;
    size_t    ngpus;
    int       verbose;
    size_t    trials;
    precision p;
    generator gen;
    app.add_option("-n, --length", N, "Length of input square matrix")->default_val(8U);
    app.add_option("-g, --ngpus", ngpus, "Number of gpus")->default_val(4U);
    app.add_option("-p, --precision", p, "Data precision: single (default), double")
        ->default_val("single");
    app.add_option(
           "-i, --inputGen", gen, "Data generation type:\n0) random (default)\n1) ordered sequence")
        ->default_val(0);
    app.add_option("-v, --verbose", verbose, "Adjust output verbosity level")->default_val(0);
    app.add_option(
           "-t, --trials", trials, "The amount of minimum trials to run per function (default 20)")
        ->default_val(20);

    // TODO option: which benchmark(s) to run, output format options

    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    std::vector<char*> cArgs(argv, argv + argc);

    std::string tabular = "--benchmark_counters_tabular=true";

    cArgs.push_back(tabular.data());

    char** cArga     = cArgs.data();
    int    cArg_size = cArgs.size();

    // TODO Better way of handling benchmark args at same time as CLI11?
    // std::cout << "Before:\n";
    // for(auto& i : cArgs)
    //     std::cout << i << " ";
    // std::cout << std::endl;

    benchmark::Initialize(&cArg_size, cArga);

    // std::cout << "After:\n";
    // for(auto& i : cArgs)
    //     std::cout << i << " ";
    // std::cout << std::endl;

    std::cout << "Comparing on " << N << " x " << N << " size matrix, across " << ngpus
              << " gpus.\n";

    // Generate input data
    std::vector<float> input(N * N);
    if(gen == h_random)
    {
        std::random_device                    rd;
        std::mt19937                          m_engine(rd()); // Mersenne Twister, rd as seed
        std::uniform_real_distribution<float> dist{-0.5, 0.5};
#pragma omp parallel for
        for(size_t i = 0; i < N * N; ++i)
            input[i] = dist(m_engine);
    }
    else if(gen == h_ordered)
    {
        for(size_t i = 0; i < N * N; i++)
            input[i] = i;
    }

    if(verbose)
    {
        std::cout << "Input Matrix:\n";
        print_host_2d<float>(N, N, input);
    }

    // Compute host-side transposed matrix for correctness check
    std::vector<float> reference_matrix(N * N);
    host_transpose(N, input, reference_matrix);

    if(verbose > 1)
    {
        std::cout << "Host Transposed Matrix:\n";
        print_host_2d<float>(N, N, reference_matrix);
    }

    // Split input and transfer it
    // Assume inputs are evenly divisible :)
    std::vector<float*>      gpubufs_input(ngpus);
    std::vector<float*>      gpubufs_output(ngpus);
    std::vector<hipStream_t> streams(ngpus * ngpus);

    // Allocate and init bufs, streams
    setup<float>(N, ngpus, gpubufs_input, gpubufs_output, input, streams);

    // Optionally output gpu bufs after distributing data
    if(verbose > 3)
    {
        const size_t buf_height = N / ngpus;
        const size_t buf_size   = N * buf_height;
        for(auto i = 0; i < ngpus; i++)
        {
            std::cout << "Input GPU Buffer " << i << ":\n";
            print<float><<<1, 1>>>(buf_size, gpubufs_input[i]);
            print2d<float><<<1, 1>>>(N, buf_height, gpubufs_input[i]);
        }
    }

    // Enable peer to peer memory access between GPUs
    for(size_t i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        for(size_t j = 0; j < ngpus; j++)
        {
            int can_access_peer;
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, i, j));
            if(can_access_peer)
                HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
        }
    }

    // -- Run stuff --
    // TODO register benchmark, and move this stuff there
    std::vector<float> h_assembled_output(N * N);
    run_memcpy<float>(N, gpubufs_input, gpubufs_output);
    assemble_output_to_host<float>(N, gpubufs_output, h_assembled_output.data());
    // bool res = is_same_matrix<float>(N, reference_matrix, h_assembled_output);
    if(verbose)
    {
        // std::cout
        //     << "Are two matrices equal? " << res
        //     << "\nOutput Assembled on Host:\n"; // Currently should not, due to lack of local transpose! *unless ngpus = n
        std::cout << "Output Assembled on Host:\n";
        print_host_2d<float>(N, N, h_assembled_output);
    }

    // Implement cleanup -> fill/memset existing bufs with 0?
    reset<float>(N, ngpus, gpubufs_output, h_assembled_output);

    // Copy kernel
    // MPI alltoall
    // RCCL alltoall

    // Setup implementations to run in gbenchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};

    benchmarks.emplace_back(
        benchmark::RegisterBenchmark("test_bench", &run_benchmark<unsigned int>, trials, N));
    benchmarks.emplace_back(
        benchmark::RegisterBenchmark("test_bench2", &run_benchmark<unsigned int>, trials, N));

    // Free up buffers, streams
    teardown<float>(ngpus, gpubufs_input, gpubufs_output, streams);

    // Disabling peer access
    // for(size_t i = 0; i < ngpus; i++)
    // {
    //     b->UseManualTime();
    //     b->Unit(benchmark::kMillisecond);
    // }

    static benchmark::ConsoleReporter terminal_reporter;
    terminal_reporter.SetErrorStream(&std::cout);
    terminal_reporter.SetOutputStream(&std::cout);

    benchmark::RunSpecifiedBenchmarks();
}

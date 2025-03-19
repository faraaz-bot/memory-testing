#include <benchmark/benchmark.h>
#include <limits.h>
#include <string>
#include <vector>

#include "../../eg/argv/CLI11.hpp"
#include "src/mem-bench.hpp"

using func_type = std::function<void(
    const benchmark_context&, const std::vector<float*>&, std::vector<float*>&)>;

template <typename T>
void run_benchmark(benchmark::State&     state,
                   benchmark_context     ctx,
                   const size_t          trials,
                   const std::vector<T>& h_input,
                   func_type             f)
{
    const size_t N       = ctx.N;
    const size_t ngpus   = ctx.ngpus;
    int          verbose = ctx.verbose;
    ctx.streams          = std::vector<hipStream_t>(ngpus * ngpus);

    // Initialize and copy data over (currently assume input is evenly divisible over ngpus)
    std::vector<float*> gpubufs_input(ngpus);
    std::vector<float*> gpubufs_output(ngpus);

    // Compute host-side transposed matrix for correctness check
    std::vector<float> reference_matrix(N * N);
    host_transpose(N, h_input, reference_matrix);

    if(verbose > 1)
    {
        std::cout << "Host Transposed Matrix:\n";
        print_host_2d<float>(N, N, reference_matrix);
    }

    // Allocate and init bufs, streams
    setup<T>(ctx.N, ngpus, gpubufs_input, gpubufs_output, h_input, ctx.streams);

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

    // std::vector<float> h_assembled_output(N * N);
    // assemble_output_to_host<float>(N, gpubufs_output, h_assembled_output.data());
    // bool res = is_same_matrix<float>(N, reference_matrix, h_assembled_output);
    // if(verbose)
    // {
    //     std::cout << "Output Assembled on Host:\n";
    //     print_host_2d<float>(N, N, h_assembled_output);
    // }

    // Setup timing events
    hipStream_t timing_stream;
    hipEvent_t  start, stop;
    HIP_CHECK(hipStreamCreate(&timing_stream));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state)
    {
        HIP_CHECK(hipEventRecord(start, timing_stream));

        for(size_t __ = 0; __ < trials; __++)
        {
            f(ctx, gpubufs_input, gpubufs_output);
        }

        HIP_CHECK(hipEventRecord(stop, timing_stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        state.SetIterationTime(elapsed_ms / 1000.f);

        // reset<float>(N, ngpus, gpubufs_output, h_assembled_output);
    }

    state.counters["Throughput (GB/S)"]
        = static_cast<double>((trials * state.iterations() * N * N * sizeof(T)))
          / static_cast<double>((1024 * 1024 * 1024));

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    teardown<T>(ngpus, gpubufs_input, gpubufs_output, ctx.streams);
}

int main(int argc, char* argv[])
{
    CLI::App app{"Memcpy bench"};

    benchmark_context ctx;
    size_t            trials;
    app.add_option("-n, --length", ctx.N, "Length of input square matrix")->default_val(8U);
    app.add_option("-g, --ngpus", ctx.ngpus, "Number of gpus")->default_val(4U);
    app.add_option("-v, --verbose", ctx.verbose, "Adjust output verbosity level")->default_val(0);
    app.add_option(
           "-t, --trials", trials, "The amount of minimum trials to run per function (default 20)")
        ->default_val(20);

    precision p;
    generator gen;
    float     min_val;
    float     max_val;
    app.add_option("-p, --precision", p, "Data precision: single (default), double")
        ->default_val("single");
    app.add_option(
           "-i, --inputGen", gen, "Data generation type:\n0) random (default)\n1) ordered sequence")
        ->default_val(0);
    app.add_option("--min", min_val, "Minimum value to use if generating random input")
        ->default_val(-1.0f);
    app.add_option("--max", max_val, "Maximum value to use if generating random input")
        ->default_val(1.0f);

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

    const size_t N = ctx.N;

    if(ctx.verbose)
        std::cout << "Comparing on " << N << " x " << N << " size matrix, across " << ctx.ngpus
                  << " gpus.\n";

    std::vector<char*> cArgs(argv, argv + argc);

    std::string tabular = "--benchmark_counters_tabular=true";

    cArgs.push_back(tabular.data());

    char** cArga     = cArgs.data();
    int    cArg_size = cArgs.size();

    // TODO Better way of handling benchmark args at same time as CLI11?
    // If gbench removes args, then we can allow extras then check leftovers later...
    // std::cout << "Before:\n";
    // for(auto& i : cArgs)
    //     std::cout << i << " ";
    // std::cout << std::endl;

    // std::cout << "After:\n";
    // for(auto& i : cArgs)
    //     std::cout << i << " ";
    // std::cout << std::endl;

    // Generate input data
    auto h_input = generate(N, N, gen, min_val, max_val);

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

    // Setup implementations to run in gbenchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};

    benchmarks.emplace_back(benchmark::RegisterBenchmark(
        "hipMemcpy2D", &run_benchmark<float>, ctx, trials, h_input, run_memcpy<float>));
    benchmarks.emplace_back(benchmark::RegisterBenchmark(
        "hipMemcpy2DAsync", &run_benchmark<float>, ctx, trials, h_input, run_memcpy_async<float>));

    benchmark::Initialize(&cArg_size, cArga);

    // for(auto& b : benchmarks)
    // {
    //     b->UseManualTime();
    //     b->Unit(benchmark::kMillisecond);
    // }

    static benchmark::ConsoleReporter terminal_reporter;
    terminal_reporter.SetErrorStream(&std::cout);
    terminal_reporter.SetOutputStream(&std::cout);

    benchmark::RunSpecifiedBenchmarks();
}

#include <benchmark/benchmark.h>
#include <limits.h>
#include <set>
#include <string>
#include <vector>

#include "../../eg/argv/CLI11.hpp"
#include "src/mem-bench.hpp"

// TODO how to add template here?
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
    std::vector<T*> gpubufs_input(ngpus);
    std::vector<T*> gpubufs_output(ngpus);

    // Compute host-side transposed matrix for correctness check
    std::vector<T> reference_matrix(N * N);
    host_copy(N, ngpus, h_input.data(), reference_matrix.data());

    if(verbose > 1)
    {
        std::cout << "Starting Input Matrix:\n";
        print_host_2d<T>(N, N, h_input);
        std::cout << "Host Reference Matrix:\n";
        print_host_2d<T>(N, N, reference_matrix);
    }

    // Allocate and init bufs, streams
    setup<T>(ctx.N, ngpus, gpubufs_input, gpubufs_output, h_input, ctx.streams);

    // Optionally output gpu bufs after distributing data
    if(verbose > 3)
    {
        const size_t buf_height = N / ngpus;
        for(auto i = 0; i < ngpus; i++)
        {
            std::cout << "Input GPU Buffer " << i << ":\n";
            // print<T><<<1, 1>>>(buf_size, gpubufs_input[i]);
            print2d<T><<<1, 1>>>(N, buf_height, gpubufs_input[i]);
        }
    }

    // Setup timing events
    hipStream_t timing_stream;
    hipEvent_t  start, stop;
    HIP_CHECK(hipStreamCreate(&timing_stream));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    float          total_ms = 0.0f;
    std::vector<T> h_assembled_output(N * N);
    for(auto _ : state)
    {
        for(size_t __ = 0; __ < trials; __++)
        {
            HIP_CHECK(hipEventRecord(start, timing_stream));
            f(ctx, gpubufs_input, gpubufs_output);
            HIP_CHECK(hipEventRecord(stop, timing_stream));
            HIP_CHECK(hipEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
            total_ms += elapsed_ms;

            // Optionally
            if(ctx.verify_results)
            {
                assemble_output_to_host<T>(N, gpubufs_output, h_assembled_output.data());
                bool res = is_same_matrix<T>(N, reference_matrix, h_assembled_output);
                if(!res)
                {
                    std::cout << "Incorrect result detected for " << state.name() << "\n";
                    std::cout << "Host Side Computation:\n";
                    print_host_2d<T>(N, N, reference_matrix);
                    std::cout << "----------------------\nDevice Side Computation:\n";
                    print_host_2d<T>(N, N, h_assembled_output);
                }
            }
            reset<T>(N, ngpus, gpubufs_output, h_assembled_output);
        }
    }

    state.SetIterationTime(total_ms / 1000.f);
    double bytesProcessed = trials * state.iterations() * N * N * sizeof(T);
    state.counters["Throughput (GB/s)"]
        = benchmark::Counter(bytesProcessed / (1024 * 1024 * 1024), benchmark::Counter::kIsRate);

    // Clean up
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    teardown<T>(ngpus, gpubufs_input, gpubufs_output, ctx.streams);
}

int main(int argc, char* argv[])
{
    CLI::App app{"Memcpy bench"};

    std::set<std::string> valid_benchmarks = {"all", "hipMemcpy2D", "hipMemcpy2DAsync"};

    std::string run_bench_helper
        = "Benchmarks to run, i.e: --run-benchmark hipMemcpy2D "
          "hipMemcpy2DAsync\n\nAvailable Benchmarks:\n------------------------\n";

    for(const auto& x : valid_benchmarks)
        run_bench_helper += x + "\n";

    run_bench_helper += "------------------------\n";

    benchmark_context     ctx;
    size_t                trials;
    std::set<std::string> param_enabled_benchmarks;
    app.add_option("-n, --length", ctx.N, "Length of input square matrix")->default_val(8U);
    app.add_option("-g, --ngpus", ctx.ngpus, "Number of gpus")->default_val(4U);
    app.add_option("-v, --verbose", ctx.verbose, "Adjust output verbosity level")->default_val(0);
    app.add_option("-c, --verify",
                   ctx.verify_results,
                   "Toggle correctness checks performed after each trial")
        ->default_val(true);
    app.add_option(
           "-t, --trials", trials, "The amount of minimum trials to run per function (default 20)")
        ->default_val(20);
    app.add_option("--run-benchmark", param_enabled_benchmarks, run_bench_helper)
        ->default_val("all");

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

    std::vector<std::string> enabled_benchmarks;
    bool                     runAll = false;

    for(auto it = param_enabled_benchmarks.begin(); it != param_enabled_benchmarks.end(); it++)
    {
        if(valid_benchmarks.find(*it) == valid_benchmarks.end())
            std::cout << *it << " is not a valid benchmark. It has been discarded!" << std::endl;
        else
        {
            if(*it == "all")
            {
                runAll = true;
                break;
            }

            enabled_benchmarks.push_back(*it);
        }
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
    std::vector<float> h_input = generate<float>(N, N, gen, min_val, max_val);

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

    if(runAll)
    {
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            "hipMemcpy2D", &run_benchmark<float>, ctx, trials, h_input, run_memcpy<float>));
        benchmarks.emplace_back(benchmark::RegisterBenchmark("hipMemcpy2DAsync",
                                                             &run_benchmark<float>,
                                                             ctx,
                                                             trials,
                                                             h_input,
                                                             run_memcpy_async<float>));
    }
    else
    {
        for(const auto& x : enabled_benchmarks)
        {
            if(x == "hipMemcpy2D")
                benchmarks.emplace_back(benchmark::RegisterBenchmark(
                    "hipMemcpy2D", &run_benchmark<float>, ctx, trials, h_input, run_memcpy<float>));

            else if(x == "hipMemcpy2DAsync")
                benchmarks.emplace_back(benchmark::RegisterBenchmark("hipMemcpy2DAsync",
                                                                     &run_benchmark<float>,
                                                                     ctx,
                                                                     trials,
                                                                     h_input,
                                                                     run_memcpy_async<float>));
        }
    }

    benchmark::Initialize(&cArg_size, cArga);

    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kSecond);
    }

    static benchmark::ConsoleReporter terminal_reporter;
    terminal_reporter.SetErrorStream(&std::cout);
    terminal_reporter.SetOutputStream(&std::cout);

    benchmark::RunSpecifiedBenchmarks();
}

#include <benchmark/benchmark.h>
#include <limits.h>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../eg/argv/CLI11.hpp"
#include "src/membench.hpp"

/**
 * Benchmarking tool for comparing speed of various memory copy methods
 * between multiple gpus. Currently will do out-of-place operations on 
 * square matrices only.
 *
 * MPI-specific benchmarks are handled with mpi-main.cpp & mpi-membench.hpp.
 */

template <typename T>
using benchmark_fn = std::function<float(const benchmark_context&, gpubuf_vec<T>&, gpubuf_vec<T>&)>;

// Execute f under Google Benchmark, for at least trials times
// Manages device memory management, timing, and verification,
// but not generating initial input data (h_input).
template <typename T>
void run_benchmark(benchmark::State&     state,
                   benchmark_context     ctx,
                   const size_t          trials,
                   const std::vector<T>& h_input,
                   benchmark_fn<T>       f)
{
    const size_t N          = ctx.N;
    const size_t ngpus      = ctx.ngpus;
    int          verbose    = ctx.verbose;
    std::string  bench_name = state.name();

    // Initialize and copy data over (currently assume input is evenly divisible over ngpus)
    gpubuf_vec<T>  gpubufs_input(N, ngpus);
    gpubuf_vec<T>  gpubufs_output(N, ngpus);
    std::vector<T> h_assembled_output(N * N);
    ctx.streams = std::vector<hipStream_t>(ngpus * ngpus);

    // Compute host-side matrix for correctness check
    // Can be either block transposed or fully transposed result
    std::vector<T> reference_matrix(N * N);

    // Allocate and init bufs, streams
    setup<T>(ctx.N, ngpus, gpubufs_input, gpubufs_output, h_input, ctx.streams);

    // Optionally output gpu bufs after distributing data
    if(verbose > 2)
    {
        const size_t buf_height = N / ngpus;
        for(auto i = 0; i < ngpus; i++)
        {
            std::cout << "Input GPU Buffer " << i << ":\n";
            print2d<T><<<1, 1>>>(buf_height, N, gpubufs_input[i]);
            HIP_CHECK(hipDeviceSynchronize());
        }
    }

    // Execute and time the benchmarks
    float total_ms = 0.0f;
    for(auto _ : state)
    {
        for(size_t t = 0; t < trials; t++)
        {
            total_ms += f(ctx, gpubufs_input, gpubufs_output);

            for(auto i = 0; i < ngpus; i++)
            {
                HIP_CHECK(hipSetDevice(i));
                HIP_CHECK(hipDeviceSynchronize());
            }

            // Set output buffers back to all 0s
            reset<T>(N, ngpus, gpubufs_output, h_assembled_output);
        }
    }

    state.SetIterationTime(total_ms / 1000.f);
    double bytesProcessed = trials * state.iterations() * N * N * sizeof(T);
    state.counters["Throughput (GB/s)"]
        = benchmark::Counter(bytesProcessed / (1024 * 1024 * 1024), benchmark::Counter::kIsRate);

    state.counters["Dimension (N x N)"] = benchmark::Counter(N);
    state.counters["Device Count"]      = benchmark::Counter(ngpus);

    teardown<T>(ngpus, gpubufs_input, gpubufs_output, ctx.streams);
}

// Register all (valid) provided functions to run as benchmarks
template <typename T>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const benchmark_context&                      ctx,
                    size_t                                        trials,
                    const std::vector<T>&                         h_input,
                    const std::set<std::string>&                  enabled_benchmarks)
{
    // Add benchmarks here
    // Note: also modify set in main() with new strings
    std::unordered_map<std::string, benchmark_fn<T>> all_benchmarks
        = {{"localTranspose", local_transpose_launcher<T>},
           {"memcpy2D", run_memcpy<T>},
           {"memcpy2D+Transpose", run_memcpy_transpose<T>},
           {"memcpy2DAsync", run_memcpy_async<T>},
           {"memcpy2DAsync+Transpose", run_memcpy_async_transpose<T>},
           {"naiveCopy", naive_copy_launcher<T>},
           {"naiveCopy+Transpose", naive_copy_transpose<T>}};
    // {"naiveCopy+FusedTranspose", naive_copy_transpose<T>},
    // {"ldsCopy", naive_copy_launcher<T>},

    bool run_all = enabled_benchmarks.count("all");
    for(const auto& kv : all_benchmarks)
    {
        if(run_all || enabled_benchmarks.count(kv.first))
            benchmarks.emplace_back(benchmark::RegisterBenchmark(
                kv.first, &run_benchmark<T>, ctx, trials, h_input, kv.second));
    }
}

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

    /*clang format off*/
    std::string gbench_options
        = std::string("Google Benchmark Options:\n\n") + std::string("--benchmark_filter=<regex>\n")
          + std::string("\tFilters out which benchmarks to run,                         i.e: "
                        "./membench --benchmark_filter=2D\n")
          + std::string("--benchmark_min_time=`<integer>x` OR `<float>s`\n")
          + std::string("\tSets the minimum amount of time each benchmark has to run,   i.e: "
                        "./membench --benchmark_min_time=10s\n")
          + std::string("--benchmark_format=<json|console|csv>\n")
          + std::string("\tSets the display format on the terminal (default console),   i.e: "
                        "./membench --benchmark_format=csv\n")
          + std::string("--benchmark_out=<filename>\n")
          + std::string("\tStore the output to filename,                                i.e: "
                        "./membench --benchmark_out=./sample.csv\n")
          + std::string("--benchmark_out_format=<json|console|csv>\n")
          + std::string("\tSet the display format on the output file (default console), i.e: "
                        "./membench --benchmark_out_format=csv\n\n")
          + std::string("--benchmark_list_tests={true|false}\n")
          + std::string("--benchmark_min_warmup_time=<min_warmup_time>\n")
          + std::string("--benchmark_repetitions=<num_repetitions>\n")
          + std::string("--benchmark_dry_run={true|false}\n")
          + std::string("--benchmark_enable_random_interleaving={true|false}\n")
          + std::string("--benchmark_report_aggregates_only={true|false}\n")
          + std::string("--benchmark_display_aggregates_only={true|false}\n")
          + std::string("--benchmark_format=<console|json|csv>\n")
          + std::string("--benchmark_color={auto|true|false}\n")
          + std::string("--benchmark_counters_tabular={true|false}\n")
          + std::string("--benchmark_context=<key>=<value>,...\n")
          + std::string("--benchmark_time_unit={ns|us|ms|s}\n") + std::string("--v=<verbosity>");
    /*clang format on*/

    app.footer(gbench_options.c_str());

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

    std::set<std::string> enabled_benchmarks;

    // Validate benchmarks to run, from command line arg data
    for(auto it = param_enabled_benchmarks.begin(); it != param_enabled_benchmarks.end(); it++)
    {
        if(valid_benchmarks.find(*it) == valid_benchmarks.end())
            std::cout << *it << " is not a valid benchmark. It has been discarded!" << std::endl;
        {
            enabled_benchmarks.insert(*it);
        }
    }

    const size_t N = ctx.N;
    if(ctx.verbose)
        std::cout << "Comparing on " << N << " x " << N << " size matrix, across " << ctx.ngpus
                  << " gpus." << std::endl;

    // TODO Better way of handling benchmark args at same time as CLI11?
    // If gbench removes args, then we can allow extras then check leftovers later...

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

    // Generate input data and register benchmarks based on precision
    switch(p)
    {
    case precision::p_single:
        add_benchmarks<float>(benchmarks,
                              ctx,
                              trials,
                              generate<float>(N, N, gen, min_val, max_val),
                              enabled_benchmarks);
        break;
    case precision::p_double:
        add_benchmarks<double>(benchmarks,
                               ctx,
                               trials,
                               generate<double>(N, N, gen, min_val, max_val),
                               enabled_benchmarks);
        break;
        // TODO Complex valued cases
    case precision::p_complex_single:
        break;
    case precision::p_complex_double:
        break;
    }

    std::vector<char*> cArgs(argv, argv + argc);

    // Default benchmark args
    std::string tabular = "--benchmark_counters_tabular=true";
    cArgs.push_back(tabular.data());
    std::string default_min_time = "--benchmark_min_time=0s";
    cArgs.push_back(default_min_time.data());

    char** cArga     = cArgs.data();
    int    cArg_size = cArgs.size();
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

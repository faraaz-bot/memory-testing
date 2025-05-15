#include "../../eg/argv/CLI11.hpp"
#include "src/mem-bench.hpp"
#include "src/mpi-helper.hpp"
#include <benchmark/benchmark.h>
#include <mpi.h>
#include <vector>

/**
 * Benchmarking tool for comparing speed of various memory copy methods
 * between multiple gpus. Currently will do out-of-place operations on 
 * square matrices only.
 */

// Execute f under Google Benchmark, for at least trials times
// Manages device memory management, timing, and verification,
// but not generating initial input data (h_input).
template <typename T>
void run_benchmark(
    benchmark::State&                                                                  state,
    benchmark_context                                                                  ctx,
    const size_t                                                                       trials,
    const std::vector<T>&                                                              h_input,
    std::function<float(const benchmark_context&, std::vector<T*>&, std::vector<T*>&)> f)
{
    const size_t N            = ctx.N;
    const size_t ngpus        = ctx.ngpus;
    int          verbose      = ctx.verbose;
    std::string  bench_name   = state.name();
    bool         is_transpose = bench_name.find("Transpose") != std::string::npos;

    // TODO Each process has its own buffer, no longer a vector
    // Initialize and copy data over (currently assume input is evenly divisible over ngpus)
    std::vector<T*> gpubufs_input(ngpus);
    std::vector<T*> gpubufs_output(ngpus);

    // Compute host-side matrix for correctness check
    // Can be either block transposed or fully transposed result
    std::vector<T> h_assembled_output(N * N);
    std::vector<T> reference_matrix(N * N);
    if(ctx.verify_results)
    {
        if(is_transpose)
            host_transpose<T>(N, h_input.data(), reference_matrix.data());
        else
            host_copy<T>(N, ngpus, h_input.data(), reference_matrix.data());
    }

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
    float  total_ms     = 0.0f;
    size_t num_failures = 0;
    size_t num_pass     = 0;
    size_t total_runs   = 0;
    for(auto _ : state)
    {
        for(size_t t = 0; t < trials; t++)
        {
            float ms = f(ctx, gpubufs_input, gpubufs_output);

            // Get max time across all ranks (note: only rank 0 will report benchmark results)
            MPI_Reduce(static_cast<void*>(&ms),
                       static_cast<void*>(total_ms),
                       1,
                       MPI_FLOAT,
                       MPI_MAX,
                       0,
                       MPI_COMM_WORLD);

            // Optionally confirm correctness by copying output back and comparing to host-side computation
            if(ctx.verify_results)
            {
                // TODO MPI_Gather instead of assemble_output_to_host()
                assemble_output_to_host<T>(
                    N, ngpus, gpubufs_output.data(), h_assembled_output.data());
                bool res = is_same_matrix<T>(N, reference_matrix, h_assembled_output);
                if(!res)
                {
                    num_failures++;
                    std::cout << "Incorrect result detected for " << state.name() << ", trial #"
                              << t << "\n";
                    if(verbose)
                    {
                        std::cout << "Original Input:\n";
                        print_host_2d<T>(N, N, h_input);
                        std::cout << "Host Side Computation:\n";
                        print_host_2d<T>(N, N, reference_matrix);
                        std::cout << "----------------------\nDevice Side Computation:\n";
                        print_host_2d<T>(N, N, h_assembled_output);
                    }
                }
                else
                {
                    num_pass++;
                }
                total_runs++;
            }
            else
            {
                if(verbose > 1)
                {
                    assemble_output_to_host<T>(
                        N, ngpus, gpubufs_output.data(), h_assembled_output.data());
                    bool res = is_same_matrix<T>(N, reference_matrix, h_assembled_output);
                    if(!res)
                    {
                        std::cout << "Original Input:\n";
                        print_host_2d<T>(N, N, h_input);
                        std::cout << "------------------------\nDevice Side Computation:\n";
                        print_host_2d<T>(N, N, h_assembled_output);
                    }
                }
            }
            // Set output buffers back to all 0s
            reset<T>(N, ngpus, gpubufs_output, h_assembled_output);
        }
    }

    if(ctx.verify_results)
        std::cout << num_pass << "/" << total_runs << " runs passed. " << num_failures
                  << " runs failed." << std::endl;

    state.SetIterationTime(total_ms / 1000.f);
    double bytesProcessed = trials * state.iterations() * N * N * sizeof(T);
    state.counters["Throughput (GB/s)"]
        = benchmark::Counter(bytesProcessed / (1024 * 1024 * 1024), benchmark::Counter::kIsRate);

    state.counters["Dimension (N x N)"] = benchmark::Counter(N);
    state.counters["Device Count"]      = benchmark::Counter(ngpus);

    teardown<T>(ngpus, gpubufs_input, gpubufs_output, ctx.streams);
}

template <typename T>
using benchmark_fn
    = std::function<float(const benchmark_context&, std::vector<T*>&, std::vector<T*>&)>;

// Register all (valid) provided functions to run as benchmarks
template <typename T>
void add_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const benchmark_context&                      ctx,
                    size_t                                        trials,
                    const std::vector<T>&                         h_input,
                    const std::set<std::string>&                  enabled_benchmarks)
{
    // Add benchmarks here
    std::unordered_map<std::string, benchmark_fn<T>> all_benchmarks = {{"mpiCopy", mpi_copy<T>}};

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
    std::set<std::string> valid_benchmarks = {"all", "mpiCopy"};

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_ARE_FATAL);
    int mpi_rank = 0;
    int mp_size;

    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mp_size);

    // Parse args
    CLI::App app{"Memcpy bench"};

    std::string run_bench_helper
        = "Benchmarks to run, i.e: --runBenchmark mpiCopy "
          "mpiCopy+Transpose\n\nAvailable Benchmarks:\n------------------------\n";

    for(const auto& x : valid_benchmarks)
        run_bench_helper += x + "\n";

    run_bench_helper += "------------------------\n";

    benchmark_context     ctx;
    size_t                trials;
    std::set<std::string> param_enabled_benchmarks;
    app.add_option("-n, --length", ctx.N, "Length of input square matrix")->default_val(8U);
    app.add_option("-v, --verbose", ctx.verbose, "Adjust output verbosity level")->default_val(0);
    app.add_flag(
        "-c, --verify", ctx.verify_results, "Toggle correctness checks performed after each trial");
    app.add_option(
           "-t, --trials", trials, "The amount of minimum trials to run per function (default 20)")
        ->default_val(20);
    app.add_option("-r, --runBenchmark", param_enabled_benchmarks, run_bench_helper)
        ->default_val("all");

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
    std::string gtest_options
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

    app.footer(gtest_options.c_str());

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
    if(!(is_power_of_two(ctx.N)))
        throw std::runtime_error("N should be a power of two");

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
        std::cout << "Comparing on " << N << " x " << N << " size matrix, across " << mp_size
                  << " gpus." << std::endl;

    // TODO Better way of handling benchmark args at same time as CLI11?
    // If gbench removes args, then we can allow extras then check leftovers later...
    // Setup implementations to run in gbenchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};

    // Generate input data and register benchmarks based on precision
    switch(p)
    {
    case p_single:
        add_benchmarks<float>(benchmarks,
                              ctx,
                              trials,
                              generate<float>(N, N, gen, min_val, max_val),
                              enabled_benchmarks);
        break;
    case p_double:
        add_benchmarks<double>(benchmarks,
                               ctx,
                               trials,
                               generate<double>(N, N, gen, min_val, max_val),
                               enabled_benchmarks);
        break;
    // TODO Complex valued cases
    case p_complex_single:
        break;
    case p_complex_double:
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

    // Only allow root proc to report if using MPI
    if(mpi_rank == 0)
    {
        std::cout << "Rank 0 is about to run some benchmarks with reporter!" << std::endl;
        benchmark::RunSpecifiedBenchmarks();
    }
    else
    {
        std::cout << "Rank " << mpi_rank << " is about to run some benchmarks with null reporter!"
                  << std::endl;
        NullReporter null_rep;
        benchmark::RunSpecifiedBenchmarks(&null_rep);
    }

    MPI_Finalize();
}

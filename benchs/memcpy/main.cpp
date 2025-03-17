#include <benchmark/benchmark.h>
#include <limits.h>
#include <string>
#include <vector>

#include "mem-bench.hpp"


template <typename T>
void run_benchmark(benchmark::State& state, const size_t trials, const size_t N) {
    std::vector<std::vector<T>> temp(N, std::vector<T>(N, 0));
 
    std::random_device                    rd;
    std::mt19937                          m_engine(rd()); // Mersenne Twister, rd as seed


    T mini = std::numeric_limits<T>::min();
    T maxi = std::numeric_limits<T>::max();

    std::uniform_real_distribution<double> dist{static_cast<double>(mini), static_cast<double>(maxi)};

    hipStream_t stream;
    hipEvent_t start, stop;

    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for(auto _ : state){
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t __ = 0; __ < trials; __++){

            for(size_t i = 0; i < N; i++){
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

    state.counters["Throughput (GB/S)"] = static_cast<double>((trials * state.iterations() * N * N * sizeof(T))) / static_cast<double>((1024 * 1024 * 1024));
    
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));

}

int main(int argc, char* argv[])
{
    std::vector<char *> cArgs(argv, argv + argc);

    std::string tabular = "--benchmark_counters_tabular=true";

    cArgs.push_back(tabular.data());

    char ** cArga = cArgs.data();
    int cArg_size = cArgs.size();


    benchmark::Initialize(&cArg_size, cArga);

    CLI::App app{"Memcpy bench"};

    size_t N;
    size_t ngpus;
    int    verbose;
    size_t trials;
    app.add_option("-n, --length", N, "Length of input square matrix")->default_val(8U);
    app.add_option("-g, --ngpus", ngpus, "Number of gpus")->default_val(4U);
    app.add_option("-v, --verbose", verbose, "Adjust output verbosity level")->default_val(0);
    app.add_option("-t, --trials", trials, "The amount of minimum trials to run per function (default 20)")
    ->default_val(20);

    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    std::vector<benchmark::internal::Benchmark*> benchmarks = {};

    benchmarks.emplace_back(
        benchmark::RegisterBenchmark(
            "test_bench",
            &run_benchmark<unsigned int>,
            trials,
            N
        )
    );

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

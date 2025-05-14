#include <benchmark/benchmark.h>

// Helper class just to avoid benchmark reporting by all MPI ranks
class NullReporter : public benchmark::BenchmarkReporter
{
public:
    NullReporter() {}
    virtual bool ReportContext(const Context&)
    {
        return true;
    }
    virtual void ReportRuns(const std::vector<Run>&) {}
    virtual void Finalize() {}
};

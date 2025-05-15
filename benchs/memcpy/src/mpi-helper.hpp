#include <benchmark/benchmark.h>
#include <mpi.h>

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

inline MPI_Datatype get_mpi_type(size_t elem_size)
{
    MPI_Datatype mpi_type;
    if(elem_size == 4) // Real FP32
        mpi_type = MPI_FLOAT;
    else if(elem_size == 8) // Complex FP32 or Real FP64
        mpi_type = MPI_DOUBLE;
    else if(elem_size == 16) // Complex FP64
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    else
        throw std::runtime_error("Invalid element size for MPI");
    return mpi_type;
}

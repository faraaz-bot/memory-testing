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

// Gather bufs, to only rank 0
// Note: recv_buf is only significant on root (rank 0), so pass nullptr on other ranks
template <typename Tfloat>
inline gpubuf<Tfloat> mpi_gather_buf(gpubuf<Tfloat>& buf, int num_ranks, int rank)
{
    size_t         N        = buf.size();
    MPI_Datatype   mpi_type = get_mpi_type(sizeof(Tfloat));
    gpubuf<Tfloat> recv_buf = (rank == 0) ? gpubuf<Tfloat>(N * num_ranks) : nullptr;
    MPI_Gather(buf, N, mpi_type, recv_buf, N, mpi_type, 0, MPI_COMM_WORLD);
}

// Print out combined buffer, by wrapping MPI_Gather + print kernel
template <typename Tfloat>
inline void mpi_print(const int N, const int M, gpubuf<Tfloat>& buf, int num_ranks, int rank)
{
    const gpubuf<Tfloat> combined_buf = mpi_gather_buf<Tfloat>(buf, num_ranks, rank);
    print2d<Tfloat><<<1, 1>>>(N, M, combined_buf.data());
}

// Combine buffers and copy to host side
template <typename Tfloat>
inline void assemble_mpi_bufs_to_host(gpubuf<Tfloat>& buf, int num_ranks, int rank)
{
}

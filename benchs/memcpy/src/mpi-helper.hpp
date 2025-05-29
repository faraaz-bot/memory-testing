#include <benchmark/benchmark.h>
#include <iomanip>
#include <mpi.h>
#include <optional>
#include <sstream>

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

// Reports an MPI error, assuming MPI_ERRORS_RETURN is set
// Will not abort or do anything else, other than logging
inline void MPI_CHECK(int ret_val, int rank)
{
    if(ret_val != MPI_SUCCESS)
    {
        char errmsg[MPI_MAX_ERROR_STRING];
        int  errlen = -1;
        MPI_Error_string(ret_val, errmsg, &errlen);
        std::cout << "rank " << rank << " error: " << errmsg << std::endl;
    }
}

template <typename Tfloat>
std::vector<Tfloat> to_hostbuf(gpubuf<Tfloat>& buf)
{
    std::vector<Tfloat> res(buf.size());
    HIP_CHECK(
        hipMemcpy(res.data(), buf.data(), sizeof(Tfloat) * buf.size(), hipMemcpyDeviceToHost));
    return res;
}

template <typename Tfloat>
std::string buf_to_string(gpubuf<Tfloat>& gpubuf)
{
    std::vector<Tfloat> buf = to_hostbuf(gpubuf);
    const size_t        N   = buf.size();
    std::stringstream   ss;
    ss << "[";
    for(auto i = 0; i < N; i++)
    {
        ss << std::setw(6) << buf[i];
        if(i != N)
            ss << ", ";
    }
    ss << "]";
    return ss.str();
}

template <typename Tfloat>
std::string buf_to_string2d(int N, int M, gpubuf<Tfloat>& gpubuf)
{
    std::vector<Tfloat> buf = to_hostbuf(gpubuf);
    std::stringstream   ss;
    ss << "[";
    for(auto i = 0; i < N; i++)
    {
        ss << "[";
        for(auto j = 0; j < M; j++)
        {
            ss << buf[i * N + j];
            if(j != M)
                ss << std::setw(6) << ", ";
        }
        if(i != N)
            ss << ", ";
        ss << "]";
    }
}

// Print out all individual GPU bufs, in rank-order
template <typename Tfloat>
void mpi_print_bufs(int num_ranks, int rank, gpubuf<Tfloat>& buf)
{
}

// 2d format variant
template <typename Tfloat>
void mpi_print_bufs2d(int num_ranks, int rank, gpubuf<Tfloat>& buf)
{
}

// Gather bufs, to only rank 0
// Note: recv_buf is only significant on root (rank 0), so pass nullptr on other ranks
template <typename Tfloat>
std::optional<gpubuf<Tfloat>> mpi_gather_buf(int num_ranks, int rank, gpubuf<Tfloat>& buf)
{
    // Note this N is actually (ctx.N * ctx.N) / num_ranks, not same as ctx.N
    size_t       N        = buf.size();
    MPI_Datatype mpi_type = get_mpi_type(sizeof(Tfloat));
    if(rank == 0)
    {
        gpubuf<Tfloat> recv_buf = gpubuf<Tfloat>(N * num_ranks);
        MPI_Gather(buf.data(), N, mpi_type, recv_buf.data(), N, mpi_type, 0, MPI_COMM_WORLD);
        return recv_buf;
    }
    MPI_Gather(buf.data(), N, mpi_type, nullptr, N, mpi_type, 0, MPI_COMM_WORLD);
    return std::nullopt;
}

// Print out combined buffer, by wrapping MPI_Gather + print kernel
template <typename Tfloat>
void mpi_assemble_print(const int N, const int M, int num_ranks, int rank, gpubuf<Tfloat>& buf)
{
    if(rank == 0)
    {
        auto result = mpi_gather_buf<Tfloat>(num_ranks, rank, buf);
        if(!result.has_value())
            throw std::runtime_error("Rank 0 was unable to gather buf for print!");
        const gpubuf<Tfloat> combined_buf = result.value();
        print2d<Tfloat><<<1, 1>>>(N, M, combined_buf.data());
    }
    else
        (void)mpi_gather_buf<Tfloat>(num_ranks, rank, buf);
}

// Combine buffers and copy to host side to hostbuf_result
// Note: Currently if memcpy fails we can deadlock here.
template <typename Tfloat>
void assemble_mpi_bufs_to_host(int num_ranks, int rank, gpubuf<Tfloat>& buf, Tfloat* hostbuf_result)
{
    if(rank == 0)
    {
        auto result = mpi_gather_buf<Tfloat>(num_ranks, rank, buf);
        if(!result.has_value())
            throw std::runtime_error("Rank 0 was unable to gather buf for print!");
        gpubuf<Tfloat> combined_buf = result.value();
        HIP_CHECK(hipMemcpy(hostbuf_result,
                            combined_buf.data(),
                            combined_buf.size() * sizeof(Tfloat),
                            hipMemcpyDeviceToHost));
    }
    (void)mpi_gather_buf<Tfloat>(num_ranks, rank, buf);
}

// Clear data in out buffer to zero
template <typename Tfloat>
void mpi_buf_reset(const int rank, gpubuf<Tfloat>& gpubuf_output, std::vector<Tfloat>& host_output)
{
    HIP_CHECK(hipMemset(gpubuf_output.data(), 0, sizeof(Tfloat) * gpubuf_output.size()));
    if(rank == 0)
        std::fill(host_output.begin(), host_output.end(), 0);
}

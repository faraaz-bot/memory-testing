#include "helper.hpp"
#include "mpi-helper.hpp"
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

// (3) MPI Implementation
// Block transpose
#ifdef MPI_ENABLED
#include <mpi.h>
template <typename Tfloat>
float mpi_copy(const benchmark_context& ctx, gpubuf<Tfloat>& in_buf, gpubuf<Tfloat>& out_buf)
{
    int mpi_rank = 0;
    int mp_size  = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mp_size);
    MPI_Datatype mtype = get_mpi_type(sizeof(Tfloat));
    std::cout << "mp_size = " << mp_size << std::endl;
    const int sub_block_size = ctx.N / mp_size;

    MPI_Datatype strided_type; // Represent data for a sub_block
    MPI_Type_vector(sub_block_size, sub_block_size, ctx.N - sub_block_size, mtype, &strided_type);

    GPUTimer timer;
    timer.tick();

    // Each rank walks along each sub_block it has and exchanges it with another (or itself for diagonal)
    for(auto i = 0; i < mp_size; i++)
    {
        if(mpi_rank == i)
        {
            // Locally copy to out_bufs, for same GPU
            // Not using HIP_CHECK to avoid exiting on this rank to avoid deadlocks
            const size_t pitch_bytes           = ctx.N * sizeof(Tfloat);
            const size_t bytes_to_copy_per_row = sub_block_size * sizeof(Tfloat);
            hipError_t   err                   = hipMemcpy2D(out_buf.data(),
                                         pitch_bytes,
                                         in_buf.data(),
                                         pitch_bytes,
                                         bytes_to_copy_per_row,
                                         sub_block_size,
                                         hipMemcpyDeviceToDevice);
            if(err != hipSuccess)
                std::cout << "HIP error during hipMemcpy2D: " << hipGetErrorString(err)
                          << " at line" << __LINE__ << ", " << __FILE__ << ".\n";
        }
        else
        {
            // Exchange between two different ranks/GPU devices
            Tfloat*    recvbuf = out_buf.data() + (mpi_rank * sub_block_size);
            MPI_Status status;
            MPI_Sendrecv(in_buf.data(),
                         sub_block_size,
                         strided_type,
                         i,
                         0,
                         recvbuf,
                         sub_block_size,
                         strided_type,
                         i,
                         0,
                         MPI_COMM_WORLD,
                         &status);
        }
    }

    timer.tock();

    return timer.elapsed();
}

// Mirror of local_transpose kernel using GPU-aware MPI instead
template <typename Tfloat>
float mpi_local_transpose(const benchmark_context& ctx,
                          gpubuf<Tfloat>&          in_buf,
                          gpubuf<Tfloat>&          out_buf)
{
    float elapsed = 0.f;
    int   mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Datatype mtype = get_mpi_type(sizeof(Tfloat));
}
#endif

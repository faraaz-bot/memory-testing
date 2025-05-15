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
float mpi_copy(const benchmark_context& ctx,
               std::vector<Tfloat*>&    in_bufs,
               std::vector<Tfloat*>&    out_bufs)
{
    const int sub_block_length = ctx.N / ctx.ngpus;
    const int sub_block_size   = sub_block_length * sub_block_length;
    int       mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Datatype mtype = get_mpi_type(sizeof(Tfloat));

    Tfloat*      sendbuf = in_bufs[mpi_rank]; // Rank = GPU
    MPI_Datatype strided_type; // Represent data for a sub_block
    MPI_Type_vector(sub_block_size, sub_block_size, ctx.N - sub_block_size, mtype, &strided_type);

    GPUTimer timer;
    timer.tick();

    // Each rank walks along each sub_block it has and exchanges it with another (or itself for diagonal)
    for(auto i = 0; i < ctx.ngpus; i++)
    {
        if(mpi_rank == i)
        {
            // Locally copy to out_bufs, for same GPU
            // Not using HIP_CHECK to avoid exiting on this rank to avoid deadlocks
            const size_t pitch_bytes           = ctx.N * sizeof(Tfloat);
            const size_t bytes_to_copy_per_row = sub_block_length * sizeof(Tfloat);
            hipError_t   err                   = hipMemcpy2D(out_bufs[i] + (i * sub_block_size),
                                         pitch_bytes,
                                         sendbuf + (i * sub_block_size),
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
            Tfloat*    recvbuf = out_bufs[i] + (mpi_rank * sub_block_length);
            MPI_Status status;
            MPI_Sendrecv(sendbuf,
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
                          std::vector<Tfloat*>&    in_bufs,
                          std::vector<Tfloat*>&    out_bufs)
{
    float elapsed = 0.f;
    int   mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Datatype mtype = get_mpi_type(sizeof(Tfloat));
}
#endif

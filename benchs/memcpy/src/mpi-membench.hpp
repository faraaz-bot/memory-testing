#include "helper.hpp"
#include "mpi-helper.hpp"
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

// (3) MPI Implementation
// Block transpose
template <typename Tfloat>
float mpi_copy(const benchmark_context& ctx, gpubuf<Tfloat>& in_buf, gpubuf<Tfloat>& out_buf)
{
    int rank      = 0;
    int num_ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Datatype mtype            = get_mpi_type(sizeof(Tfloat));
    const int    sub_block_length = ctx.N / num_ranks;
    const int    elems_per_row    = ctx.N;
    const int    elems_per_col    = sub_block_length;
    const int    stride           = ctx.N - (sub_block_length);

    // Represent strided data with custom type
    MPI_Datatype strided_type;
    MPI_Type_vector(sub_block_length, sub_block_length, stride, mtype, &strided_type);
    // MPI_Type_vector(2, 2, 6, mtype, &strided_type);
    MPI_Type_commit(&strided_type);

    GPUTimer timer;
    timer.tick();

    // Each rank transfers a full "sub_block" or brick of data
    // to perform a block transpose
    const int alltoall_count = sub_block_length * sub_block_length;
    if(rank == 0)
        printf("ctx.N = %d\nnum_ranks = %d\nsub_block_length = %d\nalltoall_count = %d\n",
               (int)ctx.N,
               num_ranks,
               sub_block_length,
               alltoall_count);
    MPI_Alltoall(in_buf.data(),
                 alltoall_count,
                 strided_type,
                 out_buf.data(),
                 alltoall_count,
                 strided_type,
                 MPI_COMM_WORLD);
    // MPI_Alltoall(in_buf.data(),
    //              alltoall_count, 4
    //              strided_type,
    //              out_buf.data(),
    //              alltoall_count, 4
    //              strided_type,
    //              MPI_COMM_WORLD);

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
    int   rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype mtype = get_mpi_type(sizeof(Tfloat));
}

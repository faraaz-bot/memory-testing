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
    MPI_Datatype mtype = get_mpi_type(sizeof(Tfloat));

    const int N                = static_cast<int>(ctx.N);
    const int sub_block_length = ctx.N / num_ranks;
    const int buf_size         = static_cast<int>(in_buf.size()); // size of buf per device/rank
    const int elems_per_row    = ctx.N;
    const int elems_per_col    = sub_block_length;

    // Create custom strided 2d type to specify non-contiguous data to move in
    // MPI_Alltoall call later.
    MPI_Datatype strided_type;
    // MPI_Type_vector(sub_block_length, sub_block_length, ctx.N, mtype, &strided_type);
    const int sizes[2]     = {N, N}; // global array lengths
    const int sub_sizes[2] = {sub_block_length, sub_block_length}; // subarray sizes
    const int start_offsets[2]
        = {rank * sub_block_length, 0}; // offset of each rank within global N x N array

    MPI_Type_create_subarray(2, // # of dims
                             sizes,
                             sub_sizes,
                             start_offsets,
                             MPI_ORDER_C,
                             mtype,
                             &strided_type);
    MPI_Type_commit(&strided_type);

    GPUTimer timer;
    timer.tick();

    // Each rank transfers a full "sub_block" or brick of data
    // to perform a block transpose
    const int alltoall_count = sub_block_length * sub_block_length;
    int       ret            = MPI_Alltoall(in_buf.data(),
                           alltoall_count,
                           strided_type,
                           out_buf.data(),
                           alltoall_count,
                           strided_type,
                           MPI_COMM_WORLD);
    MPI_CHECK(ret, rank);

    timer.tock();

    MPI_Type_free(&strided_type);

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

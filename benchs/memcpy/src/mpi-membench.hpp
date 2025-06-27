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

    // Setup MPI subarray for each block to be transferred
    const int                 sizes[2]     = {N, N / num_ranks}; // gpubuf dims
    const int                 sub_sizes[2] = {sub_block_length, sub_block_length}; // subarray sizes
    std::vector<MPI_Datatype> subarrays(num_ranks);
    for(auto i = 0; i < num_ranks; i++)
    {
        const int start_offsets[2]
            = {i * sub_block_length, 0}; // offset to start of each block within global N x N array
        MPI_Datatype subarray_type;
        MPI_Type_create_subarray(2, // # of dims
                                 sizes,
                                 sub_sizes,
                                 start_offsets,
                                 MPI_ORDER_FORTRAN, // For HIP programming
                                 mtype,
                                 &subarray_type);
        MPI_Type_commit(&subarray_type);
        subarrays[i] = subarray_type;
    }

    // Each rank sends/recvs from each rank
    std::vector<int> counts(num_ranks);
    std::fill(counts.begin(), counts.end(), 1);
    // Let subarray handle displacements, so set to 0
    std::vector<int> displs(num_ranks);
    std::fill(displs.begin(), displs.end(), 0);

    GPUTimer timer;
    timer.tick();

    // Each rank transfers a "sub_block" or brick of data
    // to each rank to perform a block transpose
    int ret = MPI_Alltoallw(in_buf.data(),
                            counts.data(), // 0,0,0,...,0
                            displs.data(), // 0,0,0,...,0
                            subarrays.data(), // types for block 1, block 2,..., block {num_rank}.
                            out_buf.data(),
                            counts.data(),
                            displs.data(),
                            subarrays.data(),
                            MPI_COMM_WORLD);
    MPI_CHECK(ret, rank);

    timer.tock();

    for(auto& type : subarrays)
        MPI_Type_free(&type);

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

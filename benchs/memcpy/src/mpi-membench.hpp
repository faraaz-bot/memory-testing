#include "helper.hpp"
#include "mpi-helper.hpp"
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

// Constants for transpose tiling
constexpr int MAX_TILE_SIZE    = 32;
constexpr int ITEMS_PER_THREAD = 4;

// (3) MPI Implementation
// (3.1) Block transpose
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

    // Each rank sends/recvs a block from each rank
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

// Adapted version of local transpose kernel from membench.hpp.
// Adjusts arguments such that each rank can pass in it's own local in/out buffer, rather than
// having device 0 initiating all buffers passed in.
// NOTE: Assumes power of two for num_ranks & N and (N/num_ranks) >= ITEMS_PER_THREAD, does not work for general params
template <typename Tfloat>
__global__ __launch_bounds__(1024) void rank_local_transpose(
    const size_t N, const size_t num_ranks, const size_t tile_size, Tfloat* in_buf, Tfloat* out_buf)
{
    __shared__ Tfloat lds[MAX_TILE_SIZE][MAX_TILE_SIZE + 1]; // Offset to avoid bank conflicts

    // Determine which GPU buffers to use as input and output
    const size_t num_tiles_in_axis = std::sqrt(gridDim.z);

    // Offsets
    // Tile (x,y) when reading in, flip to (y,x) for writing
    const auto tile_x_offset  = blockIdx.z % num_tiles_in_axis;
    const auto tile_y_offset  = blockIdx.z / num_tiles_in_axis;
    const auto sub_block_size = N / num_ranks;

    // Indices to use
    auto tile_x = tile_x_offset * tile_size;
    auto tile_y = tile_y_offset * tile_size;
    auto glb_x  = threadIdx.x + tile_x + blockIdx.x * sub_block_size;

    // Read in coalesced from global mem
#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        auto glb_y = tile_y + threadIdx.y * ITEMS_PER_THREAD + i;
        lds[threadIdx.y * ITEMS_PER_THREAD + i][threadIdx.x] = in_buf[glb_y * N + glb_x];
    }

    __syncthreads();

    tile_x = tile_y_offset * tile_size;
    tile_y = tile_x_offset * tile_size;

#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        glb_x                      = threadIdx.x + tile_x + blockIdx.x * sub_block_size;
        auto glb_y                 = tile_y + threadIdx.y * ITEMS_PER_THREAD + i;
        out_buf[glb_y * N + glb_x] = lds[threadIdx.x][threadIdx.y * ITEMS_PER_THREAD + i];
    }
}

// Mirror of local_transpose_launcher to launch local transpose kernel per rank
// Adjust args so that a (num_ranks x 1) strip of blocks is handled, rather than a grid of blocks
template <typename Tfloat>
float mpi_local_transpose_launcher(const benchmark_context& ctx,
                                   gpubuf<Tfloat>&          in_buf,
                                   gpubuf<Tfloat>&          out_buf)
{
    int       rank;
    int       num_ranks = 0;
    const int N         = static_cast<int>(ctx.N);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // local_transpose args - note: MAX_TILE_SIZE is defined in membench.hpp
    const uint32_t sub_block_size = N / num_ranks; // Length of block in each transfer
    const uint32_t actual_tile_size
        = min(MAX_TILE_SIZE, sub_block_size); // Clamp it for small sizes
    const uint32_t num_threads_x = actual_tile_size;
    const uint32_t num_threads_y = (actual_tile_size < ITEMS_PER_THREAD)
                                       ? actual_tile_size
                                       : actual_tile_size / ITEMS_PER_THREAD;
    const uint32_t num_tiles
        = ceildiv(sub_block_size * sub_block_size,
                  actual_tile_size * actual_tile_size); // How many total tiles needed per sub_block
    const dim3 grid_dim{(uint32_t)num_ranks, 1u, num_tiles};
    const dim3 block_dim{num_threads_x, num_threads_y};

    GPUTimer timer;
    timer.tick();

    rank_local_transpose<Tfloat>
        <<<grid_dim, block_dim>>>(N, num_ranks, actual_tile_size, in_buf.data(), out_buf.data());
    HIP_CHECK(hipDeviceSynchronize());
    timer.tock();

    return timer.elapsed();
}

// mpi_copy + local_transpose
template <typename Tfloat>
float mpi_copy_transpose(const benchmark_context& ctx,
                         gpubuf<Tfloat>&          in_buf,
                         gpubuf<Tfloat>&          out_buf)
{
    int num_ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    gpubuf<Tfloat> tmp(ctx.N * ctx.N / num_ranks);
    float          block_transpose_time = mpi_copy(ctx, in_buf, tmp);
    float          local_transpose_time = mpi_local_transpose_launcher(ctx, tmp, out_buf);
    return block_transpose_time + local_transpose_time;
}

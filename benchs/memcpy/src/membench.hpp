#include "helper.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

// Constants for transpose tiling
constexpr int MAX_TILE_SIZE    = 32;
constexpr int ITEMS_PER_THREAD = 4;

// Collection of all non-MPI benchmark implementations

/*
 * TODO list:
 * - Complex data
 * - Implement basic implementations for each method
 * - Optimize stuff after
 *     - Experiment with async, LDS optimizations, bank conflicts
 *     - Toggling SDMA
 *     - Pinned memory, HMM?
 * - Refactoring
 *     - main.cpp:
 *          - List verbosity details
 *          - Test/verify (-c) as a subcommand?
 *
 * - Further out tasks to consider
 *     - hipGraph vs stream (ngpus vs ngpus^2 # of streams) async comparison
 *     - Rectangular data (still out of place)
 *     - Arbitrary dim?
 *     - Optionally regenerate new inputs per trial?
*/

/* Implementations */
// These perform a block transpose between `ngpus` # of buffers, with each each block
// being a square of length (N / ngpus), where N is the length of the original
// square matrix, which is split across all GPU devices.
//
// Host function ("launcher" in case of kernel benchmark) is passed in to run_benchmark,
// with float return value being time in ms that was recorded for one execution

// (1.1) hipMemcpy2D between two devices
template <typename Tfloat>
float run_memcpy(const benchmark_context& ctx,
                 gpubuf_vec<Tfloat>&      in_bufs,
                 gpubuf_vec<Tfloat>&      out_bufs)
{
    const size_t N              = ctx.N;
    const size_t ngpus          = ctx.ngpus;
    const size_t sub_block_size = N / ngpus; // Length of block in each transfer
    const size_t bytes_to_copy_per_row
        = sub_block_size * sizeof(Tfloat); // Bytes per row in transfer
    const size_t pitch_bytes = N * sizeof(Tfloat); // Width of buf

    GPUTimer timer;
    timer.tick();
    for(auto i = 0; i < ngpus; i++) // src GPU
    {
        for(auto j = 0; j < ngpus; j++) // Offset within GPU, AKA dst GPU
        {
            HIP_CHECK(hipMemcpy2D(out_bufs[j] + (i * sub_block_size),
                                  pitch_bytes,
                                  in_bufs[i] + (j * sub_block_size),
                                  pitch_bytes,
                                  bytes_to_copy_per_row,
                                  sub_block_size,
                                  hipMemcpyDeviceToDevice));
        }
    }
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    timer.tock();
    return timer.elapsed();
}

// (1.2) hipMemcpy2D between two devices, using stream per each gpu-gpu interaction
template <typename Tfloat>
float run_memcpy_async(const benchmark_context& ctx,
                       gpubuf_vec<Tfloat>&      in_bufs,
                       gpubuf_vec<Tfloat>&      out_bufs)
{
    const size_t                    N       = ctx.N;
    const size_t                    ngpus   = ctx.ngpus;
    const std::vector<hipStream_t>& streams = ctx.streams;

    const size_t sub_block_size = N / ngpus; // Length of block in each transfer
    const size_t bytes_to_copy_per_row
        = sub_block_size * sizeof(Tfloat); // Bytes per row in transfer
    const size_t pitch_bytes = N * sizeof(Tfloat); // Width of buf

    GPUTimer timer;
    timer.tick();
    for(auto i = 0; i < ngpus; i++) // src GPU
    {
        for(auto j = 0; j < ngpus; j++) // Offset within GPU, AKA dst GPU
        {
            hipStream_t stream = streams[i * ngpus + j];
            HIP_CHECK(hipMemcpy2DAsync(out_bufs[j] + (i * sub_block_size),
                                       pitch_bytes,
                                       in_bufs[i] + (j * sub_block_size),
                                       pitch_bytes,
                                       bytes_to_copy_per_row,
                                       sub_block_size,
                                       hipMemcpyDeviceToDevice,
                                       stream));
        }
    }

    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    timer.tock();
    return timer.elapsed();
}

// TODO
// (1.3) hipMemcpy2d using hipGraph

// TODO Try adding streams / increase parallelism more?
// (2) Copy kernels that perform block transpose
template <typename Tfloat>
__global__ void naive_copy(const size_t N,
                           const size_t ngpus,
                           const size_t items_per_thread,
                           Tfloat**     in_bufs,
                           Tfloat**     out_bufs)
{
    const size_t sub_block_size = std::sqrt(items_per_thread);
    const size_t bIndex         = blockIdx.x;
    const size_t tIndex         = threadIdx.x;

    for(size_t x = 0; x < sub_block_size; x++)
    {
        for(size_t y = 0; y < sub_block_size; y++)
        {
            size_t oIndex = x * N + (tIndex * sub_block_size + y);
            size_t nIndex = x * N + (bIndex * sub_block_size + y);

            out_bufs[tIndex][nIndex] = in_bufs[bIndex][oIndex];
        }
    }
}

// FIXME: Seems to have some kind of synchronization issue, where part of data is missing...?
// Try to improve on naive with LDS usage (not expected to actually provide improvement)
// This variant will map blocks to gpus, and have each thread operates on one "sub_block"
template <typename Tfloat>
__global__ void lds_copy(const size_t N, const size_t ngpus, Tfloat** in_bufs, Tfloat** out_bufs)
{
    const size_t           bIndex = blockIdx.x;
    const size_t           tIndex = threadIdx.x;
    extern __shared__ char lds_char[]; // Should be buf_elems size, for curr GPU buf
    auto                   lds
        = reinterpret_cast<Tfloat*>(lds_char); // Workaround declaring extern lds for diff types

    const size_t items_per_thread = (N * N) / (ngpus * ngpus);
    const size_t sub_block_size   = std::sqrt(items_per_thread);

    for(size_t i = 0; i < items_per_thread; i++)
        lds[tIndex * items_per_thread + i] = in_bufs[bIndex][tIndex * items_per_thread + i];
    __syncthreads(); // Sync since we will be reading different values than what we just wrote to LDS

    for(size_t x = 0; x < sub_block_size; x++)
    {
        for(size_t y = 0; y < sub_block_size; y++)
        {

            size_t oIndex = x * N + (tIndex * sub_block_size + y);
            size_t nIndex = x * N + (bIndex * sub_block_size + y);

            out_bufs[tIndex][nIndex] = lds[oIndex];
        }
    }
}

// Handle setup of device ptr to all device bufs, launching of copy kernels, and timing them
template <typename Tfloat>
float naive_copy_launcher(const benchmark_context& ctx,
                          gpubuf_vec<Tfloat>&      in_bufs,
                          gpubuf_vec<Tfloat>&      out_bufs)
{
    const size_t ngpus = ctx.ngpus;

    // Copy over device ptrs stored in in_bufs/out_bufs into device side array
    size_t num_blocks  = ngpus;
    size_t num_threads = num_blocks;

    size_t items_per_thread = (ctx.N * ctx.N) / (num_blocks * num_threads);

    // Execute kernel and time it
    GPUTimer timer;
    timer.tick();
    naive_copy<Tfloat><<<num_blocks, num_threads>>>(
        ctx.N, ctx.ngpus, items_per_thread, in_bufs.data(), out_bufs.data());
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    timer.tock();
    return timer.elapsed();
}

// TODO: fix issue with missing data, incorrect timings (synchronization issue somewhere?)
template <typename Tfloat>
float lds_copy_launcher(const benchmark_context& ctx,
                        gpubuf_vec<Tfloat>&      in_bufs,
                        gpubuf_vec<Tfloat>&      out_bufs)
{
    const size_t ngpus           = ctx.ngpus;
    const auto   sub_block_bytes = sizeof(Tfloat) * ctx.N * ctx.N / ngpus;

    // Execute kernel and time it
    GPUTimer timer;
    timer.tick();
    lds_copy<Tfloat>
        <<<ngpus, ngpus, sub_block_bytes>>>(ctx.N, ctx.ngpus, in_bufs.data(), out_bufs.data());
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    timer.tock();

    return timer.elapsed();
}

// Performs local transposes inside of blocks. Expected to be run after
// one of the implementations performing block-wise transpose
// e.g. naiveCopy, hipMemcpy2D, hipMemcpy2DAsync
// NOTE: Assumes power of two for ngpus & N and (N/ngpus) >= ITEMS_PER_THREAD, does not work for general params
template <typename Tfloat>
__global__ __launch_bounds__(1024) void local_transpose(
    const size_t N, const size_t ngpus, const size_t tile_size, Tfloat** in_bufs, Tfloat** out_bufs)
{
    __shared__ Tfloat lds[MAX_TILE_SIZE][MAX_TILE_SIZE + 1]; // Offset to avoid bank conflicts

    // Determine which GPU buffers to use as input and output
    const size_t num_tiles_in_axis = std::sqrt(gridDim.z);
    const size_t src               = (blockIdx.y * ngpus + blockIdx.x) / ngpus;
    Tfloat*      idata             = in_bufs[src];
    Tfloat*      odata             = out_bufs[src];

    // Offsets
    // Tile (x,y) when reading in, flip to (y,x) for writing
    const auto tile_x_offset  = blockIdx.z % num_tiles_in_axis;
    const auto tile_y_offset  = blockIdx.z / num_tiles_in_axis;
    const auto sub_block_size = N / ngpus;

    // Indices to use
    auto tile_x = tile_x_offset * tile_size;
    auto tile_y = tile_y_offset * tile_size;
    auto glb_x  = threadIdx.x + tile_x + blockIdx.x * sub_block_size;

    // Read in coalesced from global mem
#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        auto glb_y = tile_y + threadIdx.y * ITEMS_PER_THREAD + i;
        lds[threadIdx.y * ITEMS_PER_THREAD + i][threadIdx.x] = idata[glb_y * N + glb_x];
    }

    __syncthreads();

    tile_x = tile_y_offset * tile_size;
    tile_y = tile_x_offset * tile_size;

#pragma unroll
    for(int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        glb_x                    = threadIdx.x + tile_x + blockIdx.x * sub_block_size;
        auto glb_y               = tile_y + threadIdx.y * ITEMS_PER_THREAD + i;
        odata[glb_y * N + glb_x] = lds[threadIdx.x][threadIdx.y * ITEMS_PER_THREAD + i];
    }
}

// Launch local_transpose kernel with standard args
template <typename Tfloat>
float local_transpose_launcher(const benchmark_context& ctx,
                               gpubuf_vec<Tfloat>&      in_bufs,
                               gpubuf_vec<Tfloat>&      out_bufs)
{
    const size_t N     = ctx.N;
    const size_t ngpus = ctx.ngpus;

    // local_transpose args
    const uint32_t sub_block_size = N / ngpus; // Length of block in each transfer
    const uint32_t actual_tile_size
        = min(MAX_TILE_SIZE, sub_block_size); // Clamp it for small sizes
    const uint32_t num_threads_x = actual_tile_size;
    const uint32_t num_threads_y = (actual_tile_size < ITEMS_PER_THREAD)
                                       ? actual_tile_size
                                       : actual_tile_size / ITEMS_PER_THREAD;
    const uint32_t num_tiles
        = ceildiv(sub_block_size * sub_block_size,
                  actual_tile_size * actual_tile_size); // How many total tiles needed per sub_block
    const dim3 grid_dim{(uint32_t)ngpus, (uint32_t)ngpus, num_tiles};
    const dim3 block_dim{num_threads_x, num_threads_y};

    // execute kernels and time them
    GPUTimer timer;
    timer.tick();

    local_transpose<Tfloat>
        <<<grid_dim, block_dim>>>(N, ngpus, actual_tile_size, in_bufs.data(), out_bufs.data());
    timer.sync_all(ngpus); // ensure all gpus have finished their work

    timer.tock();

    return timer.elapsed();
}

/* Block-wide transpose + local transpose implementations */
// Time naive_copy() + local_transpose()
template <typename Tfloat>
float naive_copy_transpose(const benchmark_context& ctx,
                           gpubuf_vec<Tfloat>&      in_bufs,
                           gpubuf_vec<Tfloat>&      out_bufs)
{
    // Calculate number of blocks/threads to launch with
    const size_t   N        = ctx.N;
    const size_t   ngpus    = ctx.ngpus;
    const uint32_t copy_ipt = (N * N) / (ngpus * ngpus);

    // Create intermediate tmp buffer between block transpose and local transpose
    gpubuf_vec<Tfloat> tmp(N, ngpus);

    // Execute kernels and time them
    GPUTimer timer;
    timer.tick();

    naive_copy<Tfloat><<<ngpus, ngpus>>>(N, ngpus, copy_ipt, in_bufs.data(), tmp.data());
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    timer.tock();
    float transpose_time = local_transpose_launcher(ctx, tmp, out_bufs);
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work

    return timer.elapsed() + transpose_time;
}

// Time run_memcpy() + local_transpose()
template <typename Tfloat>
float run_memcpy_transpose(const benchmark_context& ctx,
                           gpubuf_vec<Tfloat>&      in_bufs,
                           gpubuf_vec<Tfloat>&      out_bufs)
{
    const size_t N     = ctx.N;
    const size_t ngpus = ctx.ngpus;

    // Create intermediate tmp buffer between block transpose and local transpose
    gpubuf_vec<Tfloat> tmp(N, ngpus);

    // Execute block transpose via hipMemcpy2D + local transpose kernel and time them
    float memcpy_time    = run_memcpy(ctx, in_bufs, tmp);
    float transpose_time = local_transpose_launcher(ctx, tmp, out_bufs);

    return memcpy_time + transpose_time;
}

// Time run_memcpy_async() + local_transpose()
template <typename Tfloat>
float run_memcpy_async_transpose(const benchmark_context& ctx,
                                 gpubuf_vec<Tfloat>&      in_bufs,
                                 gpubuf_vec<Tfloat>&      out_bufs)
{
    const size_t N     = ctx.N;
    const size_t ngpus = ctx.ngpus;

    // Create intermediate tmp buffer between block transpose and local transpose
    gpubuf_vec<Tfloat> tmp(N, ngpus);

    // Execute block transpose via hipMemcpy2D + local transpose kernel and time them
    float memcpy_time    = run_memcpy(ctx, in_bufs, tmp);
    float transpose_time = local_transpose_launcher(ctx, tmp, out_bufs);

    return memcpy_time + transpose_time;
}

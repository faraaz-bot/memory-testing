#include "helper.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdio.h>
#include <vector>

const int TILE_SIZE = 64;

/**
 * Benchmarking tool for comparing speed of various memory copy methods
 * between multiple gpus. Currently will do out-of-place operations on 
 * square matrices only.
 *
 * TODO list:
 * - Implement basic implementations for each method
 *     - Adjust timing to exclude hipMemcpy for input pointers for kernels
 * - Optimize stuff after
 *     - Experiment with async, LDS optimizations, bank conflicts
 *     - Toggling SDMA
 *     - Pinned memory, HMM?
 * - Perform local transpose on data as well
 *
 * - Display/write output timings/other metrics, allow ntrials
 *     - Add Google Benchmark
 *
 * - Further out tasks to consider
 *     - hipGraph vs stream (ngpus vs ngpus^2 # of streams) async comparison
 *     - Rectangular data (still out of place)
 *     - Arbitrary dim?
 *     - Optionally regenerate new inputs per trial?
*/

/* Helpers for verifying correctness */

// PRNG for input generation
#define xorwow_next(states, max, min, val) \
    uint32_t t = states[4];                \
    uint32_t s = states[0];                \
    states[4]  = states[3];                \
    states[3]  = states[2];                \
    states[2]  = states[1];                \
    states[1]  = s;                        \
    t ^= t >> 2;                           \
    t ^= t << 1;                           \
    t ^= s ^ (s << 4);                     \
    states[0] = t;                         \
    states[5] += 362437;                   \
    uint32_t temp = t + states[5];         \
    val = min + (static_cast<Tfloat>(temp) * (max - min)) / static_cast<Tfloat>(4294967295);

template <typename Tfloat>
__global__ void populate_array(const size_t N,
                               Tfloat*      out,
                               const Tfloat min,
                               const Tfloat max,
                               const bool   isRandom,
                               size_t       seed)
{
    const size_t bIndex         = blockIdx.x;
    const size_t tIndex         = threadIdx.x;
    const size_t itemsPerThread = N;
    const size_t blockSize      = itemsPerThread * blockDim.x;
    const size_t start          = (tIndex * itemsPerThread) + (bIndex * blockSize);

    if(isRandom)
    {
        uint32_t states[6];
        states[0] = seed ^ tIndex + bIndex;
        states[1] = seed >> 1 ^ (tIndex + bIndex * 2);
        states[2] = seed >> 2 ^ (tIndex + bIndex * 3);
        states[3] = seed >> 3 ^ (tIndex + bIndex * 4);
        states[4] = seed >> 4 ^ (tIndex + bIndex * 5);
        states[5] = seed + tIndex + bIndex;

        Tfloat temp;
        for(size_t i = 0; i < 5; i++)
        {
            xorwow_next(states, max, min, temp);
        }

        for(size_t i = 0; i < itemsPerThread; i++)
        {
            if(start + i >= N * N)
                continue;
            xorwow_next(states, max, min, out[start + i]);
        }
    }
    else
    {
        for(size_t i = 0; i < itemsPerThread; i++)
        {
            if(start + i >= N * N)
                continue;
            out[start + i] = start + i;
        }
    }
}

// Helper kernel just to print N consecutive values in gpubuf
template <typename Tfloat>
__global__ void print(const int N, const Tfloat* input)
{
    printf("[ ");
    for(int i = 0; i < N; i++)
        printf("%.6f ", input[i]);
    printf("]\n");
}

// Helper kernel just to print NxM consecutive values in gpubuf, with 2d formatting
template <typename Tfloat>
__global__ void print2d(const int N, const int M, const Tfloat* input)
{
    printf("[\n");
    for(int i = 0; i < N; i++)
    {
        printf("\t[ ");
        for(int j = 0; j < M; j++)
        {
            auto idx = j + M * i;
            printf("%.6f ", input[idx]);
        }
        printf("]\n");
    }
    printf("]\n");
}

// Helper to print initial host matrix and transposed matrix
template <typename Tfloat>
void print_host_2d(const int N, const int M, const std::vector<Tfloat>& input)
{
    std::cout << "[\n";
    for(int i = 0; i < N; i++)
    {
        std::cout << "  [ ";
        for(int j = 0; j < M; j++)
        {
            auto idx = i * N + j;
            std::cout << std::setw(6) << input[idx] << " ";
        }
        std::cout << " ]\n";
    }
    std::cout << "]" << std::endl;
}

// Combine ngpu # of gpubuf partitions back in an N x N matrix on the host
// * Assumes hostbuf_result has enough memory allocated for it
template <typename Tfloat>
void assemble_output_to_host(const int                   N,
                             const std::vector<Tfloat*>& gpubufs,
                             Tfloat*                     hostbuf_result)
{
    const size_t ngpus    = gpubufs.size();
    const size_t buf_size = N * N / ngpus;
    for(auto i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipMemcpy(hostbuf_result + i * buf_size,
                            gpubufs[i],
                            buf_size * sizeof(Tfloat),
                            hipMemcpyDeviceToHost));
    }
}

// TODO May need to update this if adding complex data
// Check equality of matrices, should be exactly the same since data is copied
template <typename Tfloat>
bool is_same_matrix(const int                  N,
                    const std::vector<Tfloat>& input1,
                    const std::vector<Tfloat>& input2)
{
    for(auto i = 0; i < N * N; i++)
        if(input1[i] != input2[i])
            return false;
    return true;
}

// Reference impl on CPU that does not perform local transpose (out-of-place)
template <typename Tfloat>
void host_copy(const size_t N, const size_t ngpus, const Tfloat* input, Tfloat* output)
{
    const size_t buf_elems = N * N / ngpus;
    // Where each device's partitions would start
    std::vector<size_t> offsets(ngpus);
    for(auto i = 0; i < ngpus; i++)
        offsets[i] = i * buf_elems;

    const size_t sub_block_size  = N / ngpus; // Length of block in each transfer
    const size_t sub_block_bytes = sizeof(Tfloat) * sub_block_size;
    const size_t elems_per_row   = sub_block_size * ngpus; // Elems per row in transfer

    // Simulating GPU to GPU data layout & copy
#pragma omp parallel for
    for(auto src = 0; src < ngpus; src++)
    {
        for(auto dst = 0; dst < ngpus; dst++)
        {
            // Copy sub_block to output buf across diagonal
            for(auto row = 0; row < sub_block_size; row++)
            {
                size_t src_offset = offsets[src] + (dst * sub_block_size) + (elems_per_row * row);
                size_t dst_offset = offsets[dst] + (src * sub_block_size) + (elems_per_row * row);
                std::memcpy(output + dst_offset, input + src_offset, sub_block_bytes);
            }
        }
    }
}

// Reference impl on CPU (out-of-place)
template <typename Tfloat>
void host_transpose(const int N, const Tfloat* input, Tfloat* output)
{
#pragma omp parallel for
    for(size_t i = 0; i < N; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            // Get curr index and send data to opposing location across diagonal
            auto idx1    = i * N + j;
            auto idx2    = j * N + i;
            output[idx2] = input[idx1];
        }
    }
}

/* Setup and Teardown helpers */
// Generates data of specified type (by gen) on device and transfers to host
template <typename Tfloat>
std::vector<Tfloat> generate(size_t N, size_t M, generator gen, Tfloat min, Tfloat max)
{
    // TODO add complex data support
    // bool is_complex = (gen == p_complex_single || gen == p_complex_double);
    std::vector<Tfloat> input(N * M);

    bool isRandom = gen == gen_random;

    Tfloat* dArr;
    HIP_CHECK(hipMalloc(&dArr, sizeof(Tfloat) * N * M));

    size_t threads = N <= 1024 ? N : 1024;
    // size_t itemsPerThread = N <= 1024 ? N : 1024;
    size_t blocks = std::ceil(static_cast<double>((N * M)) / static_cast<double>((threads * N)));

    auto now    = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);

    auto   value    = now_ms.time_since_epoch();
    size_t duration = value.count();
    populate_array<<<blocks, threads>>>(N, dArr, min, max, isRandom, duration);

    HIP_CHECK(hipMemcpy(input.data(), dArr, sizeof(Tfloat) * N * M, hipMemcpyDeviceToHost));

    return input;
}

// Allocate and initialize gpu buffers, streams + distribute host input to gpu buffers
template <typename Tfloat>
void setup(size_t                     N,
           size_t                     ngpus,
           std::vector<Tfloat*>&      gpubufs_input,
           std::vector<Tfloat*>&      gpubufs_output,
           const std::vector<Tfloat>& host_input,
           std::vector<hipStream_t>&  streams)
{
    const size_t buf_elems = N * N / ngpus;

    for(auto i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(&gpubufs_input[i], sizeof(Tfloat) * buf_elems));
        HIP_CHECK(hipMemcpy(gpubufs_input[i],
                            host_input.data() + i * buf_elems,
                            buf_elems * sizeof(Tfloat),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&gpubufs_output[i], sizeof(Tfloat) * buf_elems));
        HIP_CHECK(hipMemset(gpubufs_output[i], 0, sizeof(Tfloat) * buf_elems));

        // Assign streams to current gpus, for each other gpu (including self)
        for(auto j = 0; j < ngpus; j++)
            HIP_CHECK(hipStreamCreate(&streams[i * ngpus + j]));
    }
}

// Clear data in out buffers to zero
template <typename Tfloat>
void reset(const int             N,
           const int             ngpus,
           std::vector<Tfloat*>& gpubufs_output,
           std::vector<Tfloat>&  host_assembled_buf)
{
    const size_t buf_elems = N * N / ngpus; // Number of elements in buf
    for(auto i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipMemset(gpubufs_output[i], 0, sizeof(Tfloat) * buf_elems));
    }
    std::fill(host_assembled_buf.begin(), host_assembled_buf.end(), 0);
}

// Free allocated memory and streams
template <typename Tfloat>
void teardown(const int                 ngpus,
              std::vector<Tfloat*>&     gpubufs_input,
              std::vector<Tfloat*>&     gpubufs_output,
              std::vector<hipStream_t>& streams)
{
    for(auto i = 0; i < ngpus; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(gpubufs_input[i]));
        HIP_CHECK(hipFree(gpubufs_output[i]));
        for(auto j = 0; j < ngpus; j++)
            HIP_CHECK(hipStreamDestroy(streams[i * ngpus + j]));
    }
}

// RAII struct for temporary buffers for intermediate results
// Used when performing multiple out-of-place operations
template <typename Tfloat>
struct gpubuf
{
    size_t   N;
    size_t   ngpus;
    Tfloat** bufs;

    gpubuf(size_t N, size_t ngpus)
        : N(N)
        , ngpus(ngpus)
    {
        HIP_CHECK(hipMalloc(&bufs, sizeof(Tfloat*) * ngpus));
        const size_t buf_elems = N * N / ngpus;
        for(auto i = 0; i < ngpus; i++)
        {
            HIP_CHECK(hipMalloc(&bufs[i], sizeof(Tfloat) * buf_elems));
            HIP_CHECK(hipMemset(bufs[i], 0, sizeof(Tfloat) * buf_elems));
        }
    }

    ~gpubuf()
    {
        for(auto i = 0; i < ngpus; i++)
            HIP_CHECK(hipFree(bufs[i]));
        HIP_CHECK(hipFree(bufs));
    }
};

/* Implementations */
// Host function ("launcher" in case of kernel benchmark) is passed in to benchmark
// float return value is the time in ms that was recorded for one execution

// (1.1) hipMemcpy2D between two devices
template <typename Tfloat>
float run_memcpy(const benchmark_context& ctx,
                 std::vector<Tfloat*>&    in_bufs,
                 std::vector<Tfloat*>&    out_bufs)
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
    timer.tock();
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    return timer.elapsed();
}

// (1.2) hipMemcpy2D between two devices, using stream per each gpu-gpu interaction
template <typename Tfloat>
float run_memcpy_async(const benchmark_context& ctx,
                       std::vector<Tfloat*>&    in_bufs,
                       std::vector<Tfloat*>&    out_bufs)
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

// (2) Copy kernels
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

// Try to improve on naive with LDS usage
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
                          std::vector<Tfloat*>&    in_bufs,
                          std::vector<Tfloat*>&    out_bufs)
{
    const size_t ngpus = ctx.ngpus;

    // Copy over device ptrs stored in in_bufs/out_bufs into device side array
    Tfloat** d_in_bufs;
    Tfloat** d_out_bufs;
    HIP_CHECK(hipMalloc(&d_in_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMalloc(&d_out_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMemcpy(d_in_bufs, in_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_out_bufs, out_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));

    size_t num_blocks  = ngpus;
    size_t num_threads = num_blocks;

    size_t items_per_thread = (ctx.N * ctx.N) / (num_blocks * num_threads);

    // Execute kernel and time it
    GPUTimer timer;
    timer.tick();
    naive_copy<Tfloat>
        <<<num_blocks, num_threads>>>(ctx.N, ctx.ngpus, items_per_thread, d_in_bufs, d_out_bufs);
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    timer.tock();

    HIP_CHECK(hipFree(d_in_bufs));
    HIP_CHECK(hipFree(d_out_bufs));
    return timer.elapsed();
}

// TODO: fix issue with missing data, incorrect timings (synchronization issue somewhere?)
template <typename Tfloat>
float lds_copy_launcher(const benchmark_context& ctx,
                        std::vector<Tfloat*>&    in_bufs,
                        std::vector<Tfloat*>&    out_bufs)
{
    const size_t ngpus           = ctx.ngpus;
    const auto   sub_block_bytes = sizeof(Tfloat) * ctx.N * ctx.N / ngpus;

    // Copy over device ptrs stored in in_bufs/out_bufs into device side array
    Tfloat** d_in_bufs;
    Tfloat** d_out_bufs;
    HIP_CHECK(hipMalloc(&d_in_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMalloc(&d_out_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMemcpy(d_in_bufs, in_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_out_bufs, out_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));

    // Execute kernel and time it
    GPUTimer timer;
    timer.tick();
    lds_copy<Tfloat><<<ngpus, ngpus, sub_block_bytes>>>(ctx.N, ctx.ngpus, d_in_bufs, d_out_bufs);
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work
    timer.tock();

    HIP_CHECK(hipFree(d_in_bufs));
    HIP_CHECK(hipFree(d_out_bufs));
    return timer.elapsed();
}

// Consider LDS bank conflict
// Kernels on different streams?

// Performs local transposes inside of blocks. Expected to be run after
// one of the implementations performing block-wise transpose
// e.g. naiveCopy, hipMemcpy2D, hipMemcpy2DAsync
// Note: operates in the same fashion as naiveCopy, in that blocks map to gpu bufs, and threads map to sub blocks in gpu buf
template <typename Tfloat>
__global__ __launch_bounds__(1024) void local_transpose(const size_t N,
                                                        const size_t ngpus,
                                                        Tfloat**     in_bufs,
                                                        Tfloat**     out_bufs)
{
    // extern __shared__ char lds_char[][]; // Expect tile_size * tile_size
    // auto                   lds
    //     = reinterpret_cast<Tfloat*>(lds_char); // Workaround declaring extern lds for diff types
    const size_t      tile_size = 64;
    __shared__ Tfloat lds[tile_size][tile_size];

    // Determine which GPU buffers to use as input and output
    Tfloat* idata = in_bufs[blockIdx.x];
    Tfloat* odata = out_bufs[blockIdx.y];

    const size_t sub_block_size = N / ngpus; // Length of block in each transfer

    // Loop through rows in block to

    // NOTE: This would expect each group of 4 values to be within same row since it accesses contiguously
    size_t x = blockIdx.x * tile_size + threadIdx.x;
    size_t y = blockIdx.y * tile_size + threadIdx.y;

    // Read in data in coalesced fashion to lds
#pragma unroll
    for(size_t i = 0; i < sub_block_size; i++)
        lds[threadIdx.y + i][threadIdx.x] = idata[(y + i) * 4 + x];

    __syncthreads();

    // Swaperoo
    y = blockIdx.x * tile_size + threadIdx.x;
    x = blockIdx.y * tile_size + threadIdx.y;

    // Coalesced write to output from lds
#pragma unroll
    for(size_t i = 0; i < sub_block_size; i++)
    {
        odata[(y + i) * 4 + x] = lds[threadIdx.x][threadIdx.y + i];
    }
}

// Block-wide transpose + local transpose implementations
template <typename Tfloat>
float naive_copy_transpose(const benchmark_context& ctx,
                           std::vector<Tfloat*>&    in_bufs,
                           std::vector<Tfloat*>&    out_bufs)
{
    const size_t N     = ctx.N;
    const size_t ngpus = ctx.ngpus;

    // Copy over device ptrs stored in in_bufs/out_bufs into device side array
    Tfloat** d_in_bufs;
    Tfloat** d_out_bufs;
    HIP_CHECK(hipMalloc(&d_in_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMalloc(&d_out_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMemcpy(d_in_bufs, in_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_out_bufs, out_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));

    // Create intermediate tmp buffer between block transpose and local transpose
    gpubuf<Tfloat> tmp = gpubuf<Tfloat>(N, ngpus);

    // Calculate number of blocks/threads to launch with
    // For naive_copy:
    const uint32_t items_per_thread = (N * N) / (ngpus * ngpus);

    // For local_transpose:
    const uint32_t num_threads_x  = TILE_SIZE;
    const uint32_t num_threads_y  = 4; // also items per thread
    const uint32_t sub_block_size = N / ngpus;
    const uint32_t num_sub_blocks = ngpus * ngpus;
    const uint32_t num_blocks
        = ceildiv(sub_block_size * sub_block_size,
                  TILE_SIZE * TILE_SIZE); // How many tiles can fit inside a sub_block
    const dim3 dim_grid{num_blocks, 1, 1};
    const dim3 dim_block{num_threads_x, num_threads_y, 1};

    // Execute kernels and time them
    GPUTimer timer;
    timer.tick();

    naive_copy<Tfloat><<<ngpus, ngpus>>>(N, ngpus, items_per_thread, d_in_bufs, tmp.bufs);
    timer.sync_all(ngpus); // Is this needed before local_tranpose?
    local_transpose<Tfloat><<<dim_grid, dim_block>>>(N, ngpus, tmp.bufs, d_out_bufs);
    timer.sync_all(ngpus); // Ensure all GPUs have finished their work

    timer.tock();

    HIP_CHECK(hipFree(d_in_bufs));
    HIP_CHECK(hipFree(d_out_bufs));
    return timer.elapsed();
}

#include "helper.hpp"
#include <cmath>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdio.h>
#include <vector>

/**
 * Benchmarking tool for comparing speed of various memory copy methods
 * between multiple gpus. Currently will do out-of-place operations on 
 * square matrices only.
 *
 * TODO list:
 * - Implement basic implementations for each method
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
            auto idx = i + N * j;
            printf("%.6f ", input[idx]);
        }
        printf("]\n");
    }
    printf(" ]\n");
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
void host_transpose(const int N, const std::vector<Tfloat>& input, std::vector<Tfloat>& output)
{
    output.reserve(N * N);
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
template <typename Tfloat>
std::vector<Tfloat> generate(size_t N, size_t M, generator gen, Tfloat min, Tfloat max)
{
    // TODO add complex data support
    // bool is_complex = (gen == p_complex_single || gen == p_complex_double);
    std::vector<Tfloat> input(N * M);
    if(gen == h_random)
    {
        std::random_device                     rd;
        std::mt19937                           m_engine(rd()); // Mersenne Twister, rd as seed
        std::uniform_real_distribution<Tfloat> dist{min, max};
#pragma omp parallel for
        for(size_t i = 0; i < N * N; ++i)
            input[i] = dist(m_engine);
    }
    else if(gen == h_ordered)
    {
        for(size_t i = 0; i < N * N; i++)
            input[i] = static_cast<Tfloat>(i);
    }

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

/* Implementations */

// (1.1) hipMemcpy2D between two devices
template <typename Tfloat>
void run_memcpy(const benchmark_context& ctx,
                std::vector<Tfloat*>&    in_bufs,
                std::vector<Tfloat*>&    out_bufs)
{
    const size_t N              = ctx.N;
    const size_t ngpus          = ctx.ngpus;
    const size_t sub_block_size = N / ngpus; // Length of block in each transfer
    const size_t bytes_to_copy_per_row
        = sub_block_size * sizeof(Tfloat); // Bytes per row in transfer
    const size_t pitch_bytes = N * sizeof(Tfloat); // Width of buf

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
    return;
}

// (1.2) hipMemcpy2D between two devices, using stream per each gpu-gpu interaction
template <typename Tfloat>
void run_memcpy_async(const benchmark_context& ctx,
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
    return;
}

// (2) Copy kernel
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

// Handle launching of copy kernels
template <typename Tfloat>
void naive_copy_launcher(const benchmark_context& ctx,
                         std::vector<Tfloat*>&    in_bufs,
                         std::vector<Tfloat*>&    out_bufs)
{
    const size_t ngpus = ctx.ngpus;
    Tfloat**     d_in_bufs;
    Tfloat**     d_out_bufs;
    HIP_CHECK(hipMalloc(&d_in_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMalloc(&d_out_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMemcpy(d_in_bufs, in_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_out_bufs, out_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));

    size_t num_blocks  = ngpus;
    size_t num_threads = num_blocks;

    size_t items_per_thread = (ctx.N * ctx.N) / (num_blocks * num_threads);

    naive_copy<Tfloat>
        <<<num_blocks, num_threads>>>(ctx.N, ctx.ngpus, items_per_thread, d_in_bufs, d_out_bufs);
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

template <typename Tfloat>
void lds_copy_launcher(const benchmark_context& ctx,
                       std::vector<Tfloat*>&    in_bufs,
                       std::vector<Tfloat*>&    out_bufs)
{
    const size_t ngpus           = ctx.ngpus;
    const auto   sub_block_bytes = sizeof(Tfloat) * ctx.N * ctx.N / ngpus;
    Tfloat**     d_in_bufs;
    Tfloat**     d_out_bufs;
    HIP_CHECK(hipMalloc(&d_in_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMalloc(&d_out_bufs, sizeof(Tfloat*) * ngpus));
    HIP_CHECK(hipMemcpy(d_in_bufs, in_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(d_out_bufs, out_bufs.data(), sizeof(Tfloat*) * ngpus, hipMemcpyHostToDevice));
    lds_copy<Tfloat><<<ngpus, ngpus, sub_block_bytes>>>(ctx.N, ctx.ngpus, d_in_bufs, d_out_bufs);
}
// Let each thread handle up to 4 values in LDS / LDS tiling?
// Consider LDS bank conflict

// Currently basing on a stripped down rocFFT transpose kernel, needs to be redone
template <typename Tfloat>
// __launch_bounds__(1024)
__global__ void copy(const benchmark_context& ctx,
                     std::vector<Tfloat*>&    in_bufs,
                     std::vector<Tfloat*>&    out_bufs)
{
    // TODO: Should this loop over all gpus, or be called per device?
    // Adjust func signature or ctx if needed, but we can call this from a diff host func
    // What about input size vs how LDS is used => num of blocks to use...
    //     const size_t N     = ctx.N;
    //     const size_t ngpus = ctx.ngpus;
    //
    //     __shared__ Tfloat lds[64][64]; // Need to consider
    //     size_t            tile_block_idx_x  = blockIdx.x;
    //     size_t            tile_block_idx_y  = blockIdx.y;
    //     size_t            tile_thread_idx_x = threadIdx.x;
    //     size_t            tile_thread_idx_y = threadIdx.y;
    //     // Add strides
    //
    //     // Read in values from global memory
    // #pragma unroll
    //     for(size_t i = 0; i < 4; ++i)
    //     {
    //         auto logical_row = 64 * tile_block_idx_y + tile_thread_idx_y + i * 16;
    //         auto idx0        = 64 * tile_block_idx_x + tile_thread_idx_x;
    //         auto idx1        = logical_row;
    //         auto gidx        = idx0 + idx1;
    //
    //         lds[tile_thread_idx_x][i * 16 + tile_thread_idx_y] = in_bufs[?][gidx];
    //     }
    //     __syncthreads();
    //
    //     Tfloat val[4];
    //     // Realloc threads to write along fastest dim, and read transposed from LDS
    //     tile_thread_idx_x = tile_thread_idx_y;
    //
}

// Handle launching of copy kernels
template <typename Tfloat>
void copy_kernel_launcher(const benchmark_context& ctx,
                          std::vector<Tfloat*>&    in_bufs,
                          std::vector<Tfloat*>&    out_bufs)
{
    // Experiment with streams!
}

// (3.1) MPI alltoall
// (3.2) MPI alltoallv

// (4) RCCL alltoall

#include <cmath>
#include <cstring>
#include <ctime>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <vector>

// Collection of data generation, correctness verification, structs, etc.

#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        hipError_t error = (cmd);                                                              \
        if(error != hipSuccess)                                                                \
        {                                                                                      \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";                          \
            exit(-1);                                                                          \
        }                                                                                      \
    } while(0)

inline size_t ceildiv(const size_t numerator, const size_t divisor)
{
    return (numerator + divisor - 1) / divisor;
}

inline bool is_power_of_two(const size_t n)
{
    if(n == 0)
        return false;
    return (n & (n - 1)) == 0;
}

inline size_t min(const size_t n, const size_t m)
{
    return ((n < m) ? n : m);
}

enum precision
{
    p_single,
    p_double,
    p_complex_single,
    p_complex_double,
};

enum generator
{
    gen_random,
    gen_ordered,
};

// Hold data useful for benchmarks being run
struct benchmark_context
{
    size_t                   N;
    size_t                   ngpus;
    size_t                   blocks;
    size_t                   threads;
    int                      verbose;
    bool                     verify_results = false;
    int                      mpi_size       = 0;
    std::vector<hipStream_t> streams;
};

// TODO: Can replace usages of raw hipMalloc/hipFree in src/membench.hpp
// RAII struct for single device buffer
template <typename Tfloat>
class gpubuf
{
private:
    size_t  N;
    Tfloat* buf;
    int     device = 0; // May want to use to determine where buf to switch to if needed

public:
    gpubuf(size_t N_)
        : N(N_)
    {
        HIP_CHECK(hipMalloc(&buf, sizeof(Tfloat) * N));
    }

    ~gpubuf()
    {
        HIP_CHECK(hipFree(buf));
    }

    Tfloat* data()
    {
        return buf;
    }

    // Read-only variant
    const Tfloat* data() const
    {
        return buf;
    }

    size_t size() const
    {
        return N;
    }
};

// RAII struct for temporary buffers for intermediate results
// Used when performing multiple out-of-place operations
// , specifically when storing all device ptrs together
template <typename Tfloat>
struct gpubuf_vec
{
    size_t   N;
    size_t   ngpus;
    Tfloat** bufs;

    gpubuf_vec(size_t N_, size_t ngpus_)
        : N(N_)
        , ngpus(ngpus_)
    {
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMalloc(&bufs, sizeof(Tfloat*) * ngpus));
        const size_t buf_elems = N * N / ngpus;
        for(auto i = 0; i < ngpus; i++)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMalloc(&bufs[i], sizeof(Tfloat) * buf_elems));
            HIP_CHECK(hipMemset(bufs[i], 0, sizeof(Tfloat) * buf_elems));
        }
        HIP_CHECK(hipSetDevice(0));
    }

    ~gpubuf_vec()
    {
        for(auto i = 0; i < ngpus; i++)
            HIP_CHECK(hipFree(bufs[i]));
        HIP_CHECK(hipFree(bufs));
    }
};

// RAII wrapper around hipEvent API for timing
struct GPUTimer
{
    hipEvent_t start, stop;

    GPUTimer()
    {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }

    ~GPUTimer()
    {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void tick()
    {
        HIP_CHECK(hipEventRecord(start, 0));
    }

    void tock()
    {
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
    }

    float elapsed()
    {
        float elapsed;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }

    void sync_all(size_t ngpus)
    {
        for(size_t i = 0; i < ngpus; i++)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
};

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
void assemble_output_to_host(const int N, const int ngpus, Tfloat** gpubufs, Tfloat* hostbuf_result)
{
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

// Reference impl of block transpose on CPU that does not perform local transpose (out-of-place)
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

// Used for CLI11 parsing of precision enum option
static bool lexical_cast(const std::string& word, precision& p)
{
    if(word == "single" || word == "0")
        p = p_single;
    else if(word == "double" || word == "1")
        p = p_double;
    else if(word == "c_single" || word == "2")
        p = p_complex_single;
    else if(word == "c_double" || word == "3")
        p = p_complex_double;
    else
        throw std::runtime_error("Invalid precision specified");
    return true;
}

// Used for CLI11 parsing of input gen enum option
static bool lexical_cast(const std::string& word, generator& gen)
{
    if(word == "random" || word == "0")
        gen = gen_random;
    else if(word == "ordered" || word == "1")
        gen = gen_ordered;
    else
        throw std::runtime_error("Invalid input generator specified");
    return true;
}

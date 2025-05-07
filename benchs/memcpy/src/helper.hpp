#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#ifdef MPI_ENABLED
#include <mpi.h>

inline MPI_Datatype get_mpi_type(size_t elem_size)
{
    MPI_Datatype mpi_type;
    if(elem_size == 4) // Real FP32
        mpi_type = MPI_FLOAT;
    else if(elem_size == 8) // Complex FP32 or Real FP64
        mpi_type = MPI_DOUBLE;
    else if(elem_size == 16) // Complex FP64
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    else
        throw std::runtime_error("Invalid element size for MPI");
    return mpi_type;
}
#endif

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

// Hold data useful for benchmarks being run
struct benchmark_context
{
    size_t                   N;
    size_t                   ngpus;
    size_t                   blocks;
    size_t                   threads;
    int                      verbose;
    bool                     verify_results = false;
    std::vector<hipStream_t> streams;
};

// Wrapper around hipEvent API for timing
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

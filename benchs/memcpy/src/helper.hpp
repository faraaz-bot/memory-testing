#include <hip/hip_runtime.h>
#include <vector>

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

// Hold data useful for benchmarks being run
struct benchmark_context
{
    size_t                   N;
    size_t                   ngpus;
    size_t                   blocks;
    size_t                   threads;
    int                      verbose;
    bool                     verify_results;
    std::vector<hipStream_t> streams;
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
    h_random,
    h_ordered,
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
    if(word == "h_random" || word == "0")
        gen = h_random;
    else if(word == "h_ordered" || word == "1")
        gen = h_ordered;
    else
        throw std::runtime_error("Invalid input generator specified");
    return true;
}

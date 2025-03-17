#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                                  \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)

enum precision
{
    p_half,
    p_single,
    p_double,
};

enum generator
{
    h_random,
    h_ordered,
};

// TODO figure out way to easily toggle benchmarks to run
// enum copy_operation
// {
//     memcpy_async,
//     mpi_alltoall,
//     copy_kernel,
//     all,
// };

// Used for CLI11 parsing of input gen enum
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


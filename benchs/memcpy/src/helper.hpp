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

typedef enum precision_type
{
    p_half,
    p_single,
    p_double,
} precision;

typedef enum generators
{
    h_random,
    h_ordered,
} generator

// TODO figure out way to easily toggle benchmarks to run
// enum copy_operation
// {
//     memcpy_async,
//     mpi_alltoall,
//     copy_kernel,
//     all,
// };

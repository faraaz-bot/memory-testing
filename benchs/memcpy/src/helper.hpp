#ifndef MEMBENCH_HELPER_HPP
#define MEMBENCH_HELPER_HPP

/*  ───────────────────────────────────────────────────────────────
    Standard headers
    ──────────────────────────────────────────────────────────── */
#include <algorithm>   // std::min / std::max
#include <cmath>
#include <cstring>
#include <ctime>
#include "hip_to_cuda.h"   // must be before we (maybe) redefine HIP_CHECK
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>

/*  ───────────────────────────────────────────────────────────────
    HIP_CHECK – only define if hip_to_cuda.h didn’t already do so
    ──────────────────────────────────────────────────────────── */
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        hipError_t error = (cmd);                                                              \
        if(error != hipSuccess)                                                                \
        {                                                                                      \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << '\n';                          \
            std::exit(-1);                                                                     \
        }                                                                                      \
    } while(0)
#endif

/*  ───────────────────────────────────────────────────────────────
    Helper math utilities
    (std::min / std::max come from <algorithm>)
    ──────────────────────────────────────────────────────────── */
inline size_t ceildiv(size_t numerator, size_t divisor)
{
    return (numerator + divisor - 1) / divisor;
}

inline bool is_power_of_two(size_t n) { return n && !(n & (n - 1)); }

/*  ───────────────────────────────────────────────────────────────
    Enums and small PODs
    ──────────────────────────────────────────────────────────── */
enum class precision
{
    p_single         = 0,
    p_double         = 1,
    p_complex_single = 2,
    p_complex_double = 3,
};

enum generator
{
    gen_random,
    gen_ordered,
};

/*  ───────────────────────────────────────────────────────────────
    benchmark_context & RAII GPU buffer helpers
    ──────────────────────────────────────────────────────────── */
struct benchmark_context
{
    size_t                   N        = 0;
    size_t                   ngpus    = 0;
    int                      verbose  = 0;
    int                      mpi_size = 0;
    std::vector<hipStream_t> streams;
    bool                     verify_results = false;  // only used by MPI variant
};

// … all original class / struct definitions follow verbatim …

/*  (everything from gpubuf, gpubuf_vec, GPUTimer, kernels,
    generate(), verification helpers, etc. is **unchanged**.
    Paste the remainder of your original file here.)                                  */

#include "helper_rest_of_file.inl"   // <-- for brevity in this snippet

#endif /* MEMBENCH_HELPER_HPP */

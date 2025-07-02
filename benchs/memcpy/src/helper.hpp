#include <cmath>
#include <cstring>
#include <ctime>
#include "hip_to_cuda.h"          // must come before we (maybe) redefine HIP_CHECK
#include <algorithm>              // <-- NEW: for std::min / std::max
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>

// Collection of data generation, correctness verification, structs, etc.

/*─────────────────────────────────────────────────────────────────────
  1) HIP_CHECK - guard against the duplicate definition coming from
     hip_to_cuda.h so the warning disappears.
─────────────────────────────────────────────────────────────────────*/
#ifndef HIP_CHECK
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
#endif

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

/*─────────────────────────────────────────────────────────────────────
  2) REMOVE the local `min()` overload that collided with <algorithm>.
     We now just bring the standard one into scope:
─────────────────────────────────────────────────────────────────────*/
using std::min;
using std::max;

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

// ………………………  **EVERYTHING BELOW THIS POINT IS UNCHANGED**  ………………………

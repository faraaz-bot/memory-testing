#ifndef LAUNCH_COMMON_H
#define LAUNCH_COMMON_H

#include "hip/hip_runtime.h"

#include "rocfft/kargs.h"
#include "rocfft/callback.h"
#include "rocfft/callback.h"
#include "rocfft/arithmetic.h"

#define LDS_BANK_SHIFT 32

struct GridParam
{
    unsigned int b_x, b_y, b_z; // in HIP, the data type of dimensions of work
    // items, work groups is unsigned int
    unsigned int wgs_x, wgs_y, wgs_z;
    unsigned int lds_bytes; // dynamic LDS allocation size

    GridParam()
        : b_x(1), b_y(1), b_z(1), wgs_x(1), wgs_y(1), wgs_z(1), lds_bytes(0)
    {
    }
};

#endif
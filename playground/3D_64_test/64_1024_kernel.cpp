//
// build: /opt/rocm/bin/hipcc -std=c++14  64_1024_kernel.cpp -o 64_1024_kernel -lfftw3f
//
// run:
//    - vkFFT: 64_1024_kernel 0
//    - rocFFT: 64_1024_kernel 1
//    - both: 64_1024_kernel 2
//

#include "butterfly_constant.h"
#include "callback.h"
#include "common.h"
#include "runtime_api_wrapper.h"
#include <complex.h>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int get_1d_global_idx()
{

    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x)
                   + threadIdx.x;

    return threadId;
}

const float             loc_PI      = 3.1415926535897932384626433832795f;
const float             loc_SQRT1_2 = 0.70710678118654752440084436210485f;
extern __shared__ float shared[];
template <typename T>
struct Inputs
{
    const T*          buffer;
    inline __device__ Inputs(const T* buffer)
        : buffer(buffer)
    {
    }
    inline __device__ T operator[](unsigned long long idx) const
    {
        return buffer[idx];
    }
    inline __device__ T operator[](unsigned int idx) const
    {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer)
                                           + idx * static_cast<unsigned int>(sizeof(T)));
    }
};
template <typename T>
struct Outputs
{
    T*                buffer;
    inline __device__ Outputs(T* buffer)
        : buffer(buffer)
    {
    }
    inline __device__ T& operator[](unsigned long long idx)
    {
        return buffer[idx];
    }
    inline __device__ T& operator[](unsigned int idx)
    {
        return *reinterpret_cast<T*>(reinterpret_cast<char*>(buffer)
                                     + idx * static_cast<unsigned int>(sizeof(T)));
    }
};
extern "C" __launch_bounds__(128) __global__ void VkFFT_main(const float2* inputs, float2* outputs)
{
    unsigned int sharedStride = 17;
    float2*      sdata        = (float2*)shared;

    float2 temp_0;
    temp_0.x = 0;
    temp_0.y = 0;
    float2 temp_1;
    temp_1.x = 0;
    temp_1.y = 0;
    float2 temp_2;
    temp_2.x = 0;
    temp_2.y = 0;
    float2 temp_3;
    temp_3.x = 0;
    temp_3.y = 0;
    float2 temp_4;
    temp_4.x = 0;
    temp_4.y = 0;
    float2 temp_5;
    temp_5.x = 0;
    temp_5.y = 0;
    float2 temp_6;
    temp_6.x = 0;
    temp_6.y = 0;
    float2 temp_7;
    temp_7.x = 0;
    temp_7.y = 0;
    float2 w;
    w.x = 0;
    w.y = 0;
    float2 loc_0;
    loc_0.x = 0;
    loc_0.y = 0;
    float2 iw;
    iw.x                           = 0;
    iw.y                           = 0;
    unsigned int stageInvocationID = 0;
    unsigned int blockInvocationID = 0;
    unsigned int sdataID           = 0;
    unsigned int combinedID        = 0;
    unsigned int inoutID           = 0;
    float        angle             = 0;
    {
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 0;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 128;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 256;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 384;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 512;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 640;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 768;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 896;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
    }
    __syncthreads();

    stageInvocationID = (threadIdx.y + 0) % (1);
    angle             = stageInvocationID * -3.14159265358979312e+00f;
    temp_0            = sdata[sharedStride * (threadIdx.y + 0) + threadIdx.x];
    temp_1            = sdata[sharedStride * (threadIdx.y + 8) + threadIdx.x];
    temp_2            = sdata[sharedStride * (threadIdx.y + 16) + threadIdx.x];
    temp_3            = sdata[sharedStride * (threadIdx.y + 24) + threadIdx.x];
    temp_4            = sdata[sharedStride * (threadIdx.y + 32) + threadIdx.x];
    temp_5            = sdata[sharedStride * (threadIdx.y + 40) + threadIdx.x];
    temp_6            = sdata[sharedStride * (threadIdx.y + 48) + threadIdx.x];
    temp_7            = sdata[sharedStride * (threadIdx.y + 56) + threadIdx.x];

    // if(threadIdx.x == 1 && blockIdx.y == 0)
    //     printf("threadIdx.y %d, inputs: (%2d, %2d), (%2d, %2d), (%2d, %2d), (%2d, %2d)\n",
    //            (int)threadIdx.y,
    //            (int)temp_0.x,
    //            (int)temp_0.y,
    //            (int)temp_1.x,
    //            (int)temp_1.y,
    //            (int)temp_2.x,
    //            (int)temp_2.y,
    //            (int)temp_3.x,
    //            (int)temp_3.y);

    w.x    = __cosf(angle);
    w.y    = __sinf(angle);
    loc_0  = temp_4 * w.x + float2(-temp_4.y, temp_4.x) * w.y;
    temp_4 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;
    loc_0  = temp_5 * w.x + float2(-temp_5.y, temp_5.x) * w.y;
    temp_5 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;
    loc_0  = temp_6 * w.x + float2(-temp_6.y, temp_6.x) * w.y;
    temp_6 = temp_2 - loc_0;
    temp_2 = temp_2 + loc_0;
    loc_0  = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7 = temp_3 - loc_0;
    temp_3 = temp_3 + loc_0;
    w.x    = __cosf(0.5f * angle);
    w.y    = __sinf(0.5f * angle);
    loc_0  = temp_2 * w.x + float2(-temp_2.y, temp_2.x) * w.y;
    temp_2 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;
    loc_0  = temp_3 * w.x + float2(-temp_3.y, temp_3.x) * w.y;
    temp_3 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;
    iw.x   = w.y;
    iw.y   = -w.x;
    loc_0  = temp_6 * iw.x + float2(-temp_6.y, temp_6.x) * iw.y;
    temp_6 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;
    loc_0  = temp_7 * iw.x + float2(-temp_7.y, temp_7.x) * iw.y;
    temp_7 = temp_5 - loc_0;
    temp_5 = temp_5 + loc_0;
    w.x    = __cosf(0.25f * angle);
    w.y    = __sinf(0.25f * angle);
    loc_0  = temp_1 * w.x + float2(-temp_1.y, temp_1.x) * w.y;
    temp_1 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;
    iw.x   = w.y;
    iw.y   = -w.x;
    loc_0  = temp_3 * iw.x + float2(-temp_3.y, temp_3.x) * iw.y;
    temp_3 = temp_2 - loc_0;
    temp_2 = temp_2 + loc_0;
    iw.x   = w.x * loc_SQRT1_2 + w.y * loc_SQRT1_2;
    iw.y   = w.y * loc_SQRT1_2 - w.x * loc_SQRT1_2;

    loc_0        = temp_5 * iw.x + float2(-temp_5.y, temp_5.x) * iw.y;
    temp_5       = temp_4 - loc_0;
    temp_4       = temp_4 + loc_0;
    w.x          = iw.y;
    w.y          = -iw.x;
    loc_0        = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7       = temp_6 - loc_0;
    temp_6       = temp_6 + loc_0;
    loc_0        = temp_1;
    temp_1       = temp_4;
    temp_4       = loc_0;
    loc_0        = temp_3;
    temp_3       = temp_6;
    temp_6       = loc_0;
    sharedStride = 16;
    __syncthreads();

    stageInvocationID = threadIdx.y + 0;
    blockInvocationID = stageInvocationID;
    stageInvocationID = stageInvocationID % 1;
    blockInvocationID = blockInvocationID - stageInvocationID;
    inoutID           = blockInvocationID * 8;
    inoutID           = inoutID + stageInvocationID;
    sdataID           = inoutID + 0;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_0;
    sdataID           = inoutID + 1;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_1;
    sdataID           = inoutID + 2;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_2;
    sdataID           = inoutID + 3;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_3;
    sdataID           = inoutID + 4;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_4;
    sdataID           = inoutID + 5;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_5;
    sdataID           = inoutID + 6;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_6;
    sdataID           = inoutID + 7;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_7;
    __syncthreads();

    stageInvocationID = (threadIdx.y + 0) % (8);
    angle             = stageInvocationID * -3.92699081698724139e-01f;
    temp_0            = sdata[sharedStride * (threadIdx.y + 0) + threadIdx.x];
    temp_1            = sdata[sharedStride * (threadIdx.y + 8) + threadIdx.x];
    temp_2            = sdata[sharedStride * (threadIdx.y + 16) + threadIdx.x];
    temp_3            = sdata[sharedStride * (threadIdx.y + 24) + threadIdx.x];
    temp_4            = sdata[sharedStride * (threadIdx.y + 32) + threadIdx.x];
    temp_5            = sdata[sharedStride * (threadIdx.y + 40) + threadIdx.x];
    temp_6            = sdata[sharedStride * (threadIdx.y + 48) + threadIdx.x];
    temp_7            = sdata[sharedStride * (threadIdx.y + 56) + threadIdx.x];
    w.x               = __cosf(angle);
    w.y               = __sinf(angle);
    loc_0             = temp_4 * w.x + float2(-temp_4.y, temp_4.x) * w.y;
    temp_4            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    loc_0             = temp_5 * w.x + float2(-temp_5.y, temp_5.x) * w.y;
    temp_5            = temp_1 - loc_0;
    temp_1            = temp_1 + loc_0;
    loc_0             = temp_6 * w.x + float2(-temp_6.y, temp_6.x) * w.y;
    temp_6            = temp_2 - loc_0;
    temp_2            = temp_2 + loc_0;
    loc_0             = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7            = temp_3 - loc_0;
    temp_3            = temp_3 + loc_0;
    w.x               = __cosf(0.5f * angle);
    w.y               = __sinf(0.5f * angle);
    loc_0             = temp_2 * w.x + float2(-temp_2.y, temp_2.x) * w.y;
    temp_2            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    loc_0             = temp_3 * w.x + float2(-temp_3.y, temp_3.x) * w.y;
    temp_3            = temp_1 - loc_0;
    temp_1            = temp_1 + loc_0;
    iw.x              = w.y;
    iw.y              = -w.x;
    loc_0             = temp_6 * iw.x + float2(-temp_6.y, temp_6.x) * iw.y;
    temp_6            = temp_4 - loc_0;
    temp_4            = temp_4 + loc_0;
    loc_0             = temp_7 * iw.x + float2(-temp_7.y, temp_7.x) * iw.y;
    temp_7            = temp_5 - loc_0;
    temp_5            = temp_5 + loc_0;
    w.x               = __cosf(0.25f * angle);
    w.y               = __sinf(0.25f * angle);
    loc_0             = temp_1 * w.x + float2(-temp_1.y, temp_1.x) * w.y;
    temp_1            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    iw.x              = w.y;
    iw.y              = -w.x;
    loc_0             = temp_3 * iw.x + float2(-temp_3.y, temp_3.x) * iw.y;
    temp_3            = temp_2 - loc_0;
    temp_2            = temp_2 + loc_0;
    iw.x              = w.x * loc_SQRT1_2 + w.y * loc_SQRT1_2;
    iw.y              = w.y * loc_SQRT1_2 - w.x * loc_SQRT1_2;

    loc_0  = temp_5 * iw.x + float2(-temp_5.y, temp_5.x) * iw.y;
    temp_5 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;
    w.x    = iw.y;
    w.y    = -iw.x;
    loc_0  = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7 = temp_6 - loc_0;
    temp_6 = temp_6 + loc_0;
    loc_0  = temp_1;
    temp_1 = temp_4;
    temp_4 = loc_0;
    loc_0  = temp_3;
    temp_3 = temp_6;
    temp_6 = loc_0;
    __syncthreads();

    sharedStride      = 17;
    stageInvocationID = threadIdx.y + 0;
    blockInvocationID = stageInvocationID;
    stageInvocationID = stageInvocationID % 8;
    blockInvocationID = blockInvocationID - stageInvocationID;
    inoutID           = blockInvocationID * 8;
    inoutID           = inoutID + stageInvocationID;
    sdataID           = inoutID + 0;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_0;
    sdataID           = inoutID + 8;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_1;
    sdataID           = inoutID + 16;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_2;
    sdataID           = inoutID + 24;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_3;
    sdataID           = inoutID + 32;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_4;
    sdataID           = inoutID + 40;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_5;
    sdataID           = inoutID + 48;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_6;
    sdataID           = inoutID + 56;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_7;
    __syncthreads();

    {
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 0;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 128;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 256;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 384;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 512;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 640;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 768;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 896;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
    }
}

/* inverse
const float loc_PI      = 3.1415926535897932384626433832795f;
const float loc_SQRT1_2 = 0.70710678118654752440084436210485f;
typedef struct
{
    unsigned int workGroupShiftX;
    unsigned int workGroupShiftY;
    unsigned int workGroupShiftZ;
} PushConsts;
__constant__ PushConsts consts;
extern __shared__ float shared[];
extern "C" __launch_bounds__(128) __global__ void VkFFT_main(float2* inputs, float2* outputs)
{
    unsigned int sharedStride = 17;
    float2*      sdata        = (float2*)shared;

    float2 temp_0;
    temp_0.x = 0;
    temp_0.y = 0;
    float2 temp_1;
    temp_1.x = 0;
    temp_1.y = 0;
    float2 temp_2;
    temp_2.x = 0;
    temp_2.y = 0;
    float2 temp_3;
    temp_3.x = 0;
    temp_3.y = 0;
    float2 temp_4;
    temp_4.x = 0;
    temp_4.y = 0;
    float2 temp_5;
    temp_5.x = 0;
    temp_5.y = 0;
    float2 temp_6;
    temp_6.x = 0;
    temp_6.y = 0;
    float2 temp_7;
    temp_7.x = 0;
    temp_7.y = 0;
    float2 w;
    w.x = 0;
    w.y = 0;
    float2 loc_0;
    loc_0.x = 0;
    loc_0.y = 0;
    float2 iw;
    iw.x                           = 0;
    iw.y                           = 0;
    unsigned int stageInvocationID = 0;
    unsigned int blockInvocationID = 0;
    unsigned int sdataID           = 0;
    unsigned int combinedID        = 0;
    unsigned int inoutID           = 0;
    float        angle             = 0;
    {
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 0;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 128;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 256;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 384;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 512;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 640;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 768;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
        combinedID = (threadIdx.x + 16 * threadIdx.y) + 896;
        inoutID    = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID    = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        sdata[(combinedID % 64) * sharedStride + (combinedID / 64)] = inputs[inoutID];
    }
    __syncthreads();

    stageInvocationID = (threadIdx.y + 0) % (1);
    angle             = stageInvocationID * 3.14159265358979312e+00f;
    temp_0            = sdata[sharedStride * (threadIdx.y + 0) + threadIdx.x];
    temp_1            = sdata[sharedStride * (threadIdx.y + 8) + threadIdx.x];
    temp_2            = sdata[sharedStride * (threadIdx.y + 16) + threadIdx.x];
    temp_3            = sdata[sharedStride * (threadIdx.y + 24) + threadIdx.x];
    temp_4            = sdata[sharedStride * (threadIdx.y + 32) + threadIdx.x];
    temp_5            = sdata[sharedStride * (threadIdx.y + 40) + threadIdx.x];
    temp_6            = sdata[sharedStride * (threadIdx.y + 48) + threadIdx.x];
    temp_7            = sdata[sharedStride * (threadIdx.y + 56) + threadIdx.x];
    w.x               = __cosf(angle);
    w.y               = __sinf(angle);
    loc_0             = temp_4 * w.x + float2(-temp_4.y, temp_4.x) * w.y;
    temp_4            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    loc_0             = temp_5 * w.x + float2(-temp_5.y, temp_5.x) * w.y;
    temp_5            = temp_1 - loc_0;
    temp_1            = temp_1 + loc_0;
    loc_0             = temp_6 * w.x + float2(-temp_6.y, temp_6.x) * w.y;
    temp_6            = temp_2 - loc_0;
    temp_2            = temp_2 + loc_0;
    loc_0             = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7            = temp_3 - loc_0;
    temp_3            = temp_3 + loc_0;
    w.x               = __cosf(0.5f * angle);
    w.y               = __sinf(0.5f * angle);
    loc_0             = temp_2 * w.x + float2(-temp_2.y, temp_2.x) * w.y;
    temp_2            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    loc_0             = temp_3 * w.x + float2(-temp_3.y, temp_3.x) * w.y;
    temp_3            = temp_1 - loc_0;
    temp_1            = temp_1 + loc_0;
    iw.x              = -w.y;
    iw.y              = w.x;
    loc_0             = temp_6 * iw.x + float2(-temp_6.y, temp_6.x) * iw.y;
    temp_6            = temp_4 - loc_0;
    temp_4            = temp_4 + loc_0;
    loc_0             = temp_7 * iw.x + float2(-temp_7.y, temp_7.x) * iw.y;
    temp_7            = temp_5 - loc_0;
    temp_5            = temp_5 + loc_0;
    w.x               = __cosf(0.25f * angle);
    w.y               = __sinf(0.25f * angle);
    loc_0             = temp_1 * w.x + float2(-temp_1.y, temp_1.x) * w.y;
    temp_1            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    iw.x              = -w.y;
    iw.y              = w.x;
    loc_0             = temp_3 * iw.x + float2(-temp_3.y, temp_3.x) * iw.y;
    temp_3            = temp_2 - loc_0;
    temp_2            = temp_2 + loc_0;
    iw.x              = w.x * loc_SQRT1_2 - w.y * loc_SQRT1_2;
    iw.y              = w.y * loc_SQRT1_2 + w.x * loc_SQRT1_2;

    loc_0        = temp_5 * iw.x + float2(-temp_5.y, temp_5.x) * iw.y;
    temp_5       = temp_4 - loc_0;
    temp_4       = temp_4 + loc_0;
    w.x          = -iw.y;
    w.y          = iw.x;
    loc_0        = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7       = temp_6 - loc_0;
    temp_6       = temp_6 + loc_0;
    loc_0        = temp_1;
    temp_1       = temp_4;
    temp_4       = loc_0;
    loc_0        = temp_3;
    temp_3       = temp_6;
    temp_6       = loc_0;
    sharedStride = 17;
    __syncthreads();

    stageInvocationID = threadIdx.y + 0;
    blockInvocationID = stageInvocationID;
    stageInvocationID = stageInvocationID % 1;
    blockInvocationID = blockInvocationID - stageInvocationID;
    inoutID           = blockInvocationID * 8;
    inoutID           = inoutID + stageInvocationID;
    sdataID           = inoutID + 0;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_0;
    sdataID           = inoutID + 1;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_1;
    sdataID           = inoutID + 2;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_2;
    sdataID           = inoutID + 3;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_3;
    sdataID           = inoutID + 4;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_4;
    sdataID           = inoutID + 5;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_5;
    sdataID           = inoutID + 6;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_6;
    sdataID           = inoutID + 7;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_7;
    __syncthreads();

    stageInvocationID = (threadIdx.y + 0) % (8);
    angle             = stageInvocationID * 3.92699081698724139e-01f;
    temp_0            = sdata[sharedStride * (threadIdx.y + 0) + threadIdx.x];
    temp_1            = sdata[sharedStride * (threadIdx.y + 8) + threadIdx.x];
    temp_2            = sdata[sharedStride * (threadIdx.y + 16) + threadIdx.x];
    temp_3            = sdata[sharedStride * (threadIdx.y + 24) + threadIdx.x];
    temp_4            = sdata[sharedStride * (threadIdx.y + 32) + threadIdx.x];
    temp_5            = sdata[sharedStride * (threadIdx.y + 40) + threadIdx.x];
    temp_6            = sdata[sharedStride * (threadIdx.y + 48) + threadIdx.x];
    temp_7            = sdata[sharedStride * (threadIdx.y + 56) + threadIdx.x];
    w.x               = __cosf(angle);
    w.y               = __sinf(angle);
    loc_0             = temp_4 * w.x + float2(-temp_4.y, temp_4.x) * w.y;
    temp_4            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    loc_0             = temp_5 * w.x + float2(-temp_5.y, temp_5.x) * w.y;
    temp_5            = temp_1 - loc_0;
    temp_1            = temp_1 + loc_0;
    loc_0             = temp_6 * w.x + float2(-temp_6.y, temp_6.x) * w.y;
    temp_6            = temp_2 - loc_0;
    temp_2            = temp_2 + loc_0;
    loc_0             = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7            = temp_3 - loc_0;
    temp_3            = temp_3 + loc_0;
    w.x               = __cosf(0.5f * angle);
    w.y               = __sinf(0.5f * angle);
    loc_0             = temp_2 * w.x + float2(-temp_2.y, temp_2.x) * w.y;
    temp_2            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    loc_0             = temp_3 * w.x + float2(-temp_3.y, temp_3.x) * w.y;
    temp_3            = temp_1 - loc_0;
    temp_1            = temp_1 + loc_0;
    iw.x              = -w.y;
    iw.y              = w.x;
    loc_0             = temp_6 * iw.x + float2(-temp_6.y, temp_6.x) * iw.y;
    temp_6            = temp_4 - loc_0;
    temp_4            = temp_4 + loc_0;
    loc_0             = temp_7 * iw.x + float2(-temp_7.y, temp_7.x) * iw.y;
    temp_7            = temp_5 - loc_0;
    temp_5            = temp_5 + loc_0;
    w.x               = __cosf(0.25f * angle);
    w.y               = __sinf(0.25f * angle);
    loc_0             = temp_1 * w.x + float2(-temp_1.y, temp_1.x) * w.y;
    temp_1            = temp_0 - loc_0;
    temp_0            = temp_0 + loc_0;
    iw.x              = -w.y;
    iw.y              = w.x;
    loc_0             = temp_3 * iw.x + float2(-temp_3.y, temp_3.x) * iw.y;
    temp_3            = temp_2 - loc_0;
    temp_2            = temp_2 + loc_0;
    iw.x              = w.x * loc_SQRT1_2 - w.y * loc_SQRT1_2;
    iw.y              = w.y * loc_SQRT1_2 + w.x * loc_SQRT1_2;

    loc_0  = temp_5 * iw.x + float2(-temp_5.y, temp_5.x) * iw.y;
    temp_5 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;
    w.x    = -iw.y;
    w.y    = iw.x;
    loc_0  = temp_7 * w.x + float2(-temp_7.y, temp_7.x) * w.y;
    temp_7 = temp_6 - loc_0;
    temp_6 = temp_6 + loc_0;
    loc_0  = temp_1;
    temp_1 = temp_4;
    temp_4 = loc_0;
    loc_0  = temp_3;
    temp_3 = temp_6;
    temp_6 = loc_0;
    __syncthreads();

    sharedStride      = 16;
    stageInvocationID = threadIdx.y + 0;
    blockInvocationID = stageInvocationID;
    stageInvocationID = stageInvocationID % 8;
    blockInvocationID = blockInvocationID - stageInvocationID;
    inoutID           = blockInvocationID * 8;
    inoutID           = inoutID + stageInvocationID;
    sdataID           = inoutID + 0;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_0;
    sdataID           = inoutID + 8;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_1;
    sdataID           = inoutID + 16;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_2;
    sdataID           = inoutID + 24;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_3;
    sdataID           = inoutID + 32;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_4;
    sdataID           = inoutID + 40;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_5;
    sdataID           = inoutID + 48;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_6;
    sdataID           = inoutID + 56;
    sdataID           = sharedStride * sdataID;
    sdataID           = sdataID + threadIdx.x;
    sdata[sdataID]    = temp_7;
    __syncthreads();

    {
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 0;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 128;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 256;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 384;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 512;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 640;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 768;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
        combinedID       = (threadIdx.x + 16 * threadIdx.y) + 896;
        inoutID          = (combinedID % 64) + (combinedID / 64) * 64;
        inoutID          = (inoutID) + blockIdx.y * 1024 + ((0 + blockIdx.z * 1) / 1) * 64;
        outputs[inoutID] = sdata[(combinedID % 64) * sharedStride + (combinedID / 64)];
    }
}
*/

// #define ROCFFT_ORG

template <typename scalar_type,
          const bool lds_is_real,
          StrideBin  sb,
          const bool lds_linear,
          const bool direct_load_to_reg>
__device__ void forward_length64_SBRR_device(scalar_type* R,
                                             real_type_t<scalar_type>* __restrict__ lds_real,
                                             scalar_type* __restrict__ lds_complex,
                                             const scalar_type* __restrict__ twiddles,
                                             unsigned int stride_lds,
                                             unsigned int offset_lds,
                                             unsigned int thread,
                                             bool         write)
{

#ifdef ROCFFT_ORG
    scalar_type        W;
    scalar_type        t;
    const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    unsigned int       l_offset;

    // pass 0, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
    if(!lds_is_real)
    {
        if(!direct_load_to_reg)
        {
            __syncthreads();
        }

        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[0];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[1];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[2];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[3];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[4];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[5];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[6];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_complex[l_offset] = R[7];
    }

    else
    {
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[0].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[1].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[2].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[3].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[4].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[5].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[6].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[7].x;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[0].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[1].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[2].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[3].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[4].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[5].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[6].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[7].x   = lds_real[l_offset];
        __syncthreads();
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[0].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[1].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[2].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[3].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[4].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[5].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[6].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
        l_offset = l_offset + l_offset / 32;
        lds_real[l_offset] = R[7].y;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[0].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[1].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[2].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[3].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[4].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[5].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[6].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[7].y   = lds_real[l_offset];
    }

    // pass 1, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    if(!lds_is_real)
    {
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[0]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[1]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[2]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[3]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[4]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[5]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[6]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
        l_offset = l_offset + l_offset / 32;
        R[7]     = lds_complex[l_offset];
    }

    W    = twiddles[0 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
    R[1] = t;
    W    = twiddles[1 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
    R[2] = t;
    W    = twiddles[2 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
    R[3] = t;
    W    = twiddles[3 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
    R[4] = t;
    W    = twiddles[4 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[5].x * W.x - R[5].y * W.y, R[5].y * W.x + R[5].x * W.y};
    R[5] = t;
    W    = twiddles[5 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[6].x * W.x - R[6].y * W.y, R[6].y * W.x + R[6].x * W.y};
    R[6] = t;
    W    = twiddles[6 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[7].x * W.x - R[7].y * W.y, R[7].y * W.x + R[7].x * W.y};
    R[7] = t;
    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
#else
    scalar_type w;
    w.x = 0;
    w.y = 0;
    scalar_type loc_0;
    loc_0.x = 0;
    loc_0.y = 0;
    scalar_type iw;
    iw.x = 0;
    iw.y = 0;

    unsigned int stageInvocationID = 0;
    unsigned int blockInvocationID = 0;
    unsigned int sdataID           = 0;
    unsigned int combinedID        = 0;
    unsigned int inoutID           = 0;
    float        angle             = 0;
    int          threadIdx_y       = threadIdx.x % 8;
    int          threadIdx_x       = threadIdx.x / 8;

    stageInvocationID = (threadIdx_y + 0) % (1);
    angle             = stageInvocationID * -3.14159265358979312e+00f;

    w.x   = __cosf(angle);
    w.y   = __sinf(angle);
    loc_0 = R[4] * w.x + scalar_type(-R[4].y, R[4].x) * w.y;
    R[4]  = R[0] - loc_0;
    R[0]  = R[0] + loc_0;
    loc_0 = R[5] * w.x + scalar_type(-R[5].y, R[5].x) * w.y;
    R[5]  = R[1] - loc_0;
    R[1]  = R[1] + loc_0;
    loc_0 = R[6] * w.x + scalar_type(-R[6].y, R[6].x) * w.y;
    R[6]  = R[2] - loc_0;
    R[2]  = R[2] + loc_0;
    loc_0 = R[7] * w.x + scalar_type(-R[7].y, R[7].x) * w.y;
    R[7]  = R[3] - loc_0;
    R[3]  = R[3] + loc_0;
    w.x   = __cosf(0.5f * angle);
    w.y   = __sinf(0.5f * angle);
    loc_0 = R[2] * w.x + scalar_type(-R[2].y, R[2].x) * w.y;
    R[2]  = R[0] - loc_0;
    R[0]  = R[0] + loc_0;
    loc_0 = R[3] * w.x + scalar_type(-R[3].y, R[3].x) * w.y;
    R[3]  = R[1] - loc_0;
    R[1]  = R[1] + loc_0;
    iw.x  = w.y;
    iw.y  = -w.x;
    loc_0 = R[6] * iw.x + scalar_type(-R[6].y, R[6].x) * iw.y;
    R[6]  = R[4] - loc_0;
    R[4]  = R[4] + loc_0;
    loc_0 = R[7] * iw.x + scalar_type(-R[7].y, R[7].x) * iw.y;
    R[7]  = R[5] - loc_0;
    R[5]  = R[5] + loc_0;
    w.x   = __cosf(0.25f * angle);
    w.y   = __sinf(0.25f * angle);
    loc_0 = R[1] * w.x + scalar_type(-R[1].y, R[1].x) * w.y;
    R[1]  = R[0] - loc_0;
    R[0]  = R[0] + loc_0;
    iw.x  = w.y;
    iw.y  = -w.x;
    loc_0 = R[3] * iw.x + scalar_type(-R[3].y, R[3].x) * iw.y;
    R[3]  = R[2] - loc_0;
    R[2]  = R[2] + loc_0;
    iw.x  = w.x * loc_SQRT1_2 + w.y * loc_SQRT1_2;
    iw.y  = w.y * loc_SQRT1_2 - w.x * loc_SQRT1_2;

    loc_0 = R[5] * iw.x + scalar_type(-R[5].y, R[5].x) * iw.y;
    R[5]  = R[4] - loc_0;
    R[4]  = R[4] + loc_0;
    w.x   = iw.y;
    w.y   = -iw.x;
    loc_0 = R[7] * w.x + scalar_type(-R[7].y, R[7].x) * w.y;
    R[7]  = R[6] - loc_0;
    R[6]  = R[6] + loc_0;
    loc_0 = R[1];
    R[1]  = R[4];
    R[4]  = loc_0;
    loc_0 = R[3];
    R[3]  = R[6];
    R[6]  = loc_0;

    __syncthreads();

    int sharedStride     = 17;
    stageInvocationID    = threadIdx_y + 0;
    blockInvocationID    = stageInvocationID;
    stageInvocationID    = stageInvocationID % 1;
    blockInvocationID    = blockInvocationID - stageInvocationID;
    inoutID              = blockInvocationID * 8;
    inoutID              = inoutID + stageInvocationID;
    sdataID              = inoutID + 0;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[0];
    sdataID              = inoutID + 1;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[1];
    sdataID              = inoutID + 2;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[2];
    sdataID              = inoutID + 3;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[3];
    sdataID              = inoutID + 4;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[4];
    sdataID              = inoutID + 5;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[5];
    sdataID              = inoutID + 6;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[6];
    sdataID              = inoutID + 7;
    sdataID              = sharedStride * sdataID;
    sdataID              = sdataID + threadIdx_x;
    lds_complex[sdataID] = R[7];
    __syncthreads();

    stageInvocationID = (threadIdx_y + 0) % (8);
    angle             = stageInvocationID * -3.92699081698724139e-01f;

    R[0] = lds_complex[sharedStride * (threadIdx_y + 0) + threadIdx_x];
    R[1] = lds_complex[sharedStride * (threadIdx_y + 8) + threadIdx_x];
    R[2] = lds_complex[sharedStride * (threadIdx_y + 16) + threadIdx_x];
    R[3] = lds_complex[sharedStride * (threadIdx_y + 24) + threadIdx_x];
    R[4] = lds_complex[sharedStride * (threadIdx_y + 32) + threadIdx_x];
    R[5] = lds_complex[sharedStride * (threadIdx_y + 40) + threadIdx_x];
    R[6] = lds_complex[sharedStride * (threadIdx_y + 48) + threadIdx_x];
    R[7] = lds_complex[sharedStride * (threadIdx_y + 56) + threadIdx_x];

    w.x   = __cosf(angle);
    w.y   = __sinf(angle);
    loc_0 = R[4] * w.x + scalar_type(-R[4].y, R[4].x) * w.y;
    R[4]  = R[0] - loc_0;
    R[0]  = R[0] + loc_0;
    loc_0 = R[5] * w.x + scalar_type(-R[5].y, R[5].x) * w.y;
    R[5]  = R[1] - loc_0;
    R[1]  = R[1] + loc_0;
    loc_0 = R[6] * w.x + scalar_type(-R[6].y, R[6].x) * w.y;
    R[6]  = R[2] - loc_0;
    R[2]  = R[2] + loc_0;
    loc_0 = R[7] * w.x + scalar_type(-R[7].y, R[7].x) * w.y;
    R[7]  = R[3] - loc_0;
    R[3]  = R[3] + loc_0;
    w.x   = __cosf(0.5f * angle);
    w.y   = __sinf(0.5f * angle);
    loc_0 = R[2] * w.x + scalar_type(-R[2].y, R[2].x) * w.y;
    R[2]  = R[0] - loc_0;
    R[0]  = R[0] + loc_0;
    loc_0 = R[3] * w.x + scalar_type(-R[3].y, R[3].x) * w.y;
    R[3]  = R[1] - loc_0;
    R[1]  = R[1] + loc_0;
    iw.x  = w.y;
    iw.y  = -w.x;
    loc_0 = R[6] * iw.x + scalar_type(-R[6].y, R[6].x) * iw.y;
    R[6]  = R[4] - loc_0;
    R[4]  = R[4] + loc_0;
    loc_0 = R[7] * iw.x + scalar_type(-R[7].y, R[7].x) * iw.y;
    R[7]  = R[5] - loc_0;
    R[5]  = R[5] + loc_0;
    w.x   = __cosf(0.25f * angle);
    w.y   = __sinf(0.25f * angle);
    loc_0 = R[1] * w.x + scalar_type(-R[1].y, R[1].x) * w.y;
    R[1]  = R[0] - loc_0;
    R[0]  = R[0] + loc_0;
    iw.x  = w.y;
    iw.y  = -w.x;
    loc_0 = R[3] * iw.x + scalar_type(-R[3].y, R[3].x) * iw.y;
    R[3]  = R[2] - loc_0;
    R[2]  = R[2] + loc_0;
    iw.x  = w.x * loc_SQRT1_2 + w.y * loc_SQRT1_2;
    iw.y  = w.y * loc_SQRT1_2 - w.x * loc_SQRT1_2;

    loc_0 = R[5] * iw.x + scalar_type(-R[5].y, R[5].x) * iw.y;
    R[5]  = R[4] - loc_0;
    R[4]  = R[4] + loc_0;
    w.x   = iw.y;
    w.y   = -iw.x;
    loc_0 = R[7] * w.x + scalar_type(-R[7].y, R[7].x) * w.y;
    R[7]  = R[6] - loc_0;
    R[6]  = R[6] + loc_0;
    loc_0 = R[1];
    R[1]  = R[4];
    R[4]  = loc_0;
    loc_0 = R[3];
    R[3]  = R[6];
    R[6]  = loc_0;
#endif
}
template <typename scalar_type,
          StrideBin     sb,
          EmbeddedType  ebtype,
          CallbackType  cbtype,
          DirectRegType drtype>
__global__ __launch_bounds__(128) void ip_forward_length64_SBRR_new(
    const scalar_type* __restrict__ twiddles,
    const size_t dim,
    const size_t* __restrict__ lengths,
    const size_t* __restrict__ stride,
    const size_t       nbatch,
    const unsigned int lds_padding,
    void* __restrict__ load_cb_fn,
    void* __restrict__ load_cb_data,
    uint32_t load_cb_lds_bytes,
    void* __restrict__ store_cb_fn,
    void* __restrict__ store_cb_data,
    scalar_type* __restrict__ buf)
{
    // this kernel:
    //   uses 8 threads per transform
    //   does 16 transforms per thread block
    // therefore it should be called with 128 threads per thread block
    scalar_type R[8];
    extern __shared__ unsigned char __attribute__((aligned(sizeof(scalar_type)))) lds_uchar[];
    real_type_t<scalar_type>* __restrict__ lds_real
        = reinterpret_cast<real_type_t<scalar_type>*>(lds_uchar);
    scalar_type* __restrict__ lds_complex = reinterpret_cast<scalar_type*>(lds_uchar);
    size_t       offset                   = 0;
    unsigned int offset_lds;
    unsigned int stride_lds;
    size_t       batch;
    size_t       transform;
    const bool   lds_is_real = false;
    const bool   direct_load_to_reg
        = drtype == DirectRegType::TRY_ENABLE_IF_SUPPORT && ebtype == EmbeddedType::NONE;
    const bool direct_store_from_reg = direct_load_to_reg;
    const bool lds_linear            = true;
    auto       load_cb               = get_load_cb<scalar_type, cbtype>(load_cb_fn);
    auto       store_cb              = get_store_cb<scalar_type, cbtype>(store_cb_fn);

    // large twiddles
    // - no large twiddles

    // offsets
    const size_t stride0 = (sb == SB_UNIT) ? (1) : (stride[0]);
    unsigned int thread;
    size_t       remaining;
    size_t       index_along_d;
    transform = blockIdx.x * 16 + threadIdx.x / 8;
    remaining = transform;
    for(int d = 1; d < dim; ++d)
    {
        index_along_d = remaining % lengths[d];
        remaining     = remaining / lengths[d];
        offset        = offset + index_along_d * stride[d];
    }
    batch      = remaining;
    offset     = offset + batch * stride[dim];
    stride_lds = 64 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding);
    offset_lds = stride_lds * (transform % 16);

    if(batch >= nbatch)
    {
        return;
    }

    if(direct_load_to_reg)
    {
        // load global into registers
        thread = threadIdx.x % 8;
        R[0]   = load_cb(buf, offset + (((thread + 0 + 0) + 0)) * stride0, load_cb_data, nullptr);
        R[1]   = load_cb(buf, offset + (((thread + 0 + 0) + 8)) * stride0, load_cb_data, nullptr);
        R[2]   = load_cb(buf, offset + (((thread + 0 + 0) + 16)) * stride0, load_cb_data, nullptr);
        R[3]   = load_cb(buf, offset + (((thread + 0 + 0) + 24)) * stride0, load_cb_data, nullptr);
        R[4]   = load_cb(buf, offset + (((thread + 0 + 0) + 32)) * stride0, load_cb_data, nullptr);
        R[5]   = load_cb(buf, offset + (((thread + 0 + 0) + 40)) * stride0, load_cb_data, nullptr);
        R[6]   = load_cb(buf, offset + (((thread + 0 + 0) + 48)) * stride0, load_cb_data, nullptr);
        R[7]   = load_cb(buf, offset + (((thread + 0 + 0) + 56)) * stride0, load_cb_data, nullptr);
    }

    else
    {
        // load global into lds
        thread = threadIdx.x % 8;
        lds_complex[offset_lds + thread + 0]
            = load_cb(buf, offset + (thread + 0) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + thread + 8]
            = load_cb(buf, offset + (thread + 8) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + thread + 16]
            = load_cb(buf, offset + (thread + 16) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + thread + 24]
            = load_cb(buf, offset + (thread + 24) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + thread + 32]
            = load_cb(buf, offset + (thread + 32) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + thread + 40]
            = load_cb(buf, offset + (thread + 40) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + thread + 48]
            = load_cb(buf, offset + (thread + 48) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + thread + 56]
            = load_cb(buf, offset + (thread + 56) * stride0, load_cb_data, nullptr);
    }

    // calc the thread_in_device value once and for all device funcs
    unsigned int thread_in_device = lds_linear ? threadIdx.x % 8 : threadIdx.x / 16;

    // transform
    forward_length64_SBRR_device<scalar_type,
                                 lds_is_real,
                                 lds_linear ? SB_UNIT : SB_NONUNIT,
                                 lds_linear,
                                 direct_load_to_reg>(
        R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, thread_in_device, true);

    if(direct_store_from_reg)
    {
        // store registers into global
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 0) * stride0,
                 R[0],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 8) * stride0,
                 R[1],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 16) * stride0,
                 R[2],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 24) * stride0,
                 R[3],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 32) * stride0,
                 R[4],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 40) * stride0,
                 R[5],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 48) * stride0,
                 R[6],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 56) * stride0,
                 R[7],
                 store_cb_data,
                 nullptr);
    }

    else
    {
        // store global
        __syncthreads();
        store_cb(buf,
                 offset + (thread + 0) * stride0,
                 lds_complex[offset_lds + thread + 0],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 8) * stride0,
                 lds_complex[offset_lds + thread + 8],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 16) * stride0,
                 lds_complex[offset_lds + thread + 16],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 24) * stride0,
                 lds_complex[offset_lds + thread + 24],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 32) * stride0,
                 lds_complex[offset_lds + thread + 32],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 40) * stride0,
                 lds_complex[offset_lds + thread + 40],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 48) * stride0,
                 lds_complex[offset_lds + thread + 48],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 56) * stride0,
                 lds_complex[offset_lds + thread + 56],
                 store_cb_data,
                 nullptr);

        // append extra global write for Real2C post-process only
        if(ebtype == EmbeddedType::Real2C_POST)
        {
            // use the last thread of each transform to write one more element per row
            if(thread == 7)
            {
                store_cb(buf,
                         offset + (thread + 56 + 1) * stride0,
                         lds_complex[offset_lds + thread + 56 + 1],
                         store_cb_data,
                         nullptr);
            }
        }
    }
}

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

// int vkfft_64()
// {
//     const dim3 grid(1, 1, 64);
//     const dim3 block(64, 8, 1);
//     const int  dy_lds_bytes = 32768;

//     int n = 64 * 64 * 64;

//     int bytes = n * sizeof(float2);

//     float2* h_a = (float2*)malloc(bytes);
//     float2* d_a = (float2*)malloc(bytes);

//     device_malloc(&d_a, bytes);

//     for(int i = 0; i < n; i++)
//     {
//         h_a[i] = float2(i, i);
//     }

//     HIP_ASSERT(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
//     for(int i = 0; i < n; i++)
//     {
//         h_a[i] = float2(0, 0);
//     }

//     fft_64_1024_copy_kernel<<<grid, block, dy_lds_bytes, 0>>>(d_a, d_a);

//     HIP_ASSERT(hipDeviceSynchronize());

//     std::cout << "Execution done.\n";

//     HIP_ASSERT(hipMemcpy(h_a, d_a, bytes, hipMemcpyDeviceToHost));

//     device_free(d_a);

//     free(h_a);

//     return 0;
// }

template <typename T>
__device__ void FwdRad8B1(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + lib_make_vector2<T>(-(*R7).y, (*R7).x);

    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) - C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + lib_make_vector2<T>(-(*R6).y, (*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) - C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename scalar_type, const bool lds_is_real, StrideBin sb>
__device__ void ip_forward_length64_SBRR_device(scalar_type* R,
                                                real_type_t<scalar_type>* __restrict__ lds_real,
                                                scalar_type* __restrict__ lds_complex,
                                                const scalar_type* __restrict__ twiddles,
                                                int          stride_lds,
                                                unsigned int offset_lds)
{
    int         thread;
    scalar_type W;
    scalar_type t;
    const int   lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    // if(unit_stride0)
    //     thread = threadIdx.x % 8;
    // else
    thread = threadIdx.x / 64;

    // pass 0, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    // __syncthreads();

    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);

    if(!lds_is_real)
    {
        //__syncthreads();
        // more than enough threads, some do nothing
        int stageInvocationID = thread + 0;
        int blockInvocationID = stageInvocationID;
        stageInvocationID     = stageInvocationID % 1;
        blockInvocationID     = blockInvocationID - stageInvocationID;
        int inoutID           = blockInvocationID * 8;
        inoutID               = inoutID + stageInvocationID;

        lds_complex[(inoutID + 0) * 64 + threadIdx.x % 64] = R[0];
        lds_complex[(inoutID + 1) * 64 + threadIdx.x % 64] = R[1];
        lds_complex[(inoutID + 2) * 64 + threadIdx.x % 64] = R[2];
        lds_complex[(inoutID + 3) * 64 + threadIdx.x % 64] = R[3];
        lds_complex[(inoutID + 4) * 64 + threadIdx.x % 64] = R[4];
        lds_complex[(inoutID + 5) * 64 + threadIdx.x % 64] = R[5];
        lds_complex[(inoutID + 6) * 64 + threadIdx.x % 64] = R[6];
        lds_complex[(inoutID + 7) * 64 + threadIdx.x % 64] = R[7];

        // if(blockIdx.x == 0 && threadIdx.x < 64)
        // {
        //     printf("lds 1st read: thread %d, offset %d\n",
        //            (int)threadIdx.x,
        //            (int)(offset_lds
        //                  + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride));
        // }
        __syncthreads();
    }

    else
    {
        int stageInvocationID = thread + 0;
        int blockInvocationID = stageInvocationID;
        stageInvocationID     = stageInvocationID % 1;
        blockInvocationID     = blockInvocationID - stageInvocationID;
        int inoutID           = blockInvocationID * 8;
        inoutID               = inoutID + stageInvocationID;

        // more than enough threads, some do nothing
        // if(unit_stride0)
        // {
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride]
        //         = R[0].x;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride]
        //         = R[1].x;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride]
        //         = R[2].x;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride]
        //         = R[3].x;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride]
        //         = R[4].x;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride]
        //         = R[5].x;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride]
        //         = R[6].x;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride]
        //         = R[7].x;
        // }
        // else
        {
            lds_real[(inoutID + 0) * 64 + threadIdx.x % 64] = R[0].x;
            lds_real[(inoutID + 1) * 64 + threadIdx.x % 64] = R[1].x;
            lds_real[(inoutID + 2) * 64 + threadIdx.x % 64] = R[2].x;
            lds_real[(inoutID + 3) * 64 + threadIdx.x % 64] = R[3].x;
            lds_real[(inoutID + 4) * 64 + threadIdx.x % 64] = R[4].x;
            lds_real[(inoutID + 5) * 64 + threadIdx.x % 64] = R[5].x;
            lds_real[(inoutID + 6) * 64 + threadIdx.x % 64] = R[6].x;
            lds_real[(inoutID + 7) * 64 + threadIdx.x % 64] = R[7].x;
        }

        __syncthreads();
        // if(threadIdx.x == 0 && blockIdx.x == 0)
        // {
        //     printf("aaaaaaaaaaaaaaaa\n");
        //     for(int j = 0; j < 64; j++)
        //         printf("%d(%f),", j, float(lds_complex[j].x));
        //     printf("bbbbbbbbbbbbbbbb\n");
        // }

        // more than enough threads, some do nothing
        // if(unit_stride0)
        // {
        //     R[0].x = lds_real[offset_lds + ((thread + 0 + 0) + 0) * lstride];
        //     R[1].x = lds_real[offset_lds + ((thread + 0 + 0) + 8) * lstride];
        //     R[2].x = lds_real[offset_lds + ((thread + 0 + 0) + 16) * lstride];
        //     R[3].x = lds_real[offset_lds + ((thread + 0 + 0) + 24) * lstride];
        //     R[4].x = lds_real[offset_lds + ((thread + 0 + 0) + 32) * lstride];
        //     R[5].x = lds_real[offset_lds + ((thread + 0 + 0) + 40) * lstride];
        //     R[6].x = lds_real[offset_lds + ((thread + 0 + 0) + 48) * lstride];
        //     R[7].x = lds_real[offset_lds + ((thread + 0 + 0) + 56) * lstride];
        // }
        // else
        {
            R[0].x = lds_real[threadIdx.x + 0 * 512];
            R[1].x = lds_real[threadIdx.x + 1 * 512];
            R[2].x = lds_real[threadIdx.x + 2 * 512];
            R[3].x = lds_real[threadIdx.x + 3 * 512];
            R[4].x = lds_real[threadIdx.x + 4 * 512];
            R[5].x = lds_real[threadIdx.x + 5 * 512];
            R[6].x = lds_real[threadIdx.x + 6 * 512];
            R[7].x = lds_real[threadIdx.x + 7 * 512];
        }
        __syncthreads();

        // more than enough threads, some do nothing
        // if(unit_stride0)
        // {
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride]
        //         = R[0].y;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride]
        //         = R[1].y;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride]
        //         = R[2].y;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride]
        //         = R[3].y;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride]
        //         = R[4].y;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride]
        //         = R[5].y;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride]
        //         = R[6].y;
        //     lds_real[offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride]
        //         = R[7].y;
        // }
        // else
        {

            lds_real[(inoutID + 0) * 64 + threadIdx.x % 64] = R[0].y;
            lds_real[(inoutID + 1) * 64 + threadIdx.x % 64] = R[1].y;
            lds_real[(inoutID + 2) * 64 + threadIdx.x % 64] = R[2].y;
            lds_real[(inoutID + 3) * 64 + threadIdx.x % 64] = R[3].y;
            lds_real[(inoutID + 4) * 64 + threadIdx.x % 64] = R[4].y;
            lds_real[(inoutID + 5) * 64 + threadIdx.x % 64] = R[5].y;
            lds_real[(inoutID + 6) * 64 + threadIdx.x % 64] = R[6].y;
            lds_real[(inoutID + 7) * 64 + threadIdx.x % 64] = R[7].y;
        }
        __syncthreads();
        // more than enough threads, some do nothing

        // if(unit_stride0)
        // {
        //     R[0].y = lds_real[offset_lds + ((thread + 0 + 0) + 0) * lstride];
        //     R[1].y = lds_real[offset_lds + ((thread + 0 + 0) + 8) * lstride];
        //     R[2].y = lds_real[offset_lds + ((thread + 0 + 0) + 16) * lstride];
        //     R[3].y = lds_real[offset_lds + ((thread + 0 + 0) + 24) * lstride];
        //     R[4].y = lds_real[offset_lds + ((thread + 0 + 0) + 32) * lstride];
        //     R[5].y = lds_real[offset_lds + ((thread + 0 + 0) + 40) * lstride];
        //     R[6].y = lds_real[offset_lds + ((thread + 0 + 0) + 48) * lstride];
        //     R[7].y = lds_real[offset_lds + ((thread + 0 + 0) + 56) * lstride];
        // }
        // else
        {
            R[0].y = lds_real[threadIdx.x + 0 * 512];
            R[1].y = lds_real[threadIdx.x + 1 * 512];
            R[2].y = lds_real[threadIdx.x + 2 * 512];
            R[3].y = lds_real[threadIdx.x + 3 * 512];
            R[4].y = lds_real[threadIdx.x + 4 * 512];
            R[5].y = lds_real[threadIdx.x + 5 * 512];
            R[6].y = lds_real[threadIdx.x + 6 * 512];
            R[7].y = lds_real[threadIdx.x + 7 * 512];
        }
        __syncthreads();
    }

    // pass 1, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    //__syncthreads();
    if(!lds_is_real)
    {
        // if(blockIdx.x == 0 && threadIdx.x < 64 && threadIdx.y == 0 && blockIdx.z == 0)
        // {
        //     printf("lds 1st write: thread %d, offset %d\n",
        //            (int)threadIdx.x,
        //            (int)(offset_lds + ((thread + 0 + 0) + 0) * lstride));
        // }
        R[0] = lds_complex[threadIdx.x + 0 * 512];
        R[1] = lds_complex[threadIdx.x + 1 * 512];
        R[2] = lds_complex[threadIdx.x + 2 * 512];
        R[3] = lds_complex[threadIdx.x + 3 * 512];
        R[4] = lds_complex[threadIdx.x + 4 * 512];
        R[5] = lds_complex[threadIdx.x + 5 * 512];
        R[6] = lds_complex[threadIdx.x + 6 * 512];
        R[7] = lds_complex[threadIdx.x + 7 * 512];
    }

    W    = twiddles[7 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
    R[1] = t;
    W    = twiddles[8 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
    R[2] = t;
    W    = twiddles[9 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
    R[3] = t;
    W    = twiddles[10 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
    R[4] = t;
    W    = twiddles[11 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[5].x * W.x - R[5].y * W.y, R[5].y * W.x + R[5].x * W.y};
    R[5] = t;
    W    = twiddles[12 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[6].x * W.x - R[6].y * W.y, R[6].y * W.x + R[6].x * W.y};
    R[6] = t;
    W    = twiddles[13 + 7 * ((thread + 0 + 0) % 8)];
    t    = {R[7].x * W.x - R[7].y * W.y, R[7].y * W.x + R[7].x * W.y};
    R[7] = t;

    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
}

template <typename scalar_type, StrideBin sb, EmbeddedType ebtype, CallbackType cbtype>
__global__
    __launch_bounds__(512) void ip_forward_length64_SBRR(const scalar_type* __restrict__ twiddles,
                                                         const int dim,
                                                         const int* __restrict__ lengths,
                                                         const int* __restrict__ stride,
                                                         const int          nbatch,
                                                         const unsigned int lds_padding,
                                                         void* __restrict__ load_cb_fn,
                                                         void* __restrict__ load_cb_data,
                                                         uint32_t load_cb_lds_bytes,
                                                         void* __restrict__ store_cb_fn,
                                                         void* __restrict__ store_cb_data,
                                                         scalar_type* __restrict__ buf)
{
    // this kernel:
    //   uses 8 threads per transform
    //   does 64 transforms per thread block
    // therefore it should be called with 512 threads per thread block
    scalar_type R[8];
    extern __shared__ unsigned char __attribute__((aligned(sizeof(scalar_type)))) lds_uchar[];
    real_type_t<scalar_type>* __restrict__ lds_real
        = reinterpret_cast<real_type_t<scalar_type>*>(lds_uchar);
    scalar_type* __restrict__ lds_complex = reinterpret_cast<scalar_type*>(lds_uchar);
    int          offset                   = 0;
    unsigned int offset_lds;
    //int          stride_lds;
    int        batch;
    int        transform;
    const bool lds_is_real = false; //ebtype == EmbeddedType::NONE;
    const int  stride0     = (sb == SB_UNIT) ? (1) : (stride[0]);
    //auto         load_cb     = get_load_cb<scalar_type, cbtype>(load_cb_fn);
    //auto         store_cb    = get_store_cb<scalar_type, cbtype>(store_cb_fn);

    // large twiddles
    // - no large twiddles

    // offsets
    int thread;
    int remaining;
    int index_along_d;
    transform = blockIdx.x * 64 + threadIdx.x / 8;
    remaining = transform;
    // for(int d = 1; d < dim; ++d)
    // {
    //     index_along_d = remaining % lengths[d];
    //     if(transform == 8)
    //         printf("-------- %d, %d, %d, %d, %d\n", d, remaining, index_along_d, offset, stride[d]);
    //     remaining = remaining / lengths[d];
    //     offset    = offset + index_along_d * stride[d];
    // }
    // if(blockIdx.x == 58 && threadIdx.x == 8)
    // {
    //     printf("offset %d, stride0 %d,dim %d, stride[dim] %d\n",
    //            (int)offset,
    //            (int)stride0,
    //            (int)dim,
    //            (int)stride[dim]);
    // }
    //batch  = remaining;
    //offset = offset + batch * stride[dim];
    //stride_lds = 64 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding);
    offset_lds = 64 * (transform % 64);

    // if(batch >= nbatch)
    // {
    //     return;
    // }

    // if(!lds_is_real)
    // {
    //     // load global into lds
    //     thread = threadIdx.x % 8;
    //     lds_complex[offset_lds + thread + 0]
    //         = load_cb(buf, offset + (thread + 0) * stride0, load_cb_data, nullptr);
    //     lds_complex[offset_lds + thread + 8]
    //         = load_cb(buf, offset + (thread + 8) * stride0, load_cb_data, nullptr);
    //     lds_complex[offset_lds + thread + 16]
    //         = load_cb(buf, offset + (thread + 16) * stride0, load_cb_data, nullptr);
    //     lds_complex[offset_lds + thread + 24]
    //         = load_cb(buf, offset + (thread + 24) * stride0, load_cb_data, nullptr);
    //     lds_complex[offset_lds + thread + 32]
    //         = load_cb(buf, offset + (thread + 32) * stride0, load_cb_data, nullptr);
    //     lds_complex[offset_lds + thread + 40]
    //         = load_cb(buf, offset + (thread + 40) * stride0, load_cb_data, nullptr);
    //     lds_complex[offset_lds + thread + 48]
    //         = load_cb(buf, offset + (thread + 48) * stride0, load_cb_data, nullptr);
    //     lds_complex[offset_lds + thread + 56]
    //         = load_cb(buf, offset + (thread + 56) * stride0, load_cb_data, nullptr);
    // }

    // else
    {
        // if(threadIdx.x == 0 && blockIdx.x == 0)
        //     printf("loading.........stride0 %d, lds_is_real %d\n", (int)stride0, lds_is_real);
        // load global into registers
        thread = threadIdx.x % 8;
        // more than enough threads, some do nothing
        // if(threadIdx.x < 2 && blockIdx.x == 0 && stride0 != 1)
        // {
        //     printf("------test global loading: offset %d, strides: %d %d\n",
        //            (int)offset,
        //            (int)((((thread + 0 + 0) + 0)) * stride0),
        //            (int)((((thread + 0 + 0) + 8)) * stride0));
        // }

        // if((offset + (((thread + 0 + 0) + 56)) * stride0) >= (64 * 64 * 64))
        // {
        //     printf("offset %d, stride0 %d,blockIdx %d, threadIdx %d\n",
        //            (int)offset,
        //            (int)stride0,
        //            (int)blockIdx.x,
        //            (int)threadIdx.x);
        // }

        // if(stride0 == 1)
        // {
        //     R[0] = load_cb(buf, offset + (((thread + 0 + 0) + 0)) * stride0, load_cb_data, nullptr);
        //     R[1] = load_cb(buf, offset + (((thread + 0 + 0) + 8)) * stride0, load_cb_data, nullptr);
        //     R[2]
        //         = load_cb(buf, offset + (((thread + 0 + 0) + 16)) * stride0, load_cb_data, nullptr);
        //     R[3]
        //         = load_cb(buf, offset + (((thread + 0 + 0) + 24)) * stride0, load_cb_data, nullptr);
        //     R[4]
        //         = load_cb(buf, offset + (((thread + 0 + 0) + 32)) * stride0, load_cb_data, nullptr);
        //     R[5]
        //         = load_cb(buf, offset + (((thread + 0 + 0) + 40)) * stride0, load_cb_data, nullptr);
        //     R[6]
        //         = load_cb(buf, offset + (((thread + 0 + 0) + 48)) * stride0, load_cb_data, nullptr);
        //     R[7]
        //         = load_cb(buf, offset + (((thread + 0 + 0) + 56)) * stride0, load_cb_data, nullptr);
        // }
        // else
        {
            offset = blockIdx.x * 64 * 64;
            R[0]   = buf[offset + threadIdx.x + 0 * stride0];
            R[1]   = buf[offset + threadIdx.x + 1 * stride0];
            R[2]   = buf[offset + threadIdx.x + 2 * stride0];
            R[3]   = buf[offset + threadIdx.x + 3 * stride0];
            R[4]   = buf[offset + threadIdx.x + 4 * stride0];
            R[5]   = buf[offset + threadIdx.x + 5 * stride0];
            R[6]   = buf[offset + threadIdx.x + 6 * stride0];
            R[7]   = buf[offset + threadIdx.x + 7 * stride0];
        }
    }

    unsigned int internal_lds_offset;
    // if(stride0 == 1)
    //     internal_lds_offset = offset_lds;
    // else
    internal_lds_offset = (threadIdx.x % 64) * 64;

    //__syncthreads();
    // transform
    ip_forward_length64_SBRR_device<scalar_type, lds_is_real, SB_UNIT>(
        R, lds_real, lds_complex, twiddles, 1, internal_lds_offset);

    //FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);

    // int         stageInvocationID = threadIdx.x / 8;
    // int         LUTId             = stageInvocationID + 0;
    // scalar_type loc_0;
    // scalar_type iw;
    // scalar_type w = twiddles[LUTId];
    // loc_0.x       = R[4].x * w.x - R[4].y * w.y;
    // loc_0.y       = R[4].y * w.x + R[4].x * w.y;
    // R[4].x        = R[0].x - loc_0.x;
    // R[4].y        = R[0].y - loc_0.y;
    // R[0].x        = R[0].x + loc_0.x;
    // R[0].y        = R[0].y + loc_0.y;
    // loc_0.x       = R[5].x * w.x - R[5].y * w.y;
    // loc_0.y       = R[5].y * w.x + R[5].x * w.y;
    // R[5].x        = R[1].x - loc_0.x;
    // R[5].y        = R[1].y - loc_0.y;
    // R[1].x        = R[1].x + loc_0.x;
    // R[1].y        = R[1].y + loc_0.y;
    // loc_0.x       = R[6].x * w.x - R[6].y * w.y;
    // loc_0.y       = R[6].y * w.x + R[6].x * w.y;
    // R[6].x        = R[2].x - loc_0.x;
    // R[6].y        = R[2].y - loc_0.y;
    // R[2].x        = R[2].x + loc_0.x;
    // R[2].y        = R[2].y + loc_0.y;
    // loc_0.x       = R[7].x * w.x - R[7].y * w.y;
    // loc_0.y       = R[7].y * w.x + R[7].x * w.y;
    // R[7].x        = R[3].x - loc_0.x;
    // R[7].y        = R[3].y - loc_0.y;
    // R[3].x        = R[3].x + loc_0.x;
    // R[3].y        = R[3].y + loc_0.y;
    // w             = twiddles[LUTId + 1];

    // loc_0.x = R[2].x * w.x - R[2].y * w.y;
    // loc_0.y = R[2].y * w.x + R[2].x * w.y;
    // R[2].x  = R[0].x - loc_0.x;
    // R[2].y  = R[0].y - loc_0.y;
    // R[0].x  = R[0].x + loc_0.x;
    // R[0].y  = R[0].y + loc_0.y;
    // loc_0.x = R[3].x * w.x - R[3].y * w.y;
    // loc_0.y = R[3].y * w.x + R[3].x * w.y;
    // R[3].x  = R[1].x - loc_0.x;
    // R[3].y  = R[1].y - loc_0.y;
    // R[1].x  = R[1].x + loc_0.x;
    // R[1].y  = R[1].y + loc_0.y;
    // iw.x    = -w.y;
    // iw.y    = w.x;
    // loc_0.x = R[6].x * iw.x - R[6].y * iw.y;
    // loc_0.y = R[6].y * iw.x + R[6].x * iw.y;
    // R[6].x  = R[4].x - loc_0.x;
    // R[6].y  = R[4].y - loc_0.y;
    // R[4].x  = R[4].x + loc_0.x;
    // R[4].y  = R[4].y + loc_0.y;
    // loc_0.x = R[7].x * iw.x - R[7].y * iw.y;
    // loc_0.y = R[7].y * iw.x + R[7].x * iw.y;
    // R[7].x  = R[5].x - loc_0.x;
    // R[7].y  = R[5].y - loc_0.y;
    // R[5].x  = R[5].x + loc_0.x;
    // R[5].y  = R[5].y + loc_0.y;
    // w       = twiddles[LUTId + 2];

    // loc_0.x = R[1].x * w.x - R[1].y * w.y;
    // loc_0.y = R[1].y * w.x + R[1].x * w.y;
    // R[1].x  = R[0].x - loc_0.x;
    // R[1].y  = R[0].y - loc_0.y;
    // R[0].x  = R[0].x + loc_0.x;
    // R[0].y  = R[0].y + loc_0.y;
    // iw.x    = -w.y;
    // iw.y    = w.x;
    // loc_0.x = R[3].x * iw.x - R[3].y * iw.y;
    // loc_0.y = R[3].y * iw.x + R[3].x * iw.y;
    // R[3].x  = R[2].x - loc_0.x;
    // R[3].y  = R[2].y - loc_0.y;
    // R[2].x  = R[2].x + loc_0.x;
    // R[2].y  = R[2].y + loc_0.y;
    // iw.x    = w.x * loc_SQRT1_2 - w.y * loc_SQRT1_2;
    // iw.y    = w.y * loc_SQRT1_2 + w.x * loc_SQRT1_2;

    // loc_0.x = R[5].x * iw.x - R[5].y * iw.y;
    // loc_0.y = R[5].y * iw.x + R[5].x * iw.y;
    // R[5].x  = R[4].x - loc_0.x;
    // R[5].y  = R[4].y - loc_0.y;
    // R[4].x  = R[4].x + loc_0.x;
    // R[4].y  = R[4].y + loc_0.y;
    // w.x     = -iw.y;
    // w.y     = iw.x;
    // loc_0.x = R[7].x * w.x - R[7].y * w.y;
    // loc_0.y = R[7].y * w.x + R[7].x * w.y;
    // R[7].x  = R[6].x - loc_0.x;
    // R[7].y  = R[6].y - loc_0.y;
    // R[6].x  = R[6].x + loc_0.x;
    // R[6].y  = R[6].y + loc_0.y;
    // loc_0   = R[1];
    // R[1]    = R[4];
    // R[4]    = loc_0;
    // loc_0   = R[3];
    // R[3]    = R[6];
    // R[6]    = loc_0;

    // if(!lds_is_real)
    // {
    //     // store global
    //     __syncthreads();
    //     store_cb(buf,
    //              offset + (thread + 0) * stride0,
    //              lds_complex[offset_lds + thread + 0],
    //              store_cb_data,
    //              nullptr);
    //     store_cb(buf,
    //              offset + (thread + 8) * stride0,
    //              lds_complex[offset_lds + thread + 8],
    //              store_cb_data,
    //              nullptr);
    //     store_cb(buf,
    //              offset + (thread + 16) * stride0,
    //              lds_complex[offset_lds + thread + 16],
    //              store_cb_data,
    //              nullptr);
    //     store_cb(buf,
    //              offset + (thread + 24) * stride0,
    //              lds_complex[offset_lds + thread + 24],
    //              store_cb_data,
    //              nullptr);
    //     store_cb(buf,
    //              offset + (thread + 32) * stride0,
    //              lds_complex[offset_lds + thread + 32],
    //              store_cb_data,
    //              nullptr);
    //     store_cb(buf,
    //              offset + (thread + 40) * stride0,
    //              lds_complex[offset_lds + thread + 40],
    //              store_cb_data,
    //              nullptr);
    //     store_cb(buf,
    //              offset + (thread + 48) * stride0,
    //              lds_complex[offset_lds + thread + 48],
    //              store_cb_data,
    //              nullptr);
    //     store_cb(buf,
    //              offset + (thread + 56) * stride0,
    //              lds_complex[offset_lds + thread + 56],
    //              store_cb_data,
    //              nullptr);
    // }

    // else
    {
        // store registers into global
        // more than enough threads, some do nothing
        // if(stride0 == 1)
        // {
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 0) * stride0,
        //              R[0],
        //              store_cb_data,
        //              nullptr);
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 8) * stride0,
        //              R[1],
        //              store_cb_data,
        //              nullptr);
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 16) * stride0,
        //              R[2],
        //              store_cb_data,
        //              nullptr);
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 24) * stride0,
        //              R[3],
        //              store_cb_data,
        //              nullptr);
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 32) * stride0,
        //              R[4],
        //              store_cb_data,
        //              nullptr);
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 40) * stride0,
        //              R[5],
        //              store_cb_data,
        //              nullptr);
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 48) * stride0,
        //              R[6],
        //              store_cb_data,
        //              nullptr);
        //     store_cb(buf,
        //              offset + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 56) * stride0,
        //              R[7],
        //              store_cb_data,
        //              nullptr);
        // }
        // else
        {
            offset = blockIdx.x * 64 * 64;

            // Debug index only
            // R[0].x = offset + threadIdx.x + 0 * stride0;
            // R[0].y = 0;
            // R[1].x = offset + threadIdx.x + 1 * stride0;
            // R[1].y = 0;
            // R[2].x = offset + threadIdx.x + 2 * stride0;
            // R[2].y = 0;
            // R[3].x = offset + threadIdx.x + 3 * stride0;
            // R[3].y = 0;
            // R[4].x = offset + threadIdx.x + 4 * stride0;
            // R[4].y = 0;
            // R[5].x = offset + threadIdx.x + 5 * stride0;
            // R[5].y = 0;
            // R[6].x = offset + threadIdx.x + 6 * stride0;
            // R[6].y = 0;
            // R[7].x = offset + threadIdx.x + 7 * stride0;
            // R[7].y = 0;
            buf[offset + threadIdx.x + 0 * stride0] = R[0];
            buf[offset + threadIdx.x + 1 * stride0] = R[1];
            buf[offset + threadIdx.x + 2 * stride0] = R[2];
            buf[offset + threadIdx.x + 3 * stride0] = R[3];
            buf[offset + threadIdx.x + 4 * stride0] = R[4];
            buf[offset + threadIdx.x + 5 * stride0] = R[5];
            buf[offset + threadIdx.x + 6 * stride0] = R[6];
            buf[offset + threadIdx.x + 7 * stride0] = R[7];
        }
    }
}

bool check_accuracy(const size_t         N,
                    const size_t         nbatch,
                    const fftwf_complex* ref_out,
                    float2*              new_out,
                    bool                 verbose)
{
    double max_linf_eps_single = 0.0;
    double max_l2_eps_single   = 0.0;

    VectorNorms cpu_output_norm = norm_complex<std::complex<float>, size_t, size_t>(
        reinterpret_cast<const std::complex<float>*>(ref_out), N, nbatch, 1, 64, {0});

    VectorNorms gpu_output_norm = norm_complex<std::complex<float>, size_t, size_t>(
        reinterpret_cast<const std::complex<float>*>(new_out), N, nbatch, 1, 64, {0});

    std::vector<std::pair<size_t, size_t>> linf_failures;
    const auto                             total_length = 64;
    const double linf_cutoff = single_epsilon * cpu_output_norm.l_inf * log(total_length);

    VectorNorms diff = distance_1to1_complex<std::complex<float>, size_t, size_t, size_t>(
        reinterpret_cast<const std::complex<float>*>(ref_out),
        reinterpret_cast<const std::complex<float>*>(new_out),
        N,
        nbatch,
        1,
        64,
        1,
        64,
        linf_failures,
        linf_cutoff,
        {0},
        {0});

    if(verbose)
    {
        std::cout << "CPU Output Linf norm: " << std::scientific << cpu_output_norm.l_inf << "\n";
        std::cout << "CPU Output L2 norm:   " << std::scientific << cpu_output_norm.l_2 << "\n";
        std::cout << "GPU output Linf norm: " << std::scientific << gpu_output_norm.l_inf << "\n";
        std::cout << "GPU output L2 norm:   " << std::scientific << gpu_output_norm.l_2 << "\n";
        std::cout << "GPU linf norm failures:";
        std::sort(linf_failures.begin(), linf_failures.end());
        for(const auto& i : linf_failures)
        {
            std::cout << " (" << i.first << "," << i.second << ")";
        }
        std::cout << std::endl;
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    if(diff.l_inf > linf_cutoff)
    {
        std::cout << "Linf test failed.  Linf:" << diff.l_inf
                  << "\tnormalized Linf: " << diff.l_inf / cpu_output_norm.l_inf
                  << "\tcutoff: " << linf_cutoff;
        return false;
    }

    if(diff.l_2 / cpu_output_norm.l_2 >= sqrt(log2(total_length)) * single_epsilon)
    {
        std::cout << "L2 test failed. L2: " << diff.l_2
                  << "\tnormalized L2: " << diff.l_2 / cpu_output_norm.l_2
                  << "\tepsilon: " << sqrt(log2(total_length)) * single_epsilon;
        return false;
    }

    max_linf_eps_single
        = std::max(max_linf_eps_single, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
    max_l2_eps_single
        = std::max(max_l2_eps_single, diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));

    std::cout << "single precision max l-inf epsilon: " << std::scientific << max_linf_eps_single
              << std::endl;
    std::cout << "single precision max l2 epsilon: " << std::scientific << max_l2_eps_single
              << std::endl;
    return true;
}

template <typename scalar_type>
int fft_64_1024(int trial, bool isOld)
{
    double max_linf_eps_single = 0.0;
    double max_l2_eps_single   = 0.0;

    device_reset();

    int       N      = 64;
    const int nbatch = 1024;

    int n_bytes = N * nbatch * sizeof(scalar_type);

    scalar_type* h_a   = (scalar_type*)malloc(n_bytes);
    scalar_type* d_a   = (scalar_type*)malloc(n_bytes);
    scalar_type* h_twd = (scalar_type*)malloc(N * sizeof(scalar_type));

    std::vector<size_t>      radices;
    std::vector<scalar_type> twd;
    radices.push_back(8);
    radices.push_back(8);
    twd = GenerateTwiddleTable<scalar_type>(radices, N);
    //std::cout << "twd size " << twd.size() << std::endl;

    scalar_type* d_twd;
    const size_t dim = 1;
    size_t*      d_lengths;
    size_t*      d_strides;

    const unsigned int lds_padding   = 0;
    void* __restrict__ load_cb_fn    = nullptr;
    void* __restrict__ load_cb_data  = nullptr;
    uint32_t load_cb_lds_bytes       = 0;
    void* __restrict__ store_cb_fn   = nullptr;
    void* __restrict__ store_cb_data = nullptr;

    device_malloc((void**)&d_twd, 64 * sizeof(scalar_type));
    device_malloc((void**)&d_lengths, 4 * sizeof(size_t));
    device_malloc((void**)&d_strides, 4 * sizeof(size_t));
    device_malloc((void**)&d_a, n_bytes);

    for(int i = 0; i < N * nbatch; i++)
    {
        h_a[i].x = h_a[i].y = i + 1;
    }

    //std::cout << "twd\n";
    for(int i = 0; i < 56; i++)
    {
        h_twd[i] = twd[i];
        //std::cout << "i " << i << ": " << h_twd[i].x << ", " << h_twd[i].y << std::endl;
        //std::cout << h_twd[i].x << ", " << h_twd[i].y << std::endl;
    }

    size_t h_lengths[4];
    h_lengths[0] = 64;
    h_lengths[1] = h_lengths[2] = 0;
    h_lengths[3]                = 0;
    size_t h_strides[4];
    h_strides[0] = 1;
    h_strides[1] = 64;
    h_strides[2] = 0;
    h_strides[3] = 0;

    device_memcpy_h2d(d_a, h_a, n_bytes);
    device_memcpy_h2d(d_twd, h_twd, N * sizeof(scalar_type));
    device_memcpy_h2d(d_lengths, h_lengths, 4 * sizeof(size_t));
    device_memcpy_h2d(d_strides, h_strides, 4 * sizeof(size_t));

    // for(int i = 0; i < n; i++)
    // {
    //     h_a[i].x = h_a[i].y = 0;
    // }

    fftwf_complex *ref_in, *ref_out;

    ref_in  = new fftwf_complex[N * nbatch];
    ref_out = new fftwf_complex[N * nbatch];
    std::memcpy(ref_in, h_a, n_bytes);

    fftwf_plan p = fftwf_plan_many_dft(
        1, &N, nbatch, ref_in, NULL, 1, 64, ref_out, NULL, 1, 64, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    // for(int i = 0; i < 64; i++)
    // {
    //     std::cout << "(" << ref_out[i][0] << ", " << ref_out[i][1] << ")";
    // }
    // std::cout << std::endl;

    device_event_create();

    if(isOld)
    {
        const dim3 grid(1, 64, 1);
        const dim3 block(16, 8, 1);
        const int  dy_lds_bytes = (64 * 16 + 64) * 8;

        // warm up
        VkFFT_main<<<grid, block, dy_lds_bytes, 0>>>(d_a, d_a);

        device_memcpy_d2h(h_a, d_a, n_bytes);

        if(!check_accuracy(N, nbatch, ref_out, h_a, false))
            return -1;

        // for(int i = 0; i < 128; i++)
        // {
        //     std::cout << "(" << h_a[i].x << ", " << h_a[i].y << ")";
        // }
        // std::cout << std::endl;

        //fft_64_1024_copy_kernel<<<grid, block, dy_lds_bytes, 0>>>(d_a, d_a);
        // Debug only, verify pure copy quickly
        // device_memcpy_d2h(h_a, d_a, n_bytes);
        // for(int i = 0; i < n; i++)
        // {
        //     if((int)h_a[i].x != (i + 1) || (int)h_a[i].y != (i + 1))
        //     {
        //         std::cout << "pure copy: failed at " << i << ", " << h_a[i].x << std::endl;
        //         exit(-1);
        //     }
        // }
        // std::cout << "verify pure copy done.\n";

        device_event_record_start();
        for(int itrial = 0; itrial < trial; ++itrial)
        {
            VkFFT_main<<<grid, block, dy_lds_bytes, 0>>>(d_a, d_a);
            //fft_64_1024_copy_kernel<<<grid, block, dy_lds_bytes, 0>>>(d_a, d_a);
        }
        device_event_record_stop();
        device_event_synchronize_stop();
        device_synchronize();
        std::cout << "Total elapsed time:" << std::fixed << std::setw(8) << std::setprecision(3)
                  << device_event_elapsed_time() << " ms\n";
    }
    else
    {
        const dim3 grid(64, 1, 1);
        const dim3 block(128, 1, 1);
        const int  dy_lds_bytes = 64 * 16 * 8 * 2;

        // warm up
        ip_forward_length64_SBRR_new<scalar_type,
                                     SB_NONUNIT,
                                     EmbeddedType::NONE,
                                     CallbackType::NONE,
                                     DirectRegType::TRY_ENABLE_IF_SUPPORT>
            <<<grid, block, dy_lds_bytes, 0>>>(d_twd,
                                               dim,
                                               d_lengths,
                                               d_strides,
                                               nbatch,
                                               lds_padding,
                                               load_cb_fn,
                                               load_cb_data,
                                               load_cb_lds_bytes,
                                               store_cb_fn,
                                               store_cb_data,
                                               d_a);
        //device_memcpy_d2h(h_a, d_a, n_bytes);
        // for(int i = 0; i < n; i++)
        // {
        //     if((int)h_a[i].x != (i + 1) || (int)h_a[i].y != (i + 1))
        //     {
        //         std::cout << "pure copy: failed at " << i << ", " << h_a[i].x << std::endl;
        //         exit(-1);
        //     }
        // }
        // std::cout << "verify pure copy done.\n";

        device_memcpy_d2h(h_a, d_a, n_bytes);
        // for(int i = 0; i < 128; i++)
        // {
        //     std::cout << "(" << h_a[i].x << ", " << h_a[i].y << ")";
        // }
        // std::cout << std::endl;
        if(!check_accuracy(N, nbatch, ref_out, h_a, false))
            return -1;

        device_event_record_start();
        for(int itrial = 0; itrial < trial; ++itrial)
        {
            ip_forward_length64_SBRR_new<scalar_type,
                                         SB_NONUNIT,
                                         EmbeddedType::NONE,
                                         CallbackType::NONE,
                                         DirectRegType::TRY_ENABLE_IF_SUPPORT>
                <<<grid, block, dy_lds_bytes, 0>>>(d_twd,
                                                   dim,
                                                   d_lengths,
                                                   d_strides,
                                                   nbatch,
                                                   lds_padding,
                                                   load_cb_fn,
                                                   load_cb_data,
                                                   load_cb_lds_bytes,
                                                   store_cb_fn,
                                                   store_cb_data,
                                                   d_a);
        }
        device_event_record_stop();
        device_event_synchronize_stop();
        device_synchronize();
        std::cout << "Total elapsed time:" << std::fixed << std::setw(8) << std::setprecision(3)
                  << device_event_elapsed_time() << " ms\n";
    }

    device_synchronize();

    std::cout << "Execution done.\n";

    device_memcpy_d2h(h_a, d_a, n_bytes);

    device_event_destroy();

    device_free(d_a);
    device_free(d_lengths);
    device_free(d_strides);
    device_free(d_twd);
    free(h_twd);
    free(h_a);

    fftwf_destroy_plan(p);
    delete[] ref_in;
    delete[] ref_out;

    return 0;
}

int main(int argc, char* argv[])
{
    if(argv[1][0] == '0')
    {
        std::cout << "vkFFT...\n";
        fft_64_1024<float2>(1000, 1);
    }
    else if(argv[1][0] == '1')
    {
        std::cout << "rocFFT...\n";
        fft_64_1024<float2>(1000, 0);
    }
    else if(argv[1][0] == '2')
    {
        std::cout << "vkFFT...\n";
        fft_64_1024<float2>(1000, 1);
        std::cout << "rocFFT...\n";
        fft_64_1024<float2>(1000, 0);
    }

    return 0;
}

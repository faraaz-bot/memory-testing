
#include "butterfly_constant.h"
#include "common.h"
#include "runtime_api_wrapper.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

__device__ int get_1d_global_idx()
{

    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x)
                   + threadIdx.x;

    return threadId;
}

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

extern __shared__ float shared[];
extern "C" __launch_bounds__(512) __global__
    void VkFFT_main(float2* inputs, float2* outputs, float2* twiddleLUT)
{
    unsigned int sharedStride = 64;
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
    unsigned int LUTId             = 0;
    if((((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
           + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64)
       < 64)
    {
        inoutID = (1 * (threadIdx.y + 0) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        // if(get_1d_global_idx() == 0 || get_1d_global_idx() == 64)
        // {
        //     printf("glb2reg: thread %d, 1st glb_offset %d\n",
        //            (int)get_1d_global_idx(),
        //            (int)(inoutID));
        // }
        temp_0  = inputs[inoutID];
        inoutID = (1 * (threadIdx.y + 8) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        temp_1  = inputs[inoutID];
        inoutID = (1 * (threadIdx.y + 16) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        temp_2  = inputs[inoutID];
        inoutID = (1 * (threadIdx.y + 24) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        temp_3  = inputs[inoutID];
        inoutID = (1 * (threadIdx.y + 32) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        temp_4  = inputs[inoutID];
        inoutID = (1 * (threadIdx.y + 40) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        temp_5  = inputs[inoutID];
        inoutID = (1 * (threadIdx.y + 48) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        temp_6  = inputs[inoutID];
        inoutID = (1 * (threadIdx.y + 56) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                   + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64));
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64)) + (inoutID)*64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        temp_7 = inputs[inoutID];
    }

    //FwdRad8B1(&temp_0, &temp_1, &temp_2, &temp_3, &temp_4, &temp_5, &temp_6, &temp_7);
    __syncthreads();

    if((((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
           + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64)
       < 64)
    {
        inoutID = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 0) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        // if(get_1d_global_idx() == 0 || get_1d_global_idx() == 64)
        // {
        //     printf("reg2glb: thread %d, 1st glb_offset %d\n",
        //            (int)get_1d_global_idx(),
        //            (int)(inoutID));
        // }
        outputs[inoutID] = temp_0;
        inoutID          = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 8) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        outputs[inoutID] = temp_1;
        inoutID          = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 16) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        outputs[inoutID] = temp_2;
        inoutID          = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 24) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        outputs[inoutID] = temp_3;
        inoutID          = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 32) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        outputs[inoutID] = temp_4;
        inoutID          = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 40) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        outputs[inoutID] = temp_5;
        inoutID          = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 48) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        outputs[inoutID] = temp_6;
        inoutID          = (((threadIdx.x + blockIdx.x * blockDim.x)) % (64))
                  + (1 * (threadIdx.y + 56) + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) % (1)
                     + (((threadIdx.x + blockIdx.x * blockDim.x)) / 64) * (64))
                        * 64
                  + (threadIdx.z + blockIdx.z * blockDim.z) * 4096;
        outputs[inoutID] = temp_7;
    }
}

template <typename scalar_type>
__global__
    __launch_bounds__(512) void ip_forward_length64_SBRR(const scalar_type* __restrict__ twiddles,
                                                         scalar_type* __restrict__ buf,
                                                         scalar_type* __restrict__ buf_out)
{
    // this kernel:
    //   uses 8 threads per transform
    //   does 64 transforms per thread block
    // therefore it should be called with 512 threads per thread block
    scalar_type R[8];
    // extern __shared__ unsigned char __attribute__((aligned(sizeof(scalar_type)))) lds_uchar[];
    // real_type_t<scalar_type>* __restrict__ lds_real
    //     = reinterpret_cast<real_type_t<scalar_type>*>(lds_uchar);
    // scalar_type* __restrict__ lds_complex = reinterpret_cast<scalar_type*>(lds_uchar);
    int offset = blockIdx.x * 64 * 64;

    const int stride0 = 512; //(sb == SB_UNIT) ? (1) : (stride[0]);

    {
        R[0] = buf[offset + threadIdx.x + 0 * stride0];
        R[1] = buf[offset + threadIdx.x + 1 * stride0];
        R[2] = buf[offset + threadIdx.x + 2 * stride0];
        R[3] = buf[offset + threadIdx.x + 3 * stride0];
        R[4] = buf[offset + threadIdx.x + 4 * stride0];
        R[5] = buf[offset + threadIdx.x + 5 * stride0];
        R[6] = buf[offset + threadIdx.x + 6 * stride0];
        R[7] = buf[offset + threadIdx.x + 7 * stride0];
    }

    //FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
    __syncthreads();

    {
        buf_out[offset + threadIdx.x + 0 * stride0] = R[0];
        buf_out[offset + threadIdx.x + 1 * stride0] = R[1];
        buf_out[offset + threadIdx.x + 2 * stride0] = R[2];
        buf_out[offset + threadIdx.x + 3 * stride0] = R[3];
        buf_out[offset + threadIdx.x + 4 * stride0] = R[4];
        buf_out[offset + threadIdx.x + 5 * stride0] = R[5];
        buf_out[offset + threadIdx.x + 6 * stride0] = R[6];
        buf_out[offset + threadIdx.x + 7 * stride0] = R[7];

        // for(int i = 0; i < 8; i++)
        //     buf_out[blockIdx.x * 64 * 64 + threadIdx.x + i * stride0] = R[i];
    }
}

template <typename scalar_type>
int fft_64_2nd(int trial, bool isRef)
{
    device_reset();

    int n = 64 * 64 * 64;

    int n_bytes = n * sizeof(scalar_type);

    scalar_type* h_a   = (scalar_type*)malloc(n_bytes);
    scalar_type* d_a   = (scalar_type*)malloc(n_bytes);
    scalar_type* d_b   = (scalar_type*)malloc(n_bytes);
    scalar_type* h_twd = (scalar_type*)malloc(64 * sizeof(scalar_type));

    scalar_type*       d_twd;
    const int          dim = 3;
    int*               d_lengths;
    int*               d_strides;
    const int          nbatch      = 1;
    const unsigned int lds_padding = 0;

    device_malloc((void**)&d_twd, 64 * sizeof(scalar_type));
    device_malloc((void**)&d_lengths, 4 * sizeof(int));
    device_malloc((void**)&d_strides, 4 * sizeof(int));
    device_malloc((void**)&d_a, n_bytes);
    device_malloc((void**)&d_b, n_bytes);

    for(int i = 0; i < n; i++)
    {
        h_a[i].x = h_a[i].y = i + 1;
    }

    for(int i = 0; i < 64; i++)
    {
        h_twd[i].x = h_twd[i].y = i * 2;
    }

    int h_lengths[4];
    h_lengths[0] = h_lengths[1] = h_lengths[2] = 64;
    h_lengths[3]                               = 1;
    int h_strides[4];
    h_strides[0] = 512;
    h_strides[1] = 1;
    h_strides[2] = 64 * 64 * 64;
    h_strides[3] = 64 * 64 * 64;

    device_memcpy_h2d(d_a, h_a, n_bytes);
    device_memcpy_h2d(d_twd, h_twd, 64 * sizeof(scalar_type));
    device_memcpy_h2d(d_lengths, h_lengths, 4 * sizeof(int));
    device_memcpy_h2d(d_strides, h_strides, 4 * sizeof(int));
    device_mem_set(d_b, 0, n_bytes);

    for(int i = 0; i < n; i++)
    {
        h_a[i].x = h_a[i].y = 0;
    }

    device_event_create();

    if(isRef)
    {
        const dim3 grid(1, 1, 64);
        const dim3 block(64, 8, 1);
        const int  dy_lds_bytes = 32768;

        // warm up
        VkFFT_main<<<grid, block, dy_lds_bytes, 0>>>(d_a, d_b, d_twd);

        // Debug only, verify pure copy quickly
        // device_memcpy_d2h(h_a, d_b, n_bytes);
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
            VkFFT_main<<<grid, block, dy_lds_bytes, 0>>>(d_a, d_b, d_twd);
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
        const dim3 block(512, 1, 1);
        const int  dy_lds_bytes = 32768; //16384;

        // warm up
        ip_forward_length64_SBRR<scalar_type><<<grid, block, dy_lds_bytes, 0>>>(d_twd, d_a, d_b);

        // device_memcpy_d2h(h_a, d_b, n_bytes);
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
            ip_forward_length64_SBRR<scalar_type>
                <<<grid, block, dy_lds_bytes, 0>>>(d_twd, d_a, d_b);
        }
        device_event_record_stop();
        device_event_synchronize_stop();
        device_synchronize();
        std::cout << "Total elapsed time:" << std::fixed << std::setw(8) << std::setprecision(3)
                  << device_event_elapsed_time() << " ms\n";
    }

    device_synchronize();

    std::cout << "Execution done.\n";

    device_memcpy_d2h(h_a, d_b, n_bytes);

    device_event_destroy();

    device_free(d_a);
    device_free(d_b);
    device_free(d_lengths);
    device_free(d_strides);
    device_free(d_twd);
    free(h_twd);
    free(h_a);

    return 0;
}

int main(int argc, char* argv[])
{
    std::cout << "vkFFT...\n";
    fft_64_2nd<float2>(20, 1);
    std::cout << "rocFFT...\n";
    fft_64_2nd<float2>(20, 0);

    return 0;
}

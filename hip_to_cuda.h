#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define hipError_t                    cudaError_t
#define hipSuccess                    cudaSuccess
#define hipStream_t                   cudaStream_t
#define hipEvent_t                    cudaEvent_t

#define hipMalloc                     cudaMalloc
#define hipFree                       cudaFree
#define hipMemset                     cudaMemset

#define hipMemcpy                     cudaMemcpy
#define hipMemcpyAsync                cudaMemcpyAsync
#define hipMemcpy2D                   cudaMemcpy2D
#define hipMemcpy2DAsync              cudaMemcpy2DAsync
#define hipMemcpyHostToDevice         cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost         cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice       cudaMemcpyDeviceToDevice

#define hipStreamCreate               cudaStreamCreate
#define hipStreamDestroy              cudaStreamDestroy
#define hipStreamSynchronize          cudaStreamSynchronize

#define hipEventCreate                cudaEventCreate
#define hipEventRecord                cudaEventRecord
#define hipEventElapsedTime           cudaEventElapsedTime
#define hipEventSynchronize           cudaEventSynchronize
#define hipEventDestroy               cudaEventDestroy

#define hipSetDevice                  cudaSetDevice
#define hipGetDevice                  cudaGetDevice

#define hipDeviceSynchronize          cudaDeviceSynchronize
#define hipDeviceCanAccessPeer        cudaDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess     cudaDeviceEnablePeerAccess

#define hipGetErrorString             cudaGetErrorString

/*───────────────── kernel-launch helper for hipLaunchKernelGGL ─────────────*/
template <typename Kernel, typename... Args>
static inline cudaError_t hipLaunchKernelGGL_helper(
        Kernel k, dim3 grid, dim3 block, std::size_t shmem,
        cudaStream_t stream, Args... args)
{
    k<<<grid, block, shmem, stream>>>(args...);
    return cudaGetLastError();
}
#define hipLaunchKernelGGL(kernel, grids, blocks, shmem, stream, ...) \
    hipLaunchKernelGGL_helper(kernel, grids, blocks, shmem, stream, __VA_ARGS__)

/*──────────────────────────── error guard ────────────────────────────────*/
#define HIP_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t _e = (cmd);                                              \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr,                                                 \
                    "CUDA error %s at %s:%d\n",                             \
                    cudaGetErrorString(_e), __FILE__, __LINE__);            \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

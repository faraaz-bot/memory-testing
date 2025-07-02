#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/* ── basic type / enum aliases ─────────────────────────────────── */
#define hipError_t                    cudaError_t
#define hipSuccess                    cudaSuccess
#define hipStream_t                   cudaStream_t
#define hipEvent_t                    cudaEvent_t

/* ── memory management ─────────────────────────────────────────── */
#define hipMalloc                     cudaMalloc
#define hipFree                       cudaFree
#define hipMemcpy                     cudaMemcpy
#define hipMemcpyAsync                cudaMemcpyAsync
#define hipMemset                     cudaMemset
#define hipMemcpyHostToDevice         cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost         cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice       cudaMemcpyDeviceToDevice

/* ── streams & events ──────────────────────────────────────────── */
#define hipStreamCreate               cudaStreamCreate
#define hipStreamSynchronize          cudaStreamSynchronize
#define hipEventCreate                cudaEventCreate
#define hipEventRecord                cudaEventRecord
#define hipEventElapsedTime           cudaEventElapsedTime
#define hipEventSynchronize           cudaEventSynchronize
#define hipEventDestroy               cudaEventDestroy

/* ── device control ────────────────────────────────────────────── */
#define hipSetDevice                  cudaSetDevice
#define hipGetDevice                  cudaGetDevice
#define hipDeviceCanAccessPeer        cudaDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess     cudaDeviceEnablePeerAccess

/* ── kernel launch helper ────────────────────────────────────────
   hipLaunchKernelGGL(lambda, grids, blocks, shm, stream, args...)
   → just forwards to CUDA’s <<<>>> syntax via a tiny wrapper        */
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

/* ── simple CHECK macro ────────────────────────────────────────── */
#define HIP_CHECK(cmd)                                             \
  do {                                                             \
    hipError_t _e = (cmd);                                         \
    if (_e != hipSuccess) {                                        \
      fprintf(stderr,"CUDA error %s at %s:%d\n",                   \
              cudaGetErrorString(_e), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                     \
    }                                                              \
  } while (0)

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define hipError_t                    cudaError_t
#define hipSuccess                    cudaSuccess
#define hipMemcpy                     cudaMemcpy
#define hipMemcpyAsync                cudaMemcpyAsync
#define hipMemcpy2D                   cudaMemcpy2D
#define hipMemcpy2DAsync             cudaMemcpy2DAsync
#define hipMemcpyHostToDevice         cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost         cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice       cudaMemcpyDeviceToDevice
#define hipSetDevice                  cudaSetDevice
#define hipGetDevice                  cudaGetDevice
#define hipDeviceCanAccessPeer        cudaDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess     cudaDeviceEnablePeerAccess
#define hipStream_t                   cudaStream_t
#define hipStreamCreate               cudaStreamCreate
#define hipStreamSynchronize          cudaStreamSynchronize
#define hipEvent_t                    cudaEvent_t
#define hipEventCreate                cudaEventCreate
#define hipEventRecord                cudaEventRecord
#define hipEventSynchronize           cudaEventSynchronize
#define hipGetErrorString             cudaGetErrorString
#define hipDeviceSynchronize          cudaDeviceSynchronize
#define hipEventDestroy               cuEventDestroy
#define hipEventElapsedTime           cuEventElapsedTime
#define hipStreamDestroy              cudaStreamDestroy


/* simple CHECK macro—keeps all existing error handling intact */
#define HIP_CHECK(expr)                                           \
  do {                                                            \
    hipError_t _err = (expr);                                     \
    if (_err != hipSuccess) {                                     \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
              hipGetErrorString(_err), __FILE__, __LINE__);       \
      std::exit(EXIT_FAILURE);                                    \
    }                                                             \
  } while (0)

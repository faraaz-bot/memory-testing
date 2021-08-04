//
// The helper functions to build on AMD and NV platforms
//

#ifdef CUDA
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#else
// clang-format off
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
// clang-format on
#endif

#include <iostream>

using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;

//-----------------------------------------------------------------------------
// Error check

#define GPU_ERR_CHECK(expr)                     \
    {                                           \
        gpu_assert((expr), __FILE__, __LINE__); \
    }

#ifdef CUDA
inline void gpu_assert(cudaError_t e, const char* file, int line, bool abort = true)
{
    if(e != cudaSuccess)
    {
        const char* errName = cudaGetErrorName(e);
        const char* errMsg  = cudaGetErrorString(e);
        std::cerr << "Error " << e << "(" << errName << ") " << errMsg << std::endl;
        exit(e);
    }
}
#else
inline void gpu_assert(hipError_t e, const char* file, int line, bool abort = true)
{
    if(e)
    {
        const char* errName = hipGetErrorName(e);
        const char* errMsg  = hipGetErrorString(e);
        std::cerr << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": "
                  << std::endl
                  << errMsg << std::endl;
        exit(e);
    }
}
#endif

//-----------------------------------------------------------------------------
// API wrappers

inline void device_reset()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaDeviceReset());
#else
    GPU_ERR_CHECK(hipDeviceReset());
#endif
}

inline void device_synchronize()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaDeviceSynchronize());
#else
    GPU_ERR_CHECK(hipDeviceSynchronize());
#endif
}

inline void device_malloc(void** ptr, size_t bytes)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaMalloc(ptr, bytes));
#else
    GPU_ERR_CHECK(hipMalloc(ptr, bytes));
#endif
}

inline void device_free(void* ptr)
{
#ifdef CUDA
    cudaFree(ptr);
#else
    hipFree(ptr);
#endif
}

inline void device_memcpy_h2d(void* dst, const void* src, size_t bytes)
{
#ifdef CUDA
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
#else
    hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
#endif
}

inline void device_memcpy_d2h(void* dst, const void* src, size_t bytes)
{
#ifdef CUDA
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
#else
    hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
#endif
}

inline void
    coop_launch(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, 0));
#else
    GPU_ERR_CHECK(hipLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, 0));
#endif
}

//-----------------------------------------------------------------------------
// OpenCL atomic load/store code for experiment only. HIP will have its own implementations.

#ifndef CUDA
enum class MemoryOrder : int
{
    RELAXED = __ATOMIC_RELAXED,
    ACQUIRE = __ATOMIC_ACQUIRE,
    RELEASE = __ATOMIC_RELEASE,
    ACQ_REL = __ATOMIC_ACQ_REL,
    SEQ_CST = __ATOMIC_SEQ_CST,
};

enum class MemoryScope : int
{
    WORK_ITEM       = __OPENCL_MEMORY_SCOPE_WORK_ITEM,
    WORK_GROUP      = __OPENCL_MEMORY_SCOPE_WORK_GROUP,
    DEVICE          = __OPENCL_MEMORY_SCOPE_DEVICE,
    ALL_SVM_DEVICES = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
#if defined(cl_intel_subgroups) || defined(cl_khr_subgroups)
    SUB_GROUP = __OPENCL_MEMORY_SCOPE_SUB_GROUP
#endif
};

template <typename T>
__device__ inline T hip_atomic_load(volatile T* object,
                                    MemoryOrder order = MemoryOrder::SEQ_CST,
                                    MemoryScope scope = MemoryScope::DEVICE)
{
    assert(order != MemoryOrder::RELEASE);
    assert(order != MemoryOrder::ACQ_REL);
    return __opencl_atomic_load((_Atomic T*)object, int(order), int(scope));
}

template <typename T>
__device__ inline void hip_atomic_store(volatile T* object,
                                        T           desired,
                                        MemoryOrder order = MemoryOrder::RELAXED,
                                        MemoryScope scope = MemoryScope::DEVICE)
{
    assert(order != MemoryOrder::ACQUIRE);
    assert(order != MemoryOrder::ACQ_REL);
    __opencl_atomic_store((_Atomic T*)object, desired, int(order), int(scope));
}

#endif

//
// The helper functions to build on AMD and NV platforms
//
#ifndef SYNC_WORKGROUPS_HELPER_H
#define SYNC_WORKGROUPS_HELPER_H

#ifdef CUDA
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#else
// clang-format off
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
// clang-format on
#endif

#include <atomic>
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

#ifdef CUDA
static cudaEvent_t g_timer_start, g_timer_stop;
#else
static hipEvent_t g_timer_start, g_timer_stop;
#endif

inline void device_event_create()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaEventCreate(&g_timer_start));
    GPU_ERR_CHECK(cudaEventCreate(&g_timer_stop));
#else
    GPU_ERR_CHECK(hipEventCreate(&g_timer_start));
    GPU_ERR_CHECK(hipEventCreate(&g_timer_stop));
#endif
}

inline void device_event_destroy()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaEventDestroy(g_timer_start));
    GPU_ERR_CHECK(cudaEventDestroy(g_timer_stop));
#else
    GPU_ERR_CHECK(hipEventDestroy(g_timer_start));
    GPU_ERR_CHECK(hipEventDestroy(g_timer_stop));
#endif
}

inline void device_event_record_start()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaEventRecord(g_timer_start));
#else
    GPU_ERR_CHECK(hipEventRecord(g_timer_start));
#endif
}

inline void device_event_record_stop()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaEventRecord(g_timer_stop));
#else
    GPU_ERR_CHECK(hipEventRecord(g_timer_stop));
#endif
}

inline void device_event_synchronize_stop()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaEventSynchronize(g_timer_stop));
#else
    GPU_ERR_CHECK(hipEventSynchronize(g_timer_stop));
#endif
}

inline float device_event_elapsed_time()
{
    float gpu_time;
#ifdef CUDA
    GPU_ERR_CHECK(cudaEventElapsedTime(&gpu_time, g_timer_start, g_timer_stop));
#else
    GPU_ERR_CHECK(hipEventElapsedTime(&gpu_time, g_timer_start, g_timer_stop));
#endif
    return gpu_time;
}

// The grid size of coop launched has upper bound
static bool check_occupancy(
    void* func, int grid_size, int workgroup_size, size_t dynamic_lds_size, int device_id = 0)
{
    int max_blocks_per_sm, max_blocks_per_grid;
#ifdef CUDA
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, device_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, func, workgroup_size, dynamic_lds_size);
#else
    hipDeviceProp_t device_properties;
    hipGetDeviceProperties(&device_properties, 0);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, func, workgroup_size, dynamic_lds_size);
#endif

    max_blocks_per_grid = device_properties.multiProcessorCount * max_blocks_per_sm;
    std::cout << "max_blocks_per_sm " << max_blocks_per_sm << ", max_blocks_per_grid "
              << max_blocks_per_grid << std::endl;

    if(grid_size > max_blocks_per_grid)
    {
        std::cout << "Please reduce gridsize from " << grid_size << " to " << max_blocks_per_grid
                  << std::endl;
        return false;
    }

    return true;
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

#endif

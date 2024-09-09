//
// The helper functions to build on AMD and NV platforms
//
#ifndef RUNTIME_API_WRAPPER_H
#define RUNTIME_API_WRAPPER_H

#ifdef CUDA
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

//-----------------------------------------------------------------------------
// Error check

inline const char* gpu_get_err_name(int e)
{
#ifdef CUDA
    return cudaGetErrorName((cudaError_t)e);
#else
    return hipGetErrorName((hipError_t)e);
#endif
}

inline const char* gpu_get_err_string(int e)
{
#ifdef CUDA
    return cudaGetErrorString((cudaError_t)e);
#else
    return hipGetErrorString((hipError_t)e);
#endif
}

#define GPU_ERR_CHECK(call)                                                                    \
    if(1)                                                                                      \
    {                                                                                          \
        auto ec = (call);                                                                      \
        if(ec)                                                                                 \
        {                                                                                      \
            std::cerr << "Failed at " << __FILE__ << ":" << __LINE__ << " in " << __FUNCTION__ \
                      << "()\n"                                                                \
                      << "Error code: " << ec << ", " << gpu_get_err_name(ec) << ", "          \
                      << gpu_get_err_string(ec) << std::endl;                                  \
            exit(ec);                                                                          \
        }                                                                                      \
    }

//-----------------------------------------------------------------------------
// Runtime API wrappers

inline void device_reset()
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaDeviceReset());
#else
    GPU_ERR_CHECK(hipDeviceReset());
#endif
}

inline void set_device(int device_id)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaSetDevice(device_id));
#else
    GPU_ERR_CHECK(hipSetDevice(device_id));
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
    GPU_ERR_CHECK(cudaFree(ptr));
#else
    GPU_ERR_CHECK(hipFree(ptr));
#endif
}

inline void device_memcpy_h2d(void* dst, const void* src, size_t bytes)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#else
    GPU_ERR_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#endif
}

inline void device_memcpy_d2h(void* dst, const void* src, size_t bytes)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#else
    GPU_ERR_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
#endif
}

inline void device_memcpy_to_symbol_h2d(const void* dst, const void* src, size_t bytes)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaMemcpyToSymbol(dst, src, bytes));
#else
    GPU_ERR_CHECK(hipMemcpyToSymbol(dst, src, bytes));
#endif
}

inline void device_memcpy_from_symbol(void* dst, const void* src, size_t bytes)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaMemcpyFromSymbol(dst, src, bytes));
#else
    GPU_ERR_CHECK(hipMemcpyFromSymbol(dst, src, bytes));
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

inline void device_get_attr(int device_id, int& max_thread_per_block, int& max_lds_size)
{
#ifdef CUDA
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&max_thread_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&max_lds_size, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
#else
    GPU_ERR_CHECK(hipDeviceGetAttribute(
        &max_thread_per_block, hipDeviceAttributeMaxThreadsPerBlock, device_id));
    GPU_ERR_CHECK(
        hipDeviceGetAttribute(&max_lds_size, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));
#endif
}

// simple events pair for timing convenience
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

//-----------------------------------------------------------------------------
// Unified checking functions

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
    hipGetDeviceProperties(&device_properties, device_id);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, func, workgroup_size, dynamic_lds_size);
#endif

    max_blocks_per_grid = device_properties.multiProcessorCount * max_blocks_per_sm;

    if(grid_size > max_blocks_per_grid)
    {
        std::cout << "max_blocks_per_sm " << max_blocks_per_sm << ", max_blocks_per_grid "
                  << max_blocks_per_grid << "\nPlease reduce gridsize " << grid_size << ".\n";
        return false;
    }

    return true;
}

static float max_memory_bandwidth_GB_per_s(const int device_id)
{
    int max_memory_clock_kHz = 0;
    int memory_bus_width     = 0;

    // Try to get the device bandwidth from an environment variable first
    char* peak_bw_from_env = NULL;
    peak_bw_from_env       = getenv("PEAK_MEM_BW");
    if(peak_bw_from_env != NULL)
    {
        return atof(peak_bw_from_env);
    }

#ifdef CUDA
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&max_memory_clock_kHz, cudaDevAttrMemoryClockRate, device_id));
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device_id));
#else
    GPU_ERR_CHECK(
        hipDeviceGetAttribute(&max_memory_clock_kHz, hipDeviceAttributeMemoryClockRate, device_id));
    GPU_ERR_CHECK(
        hipDeviceGetAttribute(&memory_bus_width, hipDeviceAttributeMemoryBusWidth, device_id));
#endif

    auto max_memory_clock_MHz = static_cast<float>(max_memory_clock_kHz) / 1000.0;
    // multiply by 2.0 because transfer is bidirectional
    // divide by 8.0 because bus width is in bits and we want bytes
    // divide by 1000 to convert MB to GB
    float result = (max_memory_clock_MHz * 2.0 * memory_bus_width / 8.0) / 1000.0;
    return result;
}

//----------------------------------------------------------------------------
// nontemporal load/store functions

#ifndef NONTEMPORAL

template <typename T>
__device__ __forceinline__ T load(const T& ref)
{
    return ref;
}

template <typename T>
__device__ __forceinline__ void store(const T& value, T& ref)
{
    ref = value;
}

#else

template <typename T>
__device__ __forceinline__ T load(const T& ref)
{
    return __builtin_nontemporal_load(&ref);
}

// Specialization for float2
template <>
__device__ __forceinline__ float2 load(const float2& ref)
{
    return float2(__builtin_nontemporal_load(&ref.x), __builtin_nontemporal_load(&ref.y));
}

// Specialization for double2
template <>
__device__ __forceinline__ double2 load(const double2& ref)
{
    return double2(__builtin_nontemporal_load(&ref.x), __builtin_nontemporal_load(&ref.y));
}

template <typename T>
__device__ __forceinline__ void store(const T& value, T& ref)
{
    __builtin_nontemporal_store(value, &ref);
}

// Specialization for float2
template <>
__device__ __forceinline__ void store(const float2& value, float2& ref)
{
    __builtin_nontemporal_store(value.x, &ref.x);
    __builtin_nontemporal_store(value.y, &ref.y);
}

// Specialization for double2
template <>
__device__ __forceinline__ void store(const double2& value, double2& ref)
{
    __builtin_nontemporal_store(value.x, &ref.x);
    __builtin_nontemporal_store(value.y, &ref.y);
}

#endif

//----------------------------------------------------------------------------
// Other helper functions

// Return min, mean, median, max of numbers in a vector
template <typename T>
static std::tuple<T, T, T, T> m_4(std::vector<T> const& a)
{
    std::vector<T> v(a);

    if(v.empty())
        return std::make_tuple(0, 0, 0, 0);

    auto min_max = std::minmax_element(v.begin(), v.end());
    auto min     = *(min_max.first);
    auto max     = *(min_max.second);
    auto mean    = std::accumulate(v.begin(), v.end(), 0.0) / v.size();

    auto half = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + half, v.end());
    auto median = v[half];

    if(v.size() % 2 == 0) // even case
    {
        auto max_it = std::max_element(v.begin(), v.begin() + half);
        median      = (*max_it + median) / 2.0;
    }

    return std::make_tuple(min, mean, median, max);
}

// Flip key and value of a map
template <typename A, typename B>
std::pair<B, A> flip_pair(const std::pair<A, B>& p)
{
    return std::pair<B, A>(p.second, p.first);
}

template <typename A, typename B>
std::multimap<B, A> flip_map(const std::map<A, B>& src)
{
    std::multimap<B, A> dst;
    std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()), flip_pair<A, B>);
    return dst;
}

static void print_line()
{
    std::cout << "-------------------------------------------------------------------------"
                 "-------"
              << std::endl;
}

#endif
///////////////////////////////////////////////////////////////////////////////
//  The code is trying to explore lds conflict with various simple access
//  patterns.
//
//  build with hipcc:
//      /opt/rocm/bin/hipcc lds_conflict_test.cpp  -o lds_conflict_test
//  build with nvcc:
//      nvcc -x cu -std=c++11 -D CUDA lds_conflict_test.cpp  -o lds_conflict_test
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdio.h>
#include <tuple>
#include <vector>

#ifdef CUDA
#include <cuda_runtime.h>
#else
#define HIP_ENABLE_PRINTF
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#endif

//-----------------------------------------------------------------------------
// Helper functions

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
#ifdef CUDA
#define WAVE_FRONT_SIZE 32
#else
#define WAVE_FRONT_SIZE 64
#endif

// Linear access should never cause conflict
template <typename T>
__global__ void lds_linear(T* __restrict__ data)
{
    // dynamic lds
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   lds = reinterpret_cast<T*>(shmem_ptr);

    lds[threadIdx.x] = data[0];
    __syncthreads();

    // dummy code to avoid compiler opt
    if(threadIdx.x == (WAVE_FRONT_SIZE + 1))
        data[threadIdx.x] = lds[threadIdx.x];
}

// Simulate adjacent 2 threads access the same bank
template <typename T>
__global__ void lds_conflict_2ways(T* __restrict__ data)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   lds = reinterpret_cast<T*>(shmem_ptr);

    // thread idx: 0,  1,  2,  3,  4,  5, ...
    // lds offset: 0, 32,  1, 33,  2, 34, ...
    lds[(threadIdx.x / 2) + (threadIdx.x % 2) * 32] = data[0];
    __syncthreads();

    if(threadIdx.x == (WAVE_FRONT_SIZE + 1))
        data[threadIdx.x] = lds[threadIdx.x];
}

// Every quarter of a wave access the same bank.
// To investigate: it seems slight different on NV and AMD if the data type is not int.
template <typename T>
__global__ void lds_overlapping_quarter_wave(T* __restrict__ data)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   lds = reinterpret_cast<T*>(shmem_ptr);

    // thread idx: 0,  1,  2, ... 15, 16, 17, ... 31, ...
    // lds offset: 0,  1,  2, ... 15, 32, 33, ... 47, ...
    const int quarter = WAVE_FRONT_SIZE / 4;
    int       offset  = (threadIdx.x / quarter) * 32 + (threadIdx.x % quarter);

    lds[offset] = data[0];
    __syncthreads();

    if(threadIdx.x == (WAVE_FRONT_SIZE + 1))
        data[threadIdx.x] = lds[threadIdx.x];
}

template <typename T>
void lds_access_test(const int kernel_id, const int batch)
{
    size_t total_size = WAVE_FRONT_SIZE * batch;
    size_t lds_bytes
        = WAVE_FRONT_SIZE * sizeof(T) * 64; // make sure it is large enough to play with

    dim3 grid(batch);
    dim3 workgroup(WAVE_FRONT_SIZE);

    std::cout << "--------------------------------------------------------------------------------"
              << "\nkernel_id " << kernel_id << "\nlen " << WAVE_FRONT_SIZE << ", batch " << batch
              << "\ngrid: " << grid.x << ", " << grid.y << ", " << grid.z
              << ", workgroup: " << workgroup.x << ", " << workgroup.y << ", " << workgroup.z
              << ", lds bytes: " << lds_bytes << std::endl;

    T* d_out;

#ifdef CUDA
    GPU_ERR_CHECK(cudaMalloc(&d_out, total_size * sizeof(T)));
#else
    GPU_ERR_CHECK(hipMalloc(&d_out, total_size * sizeof(T)));
#endif

    for(auto i = 0; i < 3; i++)
    {
#ifdef CUDA
        cudaEvent_t start, stop;
        GPU_ERR_CHECK(cudaEventCreate(&start));
        GPU_ERR_CHECK(cudaEventCreate(&stop));
        GPU_ERR_CHECK(cudaEventRecord(start));
#else
        hipEvent_t start, stop;
        GPU_ERR_CHECK(hipEventCreate(&start));
        GPU_ERR_CHECK(hipEventCreate(&stop));
        GPU_ERR_CHECK(hipEventRecord(start));
#endif

        switch(kernel_id)
        {
        case 0:
            lds_linear<T><<<grid, workgroup, lds_bytes>>>(d_out);
            break;
        case 1:
            lds_conflict_2ways<T><<<grid, workgroup, lds_bytes>>>(d_out);
            break;
        case 2:
            lds_overlapping_quarter_wave<T><<<grid, workgroup, lds_bytes>>>(d_out);
            break;
        default:
            break;
        }

        float gpu_time;
#ifdef CUDA
        GPU_ERR_CHECK(cudaEventRecord(stop));
        GPU_ERR_CHECK(cudaEventSynchronize(stop));
        GPU_ERR_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
#else
        GPU_ERR_CHECK(hipEventRecord(stop));
        GPU_ERR_CHECK(hipEventSynchronize(stop));
        GPU_ERR_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
#endif
        std::cout << "gpu_time: " << gpu_time << std::endl;

#ifdef CUDA
        GPU_ERR_CHECK(cudaEventDestroy(start));
        GPU_ERR_CHECK(cudaEventDestroy(stop));
#else
        GPU_ERR_CHECK(hipEventDestroy(start));
        GPU_ERR_CHECK(hipEventDestroy(stop));
#endif
    }

    std::cout << "done.\n";

#ifdef CUDA
    GPU_ERR_CHECK(cudaFree(d_out));
#else
    GPU_ERR_CHECK(hipFree(d_out));
#endif
}

int main()
{
    lds_access_test<int>(0, 1);
    lds_access_test<int>(1, 1);
    lds_access_test<int>(2, 1);

    lds_access_test<double>(0, 1);
    lds_access_test<double>(1, 1);
    lds_access_test<double>(2, 1);

    lds_access_test<float2>(0, 1);
    lds_access_test<float2>(1, 1);
    lds_access_test<float2>(2, 1);

    lds_access_test<double2>(0, 1);
    lds_access_test<double2>(1, 1);
    lds_access_test<double2>(2, 1);

    return 0;
}

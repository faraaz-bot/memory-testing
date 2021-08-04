//
//
// The code verifies approaches of all workgroups synchronization with atomic.
// The config should be safe(small enough) to run with/wihout cooperative groups
// launch as an experiment.
//
// The device kernels do simple "plus one" twice on an int array[BATCH * LEN]
// with grid(BATCH), blocks(LEN).
//
// Build:
//    with hipcc:
//      /opt/rocm/bin/hipcc sync_with_atomic.cpp  -o sync_with_atomic
//    with nvcc:
//      nvcc -x cu --std=c++11 -D CUDA sync_with_atomic.cpp  -o sync_with_atomic_cuda
//
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

#define LEN 4
#define BATCH 16

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
// Device functions

__device__ int g_counter = 0;

void __device__ plus_one_device(int* a)
{
    __shared__ int lds[LEN];
    lds[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];

    __syncthreads();

    lds[threadIdx.x]                         = lds[threadIdx.x] + 1;
    a[blockIdx.x * blockDim.x + threadIdx.x] = lds[threadIdx.x];
}

void __global__ plus_one_twice_atomic_atomicOr(int* a)
{
    // do the 1st round task
    plus_one_device(a);

    //__syncthreads();
    __threadfence();

    // the leading thread in each workgroup sets its bit in g_counter,
    // and then wait for all workgroups sync.
    if(threadIdx.x == 0)
    {
        atomicOr(&g_counter, 0x1 << blockIdx.x);

        while(atomicOr(&g_counter, 0x1 << blockIdx.x) != 0xFFFF)
        {
        }
    }

    // do the 2nd round task
    plus_one_device(a);

    // clean up g_counter with leading thread in the 1st workgroup only
    if(threadIdx.x == 0 && blockIdx.x == 0)
        atomicExch(&g_counter, 0);
}

// Do the same as plus_one_twice_atomic_atomicOr but with __threadfence wait
void __global__ plus_one_twice_atomic_memfence(int* a)
{
    plus_one_device(a);
    //__syncthreads();
    __threadfence();

    if(threadIdx.x == 0)
    {
        atomicOr(&g_counter, 0x1 << blockIdx.x);

        __threadfence();
        while(g_counter != 0xFFFF)
        {
        }
    }

    plus_one_device(a);

    if(threadIdx.x == 0 && blockIdx.x == 0)
        atomicExch(&g_counter, 0);
}

//-----------------------------------------------------------------------------

int main()
{
    int total_size  = LEN * BATCH;
    int total_bytes = total_size * sizeof(int);

    int* h_in  = new int[total_size];
    int* h_out = new int[total_size];
    int* d_data;

    int trial = 100;

    for(auto i = 0; i < total_size; i++)
        h_in[i] = i;

    void* kernelArgs[] = {reinterpret_cast<void*>(&d_data)};

    device_malloc((void**)&d_data, total_bytes);
    device_memcpy_h2d(d_data, h_in, total_bytes);

    std::cout << "------ 0 normal launch: plus_one_twice_atomic_atomicOr\n";
    for(auto j = 0; j < trial; j++)
    {
        plus_one_twice_atomic_atomicOr<<<BATCH, LEN>>>(d_data);
    }
    device_synchronize();

    std::cout << "------ 1 cooper launch: plus_one_twice_atomic_atomicOr\n";
    for(auto j = 0; j < trial; j++)
    {
        coop_launch((void*)plus_one_twice_atomic_atomicOr, BATCH, LEN, kernelArgs, 0);
    }
    device_synchronize();

    std::cout << "------ 2 normal launch: plus_one_twice_atomic_memfence\n";
    for(auto j = 0; j < trial; j++)
    {
        plus_one_twice_atomic_memfence<<<BATCH, LEN>>>(d_data);
    }
    device_synchronize();

    std::cout << "------ 3 cooper launch: plus_one_twice_atomic_memfence\n";
    for(auto j = 0; j < trial; j++)
    {
        coop_launch((void*)plus_one_twice_atomic_atomicOr, BATCH, LEN, kernelArgs, 0);
    }
    device_synchronize();

    device_memcpy_d2h(h_out, d_data, total_bytes);
    device_free(d_data);

    const int KERNEL_RUNS = 4;
    for(auto i = 0; i < total_size; i++)
    {
        if(h_out[i] != (i + KERNEL_RUNS * trial * 2)) //plus one twice for each run
        {
            std::cout << "failed at " << i << ",  " << h_out[i] << " should be "
                      << (i + KERNEL_RUNS * trial * 2) << std::endl;
            exit(-1);
        }
    }

    std::cout << "Done.\n";

    delete[] h_in;
    delete[] h_out;
    return 0;
}

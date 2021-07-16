//
// The code mocks 2 inplace rocFFT SBCC kernels behavior(access pattern) with
// simple plus_one() operation. In the 1st pass, we expect to handle the data
// along the second dim with column tiles. In the 2nd pass, we expect to handle
// the data along the slowest dim with column tiles. Here we are trying various
// solutions to do the sync between 2 passes.
//
//
// Build:
//    /opt/rocm/bin/hipcc sync_workgroups.cpp  -o sync_workgroups
//
//

// clang-format off
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
// clang-format on

#include <iostream>

using namespace cooperative_groups;
using cooperative_groups::thread_group;
namespace cg = cooperative_groups;

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
                                        MemoryOrder order = MemoryOrder::SEQ_CST,
                                        MemoryScope scope = MemoryScope::DEVICE)
{
    assert(order != MemoryOrder::ACQUIRE);
    assert(order != MemoryOrder::ACQ_REL);
    __opencl_atomic_store((_Atomic T*)object, desired, int(order), int(scope));
}

//-----------------------------------------------------------------------------

void __device__ plus_one_device(int* a, const int b_stride, const int c_stride)
{
    __shared__ int lds[3];
    int offset       = blockIdx.x / b_stride * 9 + blockIdx.x % b_stride + threadIdx.x * c_stride;
    lds[threadIdx.x] = a[offset];
    __syncthreads();
    //printf("blockIdx.x %d, threadIdx.x %d, offset %d\n", (int)blockIdx.x, (int)threadIdx.x, offset);
    lds[threadIdx.x] = lds[threadIdx.x] + 1;
    a[offset]        = lds[threadIdx.x];
}

//-----------------------------------------------------------------------------
// solution_0: call plus_one twice

void __global__ plus_one(int* a, const int b_stride, const int c_stride)
{
    plus_one_device(a, b_stride, c_stride);
}

void solution_0(int* d_data)
{
    int   b_stride     = 3;
    int   c_stride     = 3;
    void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

    hipLaunchCooperativeKernel(plus_one, dim3(9), dim3(3), kernelArgs, 0, 0);

    //hipDeviceSynchronize();

    b_stride = 9;
    c_stride = 9;

    hipLaunchCooperativeKernel(plus_one, dim3(9), dim3(3), kernelArgs, 0, 0);
}

//-----------------------------------------------------------------------------
// solution_1: sync with HIP coop
void __global__ plus_one_twice_hip_coop(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    plus_one_device(a, bs, cs);

    cg::grid_group g = cooperative_groups::this_grid();
    g.sync();

    bs = 9;
    cs = 9;
    plus_one_device(a, bs, cs);
}

void solution_1(int* d_data)
{
    int   b_stride     = 3;
    int   c_stride     = 3;
    void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

    hipLaunchCooperativeKernel(plus_one_twice_hip_coop, dim3(9), dim3(3), kernelArgs, 0, 0);
}

//-----------------------------------------------------------------------------
// solution_2: sync all workgroups with atomic
__device__ int g_counter = 0;

void __global__ plus_one_twice_sync_all_atomic(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    plus_one_device(a, bs, cs);

    __threadfence();

    if(threadIdx.x == 0)
    {
        //printf("blockIdx.x %3d, g_counter %x\n", (int)blockIdx.x, g_counter);
        atomicAdd(&g_counter, 1);
        while(hip_atomic_load<int>(&g_counter, MemoryOrder::RELAXED, MemoryScope::DEVICE) < 9)
        {
        }
    }

    bs = 9;
    cs = 9;
    plus_one_device(a, bs, cs);

    if(threadIdx.x == 0)
    {
        atomicAdd(&g_counter, -1);
    }
}

void solution_2(int* d_data)
{
    int   b_stride     = 3;
    int   c_stride     = 3;
    void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

    hipLaunchCooperativeKernel(plus_one_twice_sync_all_atomic, dim3(9), dim3(3), kernelArgs, 0, 0);
}

//-----------------------------------------------------------------------------
// solution_3: sync partioned workgroups with atomic
__device__ int g_partitioned_counters[3] = {0, 0, 0};

void __global__ plus_one_twice_sync_partion_atomic(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    plus_one_device(a, bs, cs);

    __threadfence();

    int counterIdx = blockIdx.x % 3;
    if(threadIdx.x == 0)
    {
        //printf("blockIdx.x %3d, counterIdx %x\n", (int)blockIdx.x, counterIdx);
        atomicAdd(&g_partitioned_counters[counterIdx], 1);
        while(hip_atomic_load<int>(
                  &g_partitioned_counters[counterIdx], MemoryOrder::RELAXED, MemoryScope::DEVICE)
              < 3)
        {
        }
    }

    bs = 9;
    cs = 9;
    plus_one_device(a, bs, cs);

    if(threadIdx.x == 0)
    {
        atomicAdd(&g_partitioned_counters[counterIdx], -1);
    }
}

void solution_3(int* d_data)
{
    int   b_stride     = 3;
    int   c_stride     = 3;
    void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

    hipLaunchCooperativeKernel(
        plus_one_twice_sync_partion_atomic, dim3(9), dim3(3), kernelArgs, 0, 0);
}

//-----------------------------------------------------------------------------

int main()
{
    int total_size  = 3 * 3 * 3;
    int total_bytes = total_size * sizeof(int);

    int* h_data = new int[total_size];
    int* d_data;

    for(auto i = 0; i < total_size; i++)
        h_data[i] = i;
    GPU_ERR_CHECK(hipMalloc(&d_data, total_bytes));
    GPU_ERR_CHECK(hipMemcpy(d_data, h_data, total_bytes, hipMemcpyHostToDevice));

    solution_3(d_data);

    GPU_ERR_CHECK(hipMemcpy(h_data, d_data, total_bytes, hipMemcpyDeviceToHost));

    for(auto i = 0; i < total_size; i++)
        std::cout << "a[" << i << "]" << h_data[i] << ", " << std::endl;
    GPU_ERR_CHECK(hipFree(d_data));
    delete[] h_data;
    return 0;
}

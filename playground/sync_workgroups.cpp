//
//
//
//
// Build:
//    /opt/rocm/bin/hipcc sync_workgroups.cpp  -o sync_workgroups
//
//

// clang-format off
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
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

void __device__ plus_one_device(int* a, const int b_stride, const int c_stride)
{
    __shared__ int lds[3];
    int            offset = blockIdx.x / b_stride * 9 + blockIdx.x + threadIdx.x * c_stride;
    lds[threadIdx.x]      = a[offset];
    //printf("blockIdx.x %d, threadIdx.x %d, offset %d\n", (int)blockIdx.x, (int)threadIdx.x, offset);
    __syncthreads();

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
        atomicAdd(&g_counter, 1);
        printf("blockIdx.x %3d, g_counter %x\n", (int)blockIdx.x, g_counter);
        while(g_counter < 9)
        {
        };
    }

    bs cs = 9;
    plus_one_device(a, bs, cs);
}

void solution_2(int* d_data)
{
    int   b_stride     = 3;
    int   c_stride     = 3;
    void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

    hipLaunchCooperativeKernel(plus_one_twice_sync_all_atomic, dim3(9), dim3(3), kernelArgs, 0, 0);
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

    solution_2(d_data);

    GPU_ERR_CHECK(hipMemcpy(h_data, d_data, total_bytes, hipMemcpyDeviceToHost));

    for(auto i = 0; i < total_size; i++)
        std::cout << "a[" << i << "]" << h_data[i] << ", " << std::endl;
    GPU_ERR_CHECK(hipFree(d_data));
    delete[] h_data;
    return 0;
}

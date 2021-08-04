//
//
// The test is trying to measure overhead of normal kernel and hipLaunchCooperativeKernel
// with simple device kernel.
//
// The device kernel is doing simple "plus one" on an int array[128] with grid(2), blocks(64).
// The test collects total elapsed time from hipEvent with incremental trial sizes. And
// no devicesynchronize between each trial run. And the first run of each collection is
// ignored.
//
// Build:
//    with hipcc:
//      /opt/rocm/bin/hipcc coop_launch_test.cpp  -o coop_launch_test
//    with nvcc:
//      nvcc -x cu --std=c++11 -D CUDA coop_launch_test.cpp  -o coop_launch_cuda_test
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
// Device function

#define LEN 64
#define BATCH 2

void __global__ plus_one(int* a)
{
    __shared__ int lds[LEN];
    lds[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];

    __syncthreads();

    lds[threadIdx.x]                         = lds[threadIdx.x] + 1;
    a[blockIdx.x * blockDim.x + threadIdx.x] = lds[threadIdx.x];
}

//-----------------------------------------------------------------------------

int main()
{
    int total_size  = LEN * BATCH;
    int total_bytes = total_size * sizeof(int);

    int* h_in  = new int[total_size];
    int* h_out = new int[total_size];
    int* d_data;

    for(auto i = 0; i < total_size; i++)
        h_in[i] = i;

#ifdef CUDA
    GPU_ERR_CHECK(cudaMalloc(&d_data, total_bytes));
    GPU_ERR_CHECK(cudaMemcpy(d_data, h_in, total_bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
#else
    GPU_ERR_CHECK(hipMalloc(&d_data, total_bytes));
    GPU_ERR_CHECK(hipMemcpy(d_data, h_in, total_bytes, hipMemcpyHostToDevice));
    hipEvent_t start, stop;
#endif

    for(int trial = 1; trial <= 1000000; trial *= 10)
    {
        std::cout << "--- --- Trial " << trial << std::endl;

#ifdef CUDA
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
#else
        hipEventCreate(&start);
        hipEventCreate(&stop);
#endif

        /// launch the kernel with the normal way, and check the elapsed time

        // warm up once
        plus_one<<<BATCH, LEN>>>(d_data);

#ifdef CUDA
        cudaEventRecord(start);
#else
        hipEventRecord(start);
#endif

        for(int j = 0; j < trial; j++)
        {
            plus_one<<<BATCH, LEN>>>(d_data);
        }

        float gpu_time;

#ifdef CUDA
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
#else
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&gpu_time, start, stop);
#endif

        std::cout << "\ttotal normal kernel launch "
                  << ": " << gpu_time << " ms\n";

        /// launch the kernel with the hipLaunchCooperativeKernel, and check the elapsed time

        // warm up once
        void* kernelArgs[] = {(void*)&d_data};
#ifdef CUDA
        GPU_ERR_CHECK(
            cudaLaunchCooperativeKernel((void*)plus_one, dim3(BATCH), dim3(LEN), kernelArgs, 0, 0));
#else
        GPU_ERR_CHECK(hipLaunchCooperativeKernel(plus_one, BATCH, LEN, kernelArgs, 0, 0));
#endif

#ifdef CUDA
        cudaEventRecord(start);
#else
        hipEventRecord(start);
#endif

        for(int j = 0; j < trial; j++)
        {
#ifdef CUDA
            GPU_ERR_CHECK(cudaLaunchCooperativeKernel(
                (void*)plus_one, dim3(BATCH), dim3(LEN), kernelArgs, 0, 0));
#else
            GPU_ERR_CHECK(hipLaunchCooperativeKernel(plus_one, BATCH, LEN, kernelArgs, 0, 0));
#endif
        }

#ifdef CUDA
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);
#else
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&gpu_time, start, stop);
#endif
        std::cout << "\ttotal cooper kernel launch "
                  << ": " << gpu_time << " ms\n";

#ifdef CUDA
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
#else
        hipEventDestroy(start);
        hipEventDestroy(stop);
#endif
    }

#ifdef CUDA
    cudaMemcpy(h_out, d_data, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
#else
    hipMemcpy(h_out, d_data, total_bytes, hipMemcpyDeviceToHost);
    hipFree(d_data);
#endif

    // for(auto i = 0; i < total_size; i++)
    //     std::cout << "a[" << i << "]" << h_out[i] << ", ";
    // std::cout << std::endl;

    delete[] h_in;
    delete[] h_out;
    return 0;
}

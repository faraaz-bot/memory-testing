//
//
// Build:
//    with hipcc:
//      /opt/rocm/bin/hipcc global_counter_test.cpp  -o global_counter_test
//

#include <atomic>
#include <hip/hip_runtime.h>
#include <iostream>

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

// simple kernel to use counters

void __global__ plus_eight(int* counters)
{
    // atomicAdd(&counters[0], 8);
    // atomicAdd(&counters[1], 8);

    __atomic_add_fetch(&counters[0], 8, __ATOMIC_SEQ_CST);
    __atomic_add_fetch(&counters[1], 8, __ATOMIC_SEQ_CST);

    printf("counters: %d, %d\n", counters[0], counters[1]);
}

//-----------------------------------------------------------------------------
void test_0()
{
    int* h_counters;
    int* d_counters;

    GPU_ERR_CHECK(hipHostMalloc(&h_counters, 2 * sizeof(int), hipHostMallocCoherent));
    GPU_ERR_CHECK(hipMalloc((void**)&d_counters, 2 * sizeof(int)));

    // To test the below, you need export HSA_FORCE_FINE_GRAIN_PCIE=1
    // GPU_ERR_CHECK(
    //     hipExtMallocWithFlags((void**)&d_counters, 2 * sizeof(int), hipDeviceMallocFinegrained));

    for(int j = 0; j < 3; j++)
    {
        h_counters[0] = 0;
        h_counters[1] = -1;
        GPU_ERR_CHECK(hipMemcpy(d_counters, h_counters, 2 * sizeof(int), hipMemcpyHostToDevice));
        plus_eight<<<1, 1>>>(d_counters);
    }

    GPU_ERR_CHECK(hipHostFree(h_counters));
    GPU_ERR_CHECK(hipFree(d_counters));
}

//-----------------------------------------------------------------------------
void test_1()
{
    int* h_counters;
    int* d_counters;

    GPU_ERR_CHECK(hipHostMalloc(&h_counters, 2 * sizeof(int), hipHostMallocMapped));
    //GPU_ERR_CHECK(hipHostMalloc(&h_counters, 2 * sizeof(int), hipHostMallocCoherent));
    h_counters[0] = 0;
    h_counters[1] = -1;
    GPU_ERR_CHECK(hipHostGetDevicePointer((void**)&d_counters, (void*)h_counters, 0));

    for(int j = 0; j < 3; j++)
    {
        __atomic_store_n(&h_counters[0], 0, __ATOMIC_SEQ_CST);
        __atomic_store_n(&h_counters[1], -1, __ATOMIC_SEQ_CST);
        plus_eight<<<1, 1>>>(d_counters);

        hipDeviceSynchronize();
    }

    GPU_ERR_CHECK(hipHostFree(h_counters));
}

//-----------------------------------------------------------------------------

int main()
{
    std::cout << "------test_0\n";
    test_0();
    std::cout << "------test_1\n";
    test_1();
    return 0;
}

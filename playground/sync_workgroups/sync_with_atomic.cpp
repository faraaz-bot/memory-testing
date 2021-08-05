//
//
// The code verifies approaches of all workgroups synchronization with atomic.
// The config should be safe(small enough) to run with/wihout cooperative groups
// launch as an experiment.
//
// The device kernels do simple "plus one" twice on an int array[BATCH * LEN]
// with grid(BATCH), blocks(LEN). Each thread does "plus one" inplace in the 1st
// round and does "plus one" with LEN right shift in the 2nd round, which requires
// memfence between them.
//
// Build:
//    with hipcc:
//      /opt/rocm/bin/hipcc sync_with_atomic.cpp  sync_workgroups_helper.h -o sync_with_atomic
//    with nvcc:
//      nvcc -x cu --std=c++11 -D CUDA sync_with_atomic.cpp  sync_workgroups_helper.h -o sync_with_atomic_cuda
//
//

#include "sync_workgroups_helper.h"

#define LEN 4
#define BATCH 16

//-----------------------------------------------------------------------------
// Device functions

__device__ int g_counter = 0;

void __device__ plus_one_device(int* a, int offset)
{
    __shared__ int lds[LEN];
    lds[threadIdx.x] = a[offset + threadIdx.x];

    __syncthreads();

    lds[threadIdx.x]        = lds[threadIdx.x] + 1;
    a[offset + threadIdx.x] = lds[threadIdx.x];
}

void __global__ plus_one_twice_atomic_atomicOr(int* a)
{
    // do the 1st round task
    plus_one_device(a, blockIdx.x % BATCH * LEN);

    __syncthreads();

    // the leading thread in each workgroup sets its bit in g_counter,
    // and then wait for all workgroups sync.
    if(threadIdx.x == 0)
    {
        atomicOr(&g_counter, 0x1 << blockIdx.x);

        while(atomicOr(&g_counter, 0x1 << blockIdx.x) != 0xFFFF)
        {
        }
    }

    __threadfence();

    // do the 2nd round task
    plus_one_device(a, (blockIdx.x + 1) % BATCH * LEN);

    // clean up g_counter with leading thread in the 1st workgroup only
    if(threadIdx.x == 0 && blockIdx.x == 0)
        atomicExch(&g_counter, 0);
}

// Bad example with race condition!!!
// Do the same as plus_one_twice_atomic_atomicOr but with __threadfence wait
// void __global__ plus_one_twice_atomic_memfence(int* a)
// {
//     plus_one_device(a, blockIdx.x % BATCH * LEN);

//     if(threadIdx.x == 0)
//     {
//         atomicOr(&g_counter, 0x1 << blockIdx.x);

//         __threadfence();
//         while(g_counter != 0xFFFF)
//         {
//         }
//     }

//     __threadfence();

//     plus_one_device(a, (blockIdx.x + 1) % BATCH * LEN);

//     if(threadIdx.x == 0 && blockIdx.x == 0)
//         atomicExch(&g_counter, 0);
// }

//-----------------------------------------------------------------------------

int main()
{
    int total_size  = LEN * BATCH;
    int total_bytes = total_size * sizeof(int);

    int* h_in  = new int[total_size];
    int* h_out = new int[total_size];
    int* d_data;

    int trial = 1000;

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

    // Bad example with race condition!!!
    // std::cout << "------ 2 normal launch: plus_one_twice_atomic_memfence\n";
    // for(auto j = 0; j < trial; j++)
    // {
    //     plus_one_twice_atomic_memfence<<<BATCH, LEN>>>(d_data);
    // }
    // device_synchronize();

    // std::cout << "------ 3 cooper launch: plus_one_twice_atomic_memfence\n";
    // for(auto j = 0; j < trial; j++)
    // {
    //     coop_launch((void*)plus_one_twice_atomic_atomicOr, BATCH, LEN, kernelArgs, 0);
    // }
    // device_synchronize();

    device_memcpy_d2h(h_out, d_data, total_bytes);
    device_free(d_data);

    const int KERNEL_RUNS = 2;
    for(auto i = 0; i < total_size; i++)
    {
        if(h_out[i] != (i + KERNEL_RUNS * trial * 2)) //plus one twice for each run
        {
            std::cout << "Failed at " << i << ",  " << h_out[i] << " should be "
                      << (i + KERNEL_RUNS * trial * 2) << std::endl;
            exit(-1);
        }
    }

    std::cout << "Done.\n";

    delete[] h_in;
    delete[] h_out;
    return 0;
}

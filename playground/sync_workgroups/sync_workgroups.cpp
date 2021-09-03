//
//
// The code mocks 2 inplace rocFFT SBCC kernels behavior(access pattern) with
// simple plus_one() operation. In the 1st pass, we expect to handle the data
// along the second dim with column tiles. In the 2nd pass, we expect to handle
// the data along the slowest dim with column tiles. Here we are trying various
// solutions to do the sync between 2 passes.
//
//
// solution_0: do it with 2 kernel launch.
// solution_1: do it in 1 kernel launch with HIP cooperative_groups grid.sync().
// solution_2: do it in 1 kernel launch with atomic sync.
// solution_3: do it in 1 kernel launch with atomic sync, but for partitioned workgroups.
// solution_4: do it in 1 kernel launch without atomic. The idea is from section 5.3
//             in http://eprints.cs.vt.edu/archive/00001087/01/TR_GPU_synchronization.pdf.
// solution_5: do it in 1 kernel launch but sync tasks with atomic.
//
//
// Build:
//    with hipcc:
//      /opt/rocm/bin/hipcc sync_workgroups.cpp  sync_workgroups_helper.h -o sync_workgroups
//    with nvcc:
//      nvcc -x cu --std=c++11 -D CUDA sync_workgroups.cpp -o sync_workgroups_cuda
//

#include "sync_workgroups_helper.h"

// FIXME: use coop_launch in helper.h
void coop_launch_(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem)
{
#ifdef CUDA
    GPU_ERR_CHECK(cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, 0));
#else
    GPU_ERR_CHECK(hipLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, 0));
#endif
}

#define LEN 4 // the length along one dimension
#define SOLUTION_NUM 6

typedef void (*TestCall)(int*, bool);

//-----------------------------------------------------------------------------

void __device__ plus_one_device(int* a, const int tile_id, const int b_stride, const int c_stride)
{
    __shared__ int lds[LEN];
    int offset = tile_id / b_stride * LEN * LEN + tile_id % b_stride + threadIdx.x * c_stride;

    if(threadIdx.x < LEN)
        lds[threadIdx.x] = a[offset];

    __syncthreads();

    if(threadIdx.x < LEN)
    {
        //printf("tile_id %d, block %d, threadIdx.x %d, offset %d\n",
        //    (int)tile_id, (int)blockIdx.x, (int)threadIdx.x, offset);
        lds[threadIdx.x] = lds[threadIdx.x] + 1;
        a[offset]        = lds[threadIdx.x];
    }
}

//-----------------------------------------------------------------------------
// solution_0: call plus_one twice

void __global__ plus_one(int* a, const int b_stride, const int c_stride)
{
    plus_one_device(a, blockIdx.x, b_stride, c_stride);
}

void solution_0(int* d_data, bool coop_launch)
{
    int b_stride = LEN;
    int c_stride = LEN;

    if(!coop_launch)
    {
        plus_one<<<dim3(LEN * LEN), dim3(LEN)>>>(d_data, b_stride, b_stride);

        b_stride = LEN * LEN;
        c_stride = LEN * LEN;

        plus_one<<<dim3(LEN * LEN), dim3(LEN)>>>(d_data, b_stride, b_stride);
    }
    else
    {
        void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

        coop_launch_((void*)plus_one, (LEN * LEN), LEN, kernelArgs, 0);

        b_stride = LEN * LEN;
        c_stride = LEN * LEN;

        coop_launch_((void*)plus_one, (LEN * LEN), LEN, kernelArgs, 0);
    }
}

//-----------------------------------------------------------------------------
// solution_1: sync with HIP coop
void __global__ plus_one_twice_hip_coop(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    plus_one_device(a, blockIdx.x, bs, cs);

    cg::grid_group g = cooperative_groups::this_grid();
    g.sync();

    bs = LEN * LEN;
    cs = LEN * LEN;
    plus_one_device(a, blockIdx.x, bs, cs);

    g.sync();
}

void solution_1(int* d_data, bool coop_launch)
{
    //NB: no normal launch for this approach
    int   b_stride     = LEN;
    int   c_stride     = LEN;
    void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

    coop_launch_((void*)plus_one_twice_hip_coop, (LEN * LEN), LEN, kernelArgs, 0);
}

//-----------------------------------------------------------------------------
// solution_2: sync all workgroups with atomic
__device__ int g_counter = 0;

void __global__ plus_one_twice_sync_all_atomic(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    plus_one_device(a, blockIdx.x, bs, cs);

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

    __syncthreads();
    __threadfence();

    bs = LEN * LEN;
    cs = LEN * LEN;
    plus_one_device(a, blockIdx.x, bs, cs);

    // clean up g_counter with leading thread in the 1st workgroup only
    if(threadIdx.x == 0 && blockIdx.x == 0)
        atomicExch(&g_counter, 0);
}

void solution_2(int* d_data, bool coop_launch)
{
    if(!coop_launch)
    {
        plus_one_twice_sync_all_atomic<<<dim3(LEN * LEN), dim3(LEN)>>>(d_data, LEN, LEN);
    }
    else
    {
        int   b_stride     = LEN;
        int   c_stride     = LEN;
        void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

        coop_launch_((void*)plus_one_twice_sync_all_atomic, (LEN * LEN), LEN, kernelArgs, 0);
    }
}

//-----------------------------------------------------------------------------
// solution_3: sync partioned workgroups with atomic
__device__ int g_partitioned_counters[LEN] = {0, 0, 0, 0};

void __global__ plus_one_twice_sync_partion_atomic(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    plus_one_device(a, blockIdx.x, bs, cs);

    __syncthreads();

    int counterIdx = blockIdx.x % LEN;
    if(threadIdx.x == 0)
    {
        //printf("blockIdx.x %3d, counterIdx %x\n", (int)blockIdx.x, counterIdx);
        atomicAdd(&g_partitioned_counters[counterIdx], 1);

        while(atomicAdd(&g_partitioned_counters[counterIdx], 0) != LEN)
        {
        }
    }

    __syncthreads();
    __threadfence();

    bs = LEN * LEN;
    cs = LEN * LEN;
    plus_one_device(a, blockIdx.x, bs, cs);

    if(threadIdx.x == 0 && blockIdx.x < LEN)
    {
        atomicExch(&g_partitioned_counters[counterIdx], 0);
    }
}

void solution_3(int* d_data, bool coop_launch)
{
    //check_occupancy((void*)plus_one_twice_sync_partion_atomic, LEN * LEN, LEN, 0);

    if(!coop_launch)
    {
        plus_one_twice_sync_partion_atomic<<<dim3(LEN * LEN), dim3(LEN)>>>(d_data, LEN, LEN);
    }
    else
    {
        int   b_stride     = LEN;
        int   c_stride     = LEN;
        void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

        coop_launch_((void*)plus_one_twice_sync_partion_atomic, (LEN * LEN), LEN, kernelArgs, 0);
    }
}

//-----------------------------------------------------------------------------
// solution_4: sync all workgroups without atomic
__device__ int g_in[LEN * LEN]  = {0};
__device__ int g_out[LEN * LEN] = {0};

__device__ void grid_sync(int val)
{
    if(threadIdx.x == 0)
    {
        atomicExch(&g_in[blockIdx.x], val);
    }

    if(blockIdx.x == 0) // assume work group 0 has enough threads to cover gridDim.x
    {
        if(threadIdx.x < gridDim.x)
        {
            while(atomicAdd(&g_in[threadIdx.x], 0) != val)
            {
            }
        }
        __syncthreads();

        if(threadIdx.x < gridDim.x)
        {
            atomicExch(&g_out[blockIdx.x], val);
        }
    }

    if(threadIdx.x == 0)
    {
        while(atomicAdd(&g_out[blockIdx.x], 0) != val)
        {
        }
    }
    __syncthreads();
}

void __global__ plus_one_twice_sync_all_no_atomic(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    plus_one_device(a, blockIdx.x, bs, cs);

    __threadfence();

    grid_sync(1);

    bs = LEN * LEN;
    cs = LEN * LEN;
    plus_one_device(a, blockIdx.x, bs, cs);

    grid_sync(0);
}

void solution_4(int* d_data, bool coop_launch)
{
    //NB: normal launch doesn't work yet
    //plus_one_twice_sync_all_no_atomic<<<dim3(LEN * LEN), dim3(LEN)>>>(d_data, LEN, LEN);

    int   b_stride     = LEN;
    int   c_stride     = LEN;
    void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

    coop_launch_((void*)plus_one_twice_sync_all_no_atomic, (LEN * LEN), (LEN * LEN), kernelArgs, 0);
}

//-----------------------------------------------------------------------------
// solution_5: sync tasks with atomic
__device__ int g_counter_pass0 = 0;
__device__ int g_counter_pass1 = -1;

#define TASK_NUM (LEN * LEN)

void __global__ plus_one_twice_sync_tasks_atomic(int* a, const int b_stride, const int c_stride)
{
    int bs = b_stride;
    int cs = c_stride;

    int task_id = 0;
    int done    = 0;

    __shared__ int l_task_id;

    if(threadIdx.x == 0 && blockIdx.x == 0)
        atomicExch(&g_counter_pass1, -1);
    if(threadIdx.x == 0)
    {
        while(atomicAdd(&g_counter_pass1, 0) > -1)
        {
        }
    }
    __syncthreads();

    while(done != 1)
    {
        if(threadIdx.x == 0) // only the leading thread in workgroup acquires task_id with atomic
        {
            task_id   = atomicAdd(&g_counter_pass0, 1);
            l_task_id = task_id; // broadcast to lds to share with all working threads later
        }

        __syncthreads();

        task_id = l_task_id;

        if(task_id < TASK_NUM)
        {
            plus_one_device(a, task_id, bs, cs);

            __syncthreads();

            // only the leading thread in the workgroup handling the last task triggers next pass
            if(task_id == TASK_NUM - 1)
            {
                __threadfence();

                if(threadIdx.x == 0)
                {
                    //printf("--- block %d triggers next pass\n", (int)blockIdx.x);
                    atomicExch(&g_counter_pass1, 0);
                }

                done = 1;
            }
        }
        else
        {
            done = 1;
            __threadfence();
        }
    }

    done = 0;

    // barrier for next pass
    if(threadIdx.x == 0)
    {
        while(atomicAdd(&g_counter_pass1, 0) < 0)
        {
        }
    }
    __syncthreads();

    //__threadfence();

    bs = LEN * LEN;
    cs = LEN * LEN;

    while(done != 1)
    {
        if(threadIdx.x == 0)
        {
            task_id   = atomicAdd(&g_counter_pass1, 1);
            l_task_id = task_id;
        }

        __syncthreads();

        task_id = l_task_id;

        if(task_id < TASK_NUM && task_id > -1)
        {
            plus_one_device(a, task_id, bs, cs);

            __syncthreads();

            if(task_id == TASK_NUM - 1)
            {
                done = 1;
            }
        }
        else
        {
            done = 1;
            if(threadIdx.x == 0)
            {
                atomicExch(&g_counter_pass0, 0);
            }
        }
    }
}

void solution_5(int* d_data, bool coop_launch)
{
    if(!coop_launch)
    {
        plus_one_twice_sync_tasks_atomic<<<dim3(LEN * LEN), dim3(LEN)>>>(d_data, LEN, LEN);
    }
    else
    {
        int   b_stride     = LEN;
        int   c_stride     = LEN;
        void* kernelArgs[] = {(void*)&d_data, (void*)&b_stride, (void*)&c_stride};

        coop_launch_((void*)plus_one_twice_sync_tasks_atomic, (LEN * LEN), LEN, kernelArgs, 0);
    }
}

//-----------------------------------------------------------------------------

int main()
{
    int total_size  = LEN * LEN * LEN;
    int total_bytes = total_size * sizeof(int);

    int* h_in  = new int[total_size];
    int* h_out = new int[total_size * SOLUTION_NUM];
    int* d_data;

    for(auto i = 0; i < total_size; i++)
        h_in[i] = i;

    TestCall solution[SOLUTION_NUM];
    solution[0] = &solution_0;
    solution[1] = &solution_1;
    solution[2] = &solution_2;
    solution[3] = &solution_3;
    solution[4] = &solution_4;
    solution[5] = &solution_5;

    for(int coop_launch = 0; coop_launch <= 1; coop_launch++)
    {
        std::cout << "-- coop_launch " << coop_launch << std::endl;
        for(auto i = 0; i < SOLUTION_NUM; i++)
        {
            if(i != 4)
            {
                device_reset();
               device_malloc((void**)&d_data, total_bytes);
                device_memcpy_h2d(d_data, h_in, total_bytes);

                device_event_create();

                // warm up once
                solution[i](d_data, coop_launch);

                device_event_record_start();

                for(int j = 0; j < 1000; j++)
                {
                    solution[i](d_data, coop_launch);
                }

                device_event_record_stop();
                device_event_synchronize_stop();
                device_synchronize();
                std::cout << "solution_" << i << ": " << device_event_elapsed_time() << " ms\n";

                device_memcpy_d2h(&h_out[i * total_size], d_data, total_bytes);
                device_free(d_data);
                device_event_destroy();
            }
        }
    }

    // take the 1st run as reference to verify the output of each solution
    std::cout << "verify...\n";
    for(auto i = 1; i < SOLUTION_NUM; i++)
    {
        if(i != 4)
        {
            int j = 0;
            for(; j < total_size; j++)
            {
                if(h_out[i * total_size + j] != h_out[j])
                {
                    std::cout << "solution_" << i << ": failed at " << j << ", ref " << h_out[j]
                              << " vs " << h_out[i * total_size + j] << std::endl;
                    break;
                }
            }
            if(j == total_size)
                std::cout << "solution_" << i << ": done\n";
        }
        else
        {
            std::cout << "solution_" << i << ": bypass\n";
        }
    }

    // for(auto i = 1; i < SOLUTION_NUM; i++)
    // {
    //     std::cout << "\nsolution_" << i << std::endl;
    //     for(auto j = 0; j < total_size; j++)
    //         std::cout << "a[" << j << "]" << h_out[i * total_size + j] << ", ";
    // }
    // std::cout << std::endl;

    delete[] h_in;
    delete[] h_out;
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
//
//  Description:
//
//  Build:
//    with hipcc:
//      -- default:
//              hipcc global_1024.cpp runtime_api_wrapper.h CLI11.hpp -o global_1024
//
///////////////////////////////////////////////////////////////////////////////

#include "runtime_api_wrapper.h"

///----------------------------------------------------------------------------
/// Utils

std::map<int, std::string> Kernel_Names = {
    {0, "s0"},
    {1, "s1"},
};

// Init one element in an array with an integer number.
// The real and imag part of complex number take the same.
template <typename T>
void init_element(std::vector<T>& dst, size_t offset, int val)
{
    dst[offset] = val;
}

template <>
void init_element(std::vector<float2>& dst, size_t offset, int val)
{
    dst[offset].x = dst[offset].y = val;
}

template <>
void init_element(std::vector<double2>& dst, size_t offset, int val)
{
    dst[offset].x = dst[offset].y = val;
}

// Verify one element in an array with an integer number.
// The real and imag part of complex number take the same.
template <typename T>
bool verify_element(std::vector<T> const& dst, size_t offset, int val)
{
    return (int)dst[offset] == val;
}

template <>
bool verify_element(std::vector<float2> const& dst, size_t offset, int val)
{
    return ((int)(dst[offset].x) == val) && ((int)(dst[offset].y) == val);
}

template <>
bool verify_element(std::vector<double2> const& dst, size_t offset, int val)
{
    return ((int)(dst[offset].x) == val) && ((int)(dst[offset].y) == val);
}

///----------------------------------------------------------------------------
/// Test kernels

// Solution0: unroll for loop
template <typename T>
__launch_bounds__(128) __global__ void solution0(const T* __restrict__ in, T* __restrict__ out)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   lds = reinterpret_cast<T*>(shmem_ptr);

    T   R[8];
    int offset = blockIdx.x * 1024;
    int thread = threadIdx.x;

#pragma unroll
    for(int i = 0; i < 8; i++)
        R[i] = in[offset + thread + i * 128];

    __syncthreads();

#pragma unroll
    for(int i = 0; i < 8; i++)
        out[offset + thread + i * 128] = R[i];
}

// Solution1: flatten, and with temp vars
template <typename T>
__launch_bounds__(128) __global__
    void solution1(const T* __restrict__ inputs, T* __restrict__ outputs)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   sdata = reinterpret_cast<T*>(shmem_ptr);

    T temp_0;
    temp_0.x = 0.0f;
    temp_0.y = 0.0f;
    T temp_1;
    temp_1.x = 0.0f;
    temp_1.y = 0.0f;
    T temp_2;
    temp_2.x = 0.0f;
    temp_2.y = 0.0f;
    T temp_3;
    temp_3.x = 0.0f;
    temp_3.y = 0.0f;
    T temp_4;
    temp_4.x = 0.0f;
    temp_4.y = 0.0f;
    T temp_5;
    temp_5.x = 0.0f;
    temp_5.y = 0.0f;
    T temp_6;
    temp_6.x = 0.0f;
    temp_6.y = 0.0f;
    T temp_7;
    temp_7.x = 0.0f;
    temp_7.y = 0.0f;
    T w;
    w.x = 0.0f;
    w.y = 0.0f;
    T loc_0;
    loc_0.x = 0.0f;
    loc_0.y = 0.0f;
    unsigned int tempInt;
    tempInt = 0;
    unsigned int tempInt2;
    tempInt2 = 0;
    unsigned int shiftX;
    shiftX = 0;
    unsigned int shiftY;
    shiftY = 0;
    unsigned int shiftZ;
    shiftZ = 0;
    T iw;
    iw.x = 0.0f;
    iw.y = 0.0f;
    unsigned int stageInvocationID;
    stageInvocationID = 0;
    unsigned int blockInvocationID;
    blockInvocationID = 0;
    unsigned int sdataID;
    sdataID = 0;
    unsigned int combinedID;
    combinedID = 0;
    unsigned int inoutID;
    inoutID = 0;
    unsigned int inoutID_x;
    inoutID_x = 0;
    unsigned int inoutID_y;
    inoutID_y = 0;
    unsigned int LUTId;
    LUTId      = 0;
    shiftX     = blockIdx.y;
    shiftY     = blockIdx.x;
    shiftY     = shiftY * 1;
    shiftZ     = 0;
    shiftZ     = shiftZ + 0;
    combinedID = threadIdx.x + 0;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_0     = inputs[inoutID];
    combinedID = threadIdx.x + 128;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_1     = inputs[inoutID];
    combinedID = threadIdx.x + 256;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_2     = inputs[inoutID];
    combinedID = threadIdx.x + 384;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_3     = inputs[inoutID];
    combinedID = threadIdx.x + 512;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_4     = inputs[inoutID];
    combinedID = threadIdx.x + 640;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_5     = inputs[inoutID];
    combinedID = threadIdx.x + 768;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_6     = inputs[inoutID];
    combinedID = threadIdx.x + 896;
    inoutID_x  = combinedID % 1024;
    inoutID_y  = combinedID / 1024;
    inoutID_y  = inoutID_y + shiftY;
    inoutID    = inoutID_x;
    tempInt    = inoutID_y * 1;
    inoutID    = inoutID + tempInt;
    inoutID    = inoutID + shiftZ;
    temp_7     = inputs[inoutID];

    __syncthreads();

    combinedID       = threadIdx.x + 0;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_0;
    combinedID       = threadIdx.x + 128;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_1; //temp_4;
    combinedID       = threadIdx.x + 256;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_2;
    combinedID       = threadIdx.x + 384;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_3; //temp_6;
    combinedID       = threadIdx.x + 512;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_4; //temp_1;
    combinedID       = threadIdx.x + 640;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_5;
    combinedID       = threadIdx.x + 768;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_6; //temp_3;
    combinedID       = threadIdx.x + 896;
    inoutID_x        = combinedID % 1024;
    inoutID_y        = combinedID / 1024;
    inoutID_y        = inoutID_y + shiftY;
    inoutID          = inoutID_x;
    tempInt          = inoutID_y * 1;
    inoutID          = inoutID + tempInt;
    inoutID          = inoutID + shiftZ;
    outputs[inoutID] = temp_7;
}

//
template <typename T>
int test(int kernel_id)
{
    float max_memory_bw = max_memory_bandwidth_GB_per_s(0);

    size_t len           = 1024;
    size_t batch         = 499712;
    size_t i_total_size  = len * batch;
    size_t o_total_size  = len * batch;
    size_t i_total_bytes = i_total_size * sizeof(T);
    size_t o_total_bytes = o_total_size * sizeof(T);
    size_t lds_bytes     = 8192; //Not actually in use, but impact occupancy

    dim3   grid  = {499712 /*500000*/, 1, 1};
    dim3   block = {128, 1, 1};
    size_t trial = 10;

    float              efficiency_pct = 0;
    std::vector<float> gpu_time_samples;

    std::cout << "Kernel_" << kernel_id << " " << Kernel_Names[kernel_id] << "\n\tLength:\t" << len
              << ", \tBatch: " << batch << "\n\tGrid:\t" << grid.x << ", " << grid.y << ", "
              << grid.z << ", \tBlock: " << block.x << ", " << block.y << ", " << block.z
              << "\n\tTotal R/W MEM(MB): " << (i_total_bytes + o_total_bytes) / 1e6
              << "\tLDS(MB): " << lds_bytes / 1024 << std::endl;

    std::vector<T> in(i_total_size);
    std::vector<T> out(o_total_size);
    T *            d_in, *d_out;

    std::cout << "Generate input...\n";
    for(auto i = 0; i < batch; i++)
        for(auto j = 0; j < len; j++)
        {
            init_element(in, i * len + j, (i * len + j) % 16384);
        }

    for(size_t i = 0; i < o_total_size; i++)
    {
        //out[i].x = out[i].y = -1;
        init_element(out, i, -1);
    }

    std::cout << " Run | GPU Event Time (ms) | Memory Bandwidth (GB/s) | Memory Efficiency (%) |"
              << std::endl;

    device_malloc((void**)&d_in, i_total_bytes);
    device_malloc((void**)&d_out, o_total_bytes);
    device_memcpy_h2d(d_in, in.data(), i_total_bytes);
    device_memcpy_h2d(d_out, out.data(), o_total_bytes);

    for(auto i = 0; i < trial; i++)
    {
        device_event_create();
        device_event_record_start();

        switch(kernel_id)
        {
        case 0:
            solution0<T><<<grid, block>>>(d_in, d_in);
            break;
        case 1:
            solution1<T><<<grid, block>>>(d_in, d_in);
            break;
        default:
            break;
        }

        device_event_record_stop();
        device_event_synchronize_stop();
        device_synchronize();

        float gpu_time = device_event_elapsed_time();

        double exec_bw = (double)(i_total_bytes + o_total_bytes) / (gpu_time * 1e6);
        if(max_memory_bw != 0.0)
        {
            efficiency_pct = 100.0 * exec_bw / max_memory_bw;
        }
        gpu_time_samples.push_back(gpu_time);

        //if(verbose == 2)
        std::cout << std::setw(5) << i << "| " << std::setw(20) << std::fixed
                  << std::setprecision(5) << gpu_time << "| " << std::setw(25) << exec_bw << "| "
                  << std::setw(22) << efficiency_pct << "|" << std::endl;

        device_event_destroy();
    }

    // stats on gpu elapsed time
    std::tuple<float, float, float, float> stats           = m_4<float>(gpu_time_samples);
    float                                  median_gpu_time = std::get<2>(stats);

    double median_exec_bw = (double)(i_total_bytes + o_total_bytes) / (median_gpu_time * 1e6);
    if(max_memory_bw != 0.0)
    {
        efficiency_pct = 100.0 * median_exec_bw / max_memory_bw;
    }

    std::cout << std::endl
              << std::setw(5) << "Median " << std::setw(20) << std::fixed << std::setprecision(5)
              << median_gpu_time << ", " << std::setw(25) << median_exec_bw << ", " << std::setw(22)
              << efficiency_pct << "," << std::endl
              << std::endl;

    // if(verify)
    {
        device_memcpy_d2h(out.data(), d_in, o_total_bytes);
        std::cout << "Verify output...";

        for(auto i = 0; i < batch; i++)
        {
            for(auto j = 0; j < len; j++)
            {
                //std::cout << "(" << std::setw(2) << out[i*len + j].x << ", "
                //    << std::setw(2) <<  out[i*len + j].y  << "), ";
                auto idx = i * len + j;
                if(!verify_element(out, idx, (i * len + j) % 16384))
                {
                    std::cerr << "failed at [" << idx << "]: "
                              << "expected " << (i * len + j) % 16384 << std::endl;
                    exit(-1);
                }
            }
            //std::cout << std::endl;
        }

        std::cout << "Done." << std::endl;
    }

    print_line();

    device_free(d_in);
    device_free(d_out);

    return 0;
}

int main(int argc, char* argv[])
{
    test<float2>(0);
    test<float2>(1);
}
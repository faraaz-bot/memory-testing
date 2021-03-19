///////////////////////////////////////////////////////////////////////////////
//  build with hipcc:
//      /opt/rocm/bin/hipcc len336_memory_access.cpp  -o len336_memory_access -I /opt/rocm/hip/include/hip
//  build with nvcc:
//      nvcc -x cu -std=c++11 -D CUDA len336_memory_access.cpp  -o len336_memory_access
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

static float max_memory_bandwidth_GB_per_s()
{
#ifdef CUDA
    return 900; // assume Tesla V100-SXM2 32GB
#else
    int deviceid = 0;
    hipGetDevice(&deviceid);
    int max_memory_clock_kHz = 0;
    int memory_bus_width     = 0;
    hipDeviceGetAttribute(&max_memory_clock_kHz, hipDeviceAttributeMemoryClockRate, deviceid);
    hipDeviceGetAttribute(&memory_bus_width, hipDeviceAttributeMemoryBusWidth, deviceid);
    auto max_memory_clock_MHz = static_cast<float>(max_memory_clock_kHz) / 1000.0;
    // multiply by 2.0 because transfer is bidirectional
    // divide by 8.0 because bus width is in bits and we want bytes
    // divide by 1000 to convert MB to GB
    float result = (max_memory_clock_MHz * 2.0 * memory_bus_width / 8.0) / 1000.0;
    return result;
#endif
}

// Return min, mean, median, max of numbers in a vector
template <typename T>
std::tuple<T, T, T, T> m_4(std::vector<T> const& a)
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

//-----------------------------------------------------------------------------

// Copy from global memory to global memory directly
template <typename T>
__global__ void copy_direct(const T* __restrict__ idata,
                            T* __restrict__ odata,
                            const int n,
                            const int elem_per_thread,
                            const int rows,
                            const int padding)
{
    // Todo: template and unroll some of the params
    int base = blockIdx.x * (n + padding) * rows;

    for(int j = 0; j < rows; j++)
        for(int i = 0; i < elem_per_thread; i++)
        {
            int idx    = base + j * (n + padding) + i * blockDim.x + threadIdx.x;
            odata[idx] = idata[idx];
        }
}

// Copy from global memory to global memory through LDS
template <typename T>
__global__ void copy_lds(const T* __restrict__ idata,
                         T* __restrict__ odata,
                         const int n,
                         const int elem_per_thread,
                         const int rows,
                         const int padding)
{
#ifdef CUDA
    extern __shared__ T lds[];
#else
    HIP_DYNAMIC_SHARED(T, lds);
#endif

    int base = blockIdx.x * (n + padding) * rows;

    for(int j = 0; j < rows; j++)
        for(int i = 0; i < elem_per_thread; i++)
        {
            int idx  = j * n + i * blockDim.x + threadIdx.x;
            lds[idx] = idata[base + j * padding + idx];
        }

    __syncthreads();

    for(int j = 0; j < rows; j++)
        for(int i = 0; i < elem_per_thread; i++)
        {
            int idx                         = j * n + i * blockDim.x + threadIdx.x;
            odata[base + j * padding + idx] = lds[idx];
        }
}

// Tiled transpose through LDS
template <typename T>
__global__ void transpose_lds(const T* __restrict__ idata,
                              T* __restrict__ odata,
                              const int n,
                              const int elem_per_thread,
                              const int rows,
                              const int padding)
{
#ifdef CUDA
    extern __shared__ T lds[];
#else
    HIP_DYNAMIC_SHARED(T, lds);
#endif

    int o_base = blockIdx.x * n * rows;
    int i_base = o_base + blockIdx.x * padding * rows;

    for(int j = 0; j < rows; j++)
        for(int i = 0; i < elem_per_thread; i++)
        {
            int idx  = j * n + i * blockDim.x + threadIdx.x;
            lds[idx] = idata[i_base + j * padding + idx];
            //printf("thread %d, global read  idx %2d, lds idx %2d, value %2d\n",
            //    (int)threadIdx.x, (base + j * padding + idx),  idx, (int)(lds[idx].x));
        }

    __syncthreads();

    for(int i = 0; i < elem_per_thread; i++)
        for(int j = 0; j < rows; j++)
        {
            int lds_idx = i * blockDim.x + (blockDim.x * j + threadIdx.x) % rows * n
                          + (blockDim.x * j + threadIdx.x) / rows;
            int gw_idx    = o_base + i * blockDim.x * rows + j * blockDim.x + threadIdx.x;
            odata[gw_idx] = lds[lds_idx];
            //printf("thread %d, global write idx %2d, lds idx %2d, value %2d\n",
            //    (int)threadIdx.x, gw_idx,  lds_idx, (int)(lds[lds_idx].x));
        }
}

// Benchmark memory bandwidth efficiency with specific pattern
//
//   kernel_id       : 0 for copy_direct, 1 for copy_lds, 2 for transpose_lds
//   len             : the size of one row(the fast dimension)
//   batch           : the total size of how many rows(the second fast dimension)
//   elem_per_thread : element number handled by each thread
//   rows            : tile width
//   padding         : padding element number at the end of each row
//   trial           : repeat times
//   verbose         : show details
//   grid_x          : -1 for auto-calc
//   block_x         : -1 for auto-calc
//
template <typename T>
float mem_access_test(const int  kernel_id,
                      const int  len,
                      const int  batch,
                      const int  elem_per_thread,
                      const int  rows,
                      const int  padding,
                      const int  trial   = 10,
                      const bool verbose = false,
                      const int  grid_x  = -1,
                      const int  block_x = -1)
{
    size_t i_total_size  = len * batch;
    size_t o_total_size  = len * batch;
    size_t i_total_bytes = i_total_size * sizeof(T);
    size_t o_total_bytes = o_total_size * sizeof(T);
    size_t lds_bytes     = len * rows * sizeof(T);

    size_t i_padded_bytes = (len + padding) * batch * sizeof(T);
    size_t o_padded_bytes = i_padded_bytes;

    float              max_memory_bw  = max_memory_bandwidth_GB_per_s();
    float              efficiency_pct = 0;
    std::vector<float> efficiency_pct_samples;

    assert(batch % rows == 0);
    assert(len % elem_per_thread == 0);

    dim3 grid((grid_x == -1) ? (batch / rows) : grid_x);
    dim3 block((block_x == -1) ? (len / elem_per_thread) : block_x);

    std::ofstream   null_file("/dev/null");
    std::streambuf* stream_buffer = std::cout.rdbuf();
    if(!verbose)
        std::cout.rdbuf(null_file.rdbuf());

    std::cout << "--------------------------------------------------------------------------------"
              << "\nkernel_id " << kernel_id << "\nlen " << len << ", batch " << batch
              << ", tile len " << len / elem_per_thread << ", tile width " << rows
              << "\nelem_per_thread " << elem_per_thread << ", padding for each row " << padding
              << "\ngrid: " << grid.x << ", " << grid.y << ", " << grid.z << ", block: " << block.x
              << ", " << block.y << ", " << block.z << std::endl;
    if(kernel_id != 0)
        std::cout << "lds bytes: " << lds_bytes << std::endl;

    std::vector<T> in(i_padded_bytes);
    std::vector<T> out(o_padded_bytes);
    T *            d_in, *d_out;

    std::cout << "Generate input...\n";
    for(auto i = 0; i < batch; i++)
        for(auto j = 0; j < len; j++)
        {
            in[i * (len + padding) + j].x = in[i * (len + padding) + j].y = i * len + j;
        }

    for(size_t i = 0; i < o_padded_bytes; i++)
    {
        out[i].x = out[i].y = -1;
    }

    std::cout << " Run | GPU Event Time (ms) | Memory Throughput (GB/s) | Memory Efficiency (%) |"
              << std::endl;

#ifdef CUDA
    cudaMalloc(&d_in, i_padded_bytes);
    cudaMalloc(&d_out, o_padded_bytes);
    cudaMemcpy(d_in, in.data(), i_padded_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data(), o_padded_bytes, cudaMemcpyHostToDevice);
#else
    hipMalloc(&d_in, i_padded_bytes);
    hipMalloc(&d_out, o_padded_bytes);
    hipMemcpy(d_in, in.data(), i_padded_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_out, out.data(), o_padded_bytes, hipMemcpyHostToDevice);
#endif

    for(auto i = 0; i < trial; i++)
    {
#ifdef CUDA
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
#else
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start);
#endif

        switch(kernel_id)
        {
        case 0:
            copy_direct<T><<<grid, block>>>(d_in, d_out, len, elem_per_thread, rows, padding);
            break;
        case 1:
            copy_lds<T>
                <<<grid, block, lds_bytes>>>(d_in, d_out, len, elem_per_thread, rows, padding);
            break;
        case 2:
            transpose_lds<T>
                <<<grid, block, lds_bytes>>>(d_in, d_out, len, elem_per_thread, rows, padding);
            break;
        default:
            break;
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
        double exec_bw = (double)(i_total_bytes + o_total_bytes) / (gpu_time * 1e6);
        if(max_memory_bw != 0.0)
        {
            efficiency_pct = 100.0 * exec_bw / max_memory_bw;
        }
        efficiency_pct_samples.push_back(efficiency_pct);

        std::cout << std::setw(5) << i << "| " << std::setw(20) << std::fixed
                  << std::setprecision(5) << gpu_time << "| " << std::setw(25) << exec_bw << "| "
                  << std::setw(22) << efficiency_pct << "|" << std::endl;

#ifdef CUDA
        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess)
        {
            std::cout << "Error: " << cudaGetErrorName(err) << ", " << cudaGetErrorString(err)
                      << std::endl;
            exit(-1);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
#else
        hipError_t err = hipPeekAtLastError();
        if(err != hipSuccess)
        {
            std::cout << "Error: " << hipGetErrorName(err) << ", " << hipGetErrorString(err)
                      << std::endl;
            exit(-1);
        }

        hipEventDestroy(start);
        hipEventDestroy(stop);
#endif
    }

#ifdef CUDA
    cudaMemcpy(out.data(), d_out, o_padded_bytes, cudaMemcpyDeviceToHost);
#else
    hipMemcpy(out.data(), d_out, o_padded_bytes, hipMemcpyDeviceToHost);
#endif

    std::cout << "Verify output...";

    if(kernel_id == 2)
    {
        for(auto i = 0; i < batch / rows; i++)
        {
            auto base = i * len * rows;
            for(auto j = 0; j < len * rows; j++) // check rows x len transpose
            {
                //std::cout << "(" << std::setw(2) << out[i*len*rows + j].x << ", "
                //    << std::setw(2) <<  out[i*len*rows + j].y  << ")\n";
                int value = base + (j % rows) * len + j / rows;

                if(((int)(out[i * len * rows + j].x) != value)
                   || ((int)(out[i * len * rows + j].y) != value))
                {
                    std::cerr << "failed at [" << base + j << "]: " << out[i * len * rows + j].x
                              << ", " << out[i * len * rows + j].y << ", expected " << value
                              << std::endl;
                    exit(0);
                }
            }
        }
    }
    else
    {
        for(auto i = 0; i < batch; i++)
        {
            for(auto j = 0; j < len; j++)
            {
                //std::cout << "(" << std::setw(2) << out[i*len + j].x << ", "
                //    << std::setw(2) <<  out[i*len + j].y  << "), ";
                auto idx = i * (len + padding) + j;
                if(((int)(out[idx].x) != i * len + j) || ((int)(out[idx].y) != i * len + j))
                {
                    std::cerr << "failed at [" << idx << "]: " << out[idx].x << ", " << out[idx].y
                              << ", expected " << i * len + j << std::endl;
                    exit(0);
                }
            }
            //std::cout << std::endl;
        }
    }

    std::cout << "done.\n";

#ifdef CUDA
    cudaFree(d_in);
    cudaFree(d_out);
#else
    hipFree(d_in);
    hipFree(d_out);
#endif

    if(!verbose)
        std::cout.rdbuf(stream_buffer);

    std::tuple<float, float, float, float> stats = m_4<float>(efficiency_pct_samples);

    return std::get<2>(stats);
}

// A wrapper to run mem_access_test() with various params
template <typename T>
void tuning_mem_access_test(const int len,
                            const int batch,
                            const int elem_per_thread_max,
                            const int rows_max,
                            const int padding_max,
                            const int trial = 10)
{
    typedef std::tuple<int, int, int, int, int> key_t;

    std::map<key_t, float> results[3];

    for(auto padding = 0; padding < padding_max; padding++)
        for(auto rows = 1; rows < rows_max; rows++)
            for(auto elem_per_thread = 1; elem_per_thread < elem_per_thread_max; elem_per_thread++)
                for(auto kernel_id = 0; kernel_id < 3; kernel_id++)
                {
                    if(batch % rows == 0 && len % elem_per_thread == 0)
                    {
                        results[kernel_id]
                               [std::make_tuple(len, batch, elem_per_thread, rows, padding)]
                            = mem_access_test<T>(kernel_id,
                                                 len,
                                                 batch,
                                                 elem_per_thread,
                                                 rows,
                                                 padding,
                                                 trial,
                                                 false);
                    }
                }

    // Pop up the top 3 runs for each kernel and rerun with showing details
    for(int kernel_id = 0; kernel_id < 3; kernel_id++)
    {
        std::multimap<float, key_t> sorted_results = flip_map(results[kernel_id]);
        auto                        start          = sorted_results.crbegin();
        for(int i = 0; i < 3; i++)
        {
            auto  it  = std::next(start, i);
            float ret = mem_access_test<T>(kernel_id,
                                           std::get<0>(it->second),
                                           std::get<1>(it->second),
                                           std::get<2>(it->second),
                                           std::get<3>(it->second),
                                           std::get<4>(it->second),
                                           trial,
                                           true);
            std::cout << "Median orignal"
                      << ": " << it->first << ", rerun: " << ret << std::endl;
        }
    }
}

int main()
{
    std::cout << "Run case 336 ---------------------------------------\n";
    tuning_mem_access_test<double2>(336, 18816, 9, 6, 10);

    std::cout << "Run case 256 ---------------------------------------\n";
    tuning_mem_access_test<double2>(256, 24696, 9, 6, 10);

    return 0;
}

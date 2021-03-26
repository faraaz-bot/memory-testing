///////////////////////////////////////////////////////////////////////////////
//  build with hipcc:
//      /opt/rocm/bin/hipcc len336_memory_access.cpp  -o len336_memory_access
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

static float max_memory_bandwidth_GB_per_s(const int device_id)
{
    int max_memory_clock_kHz = 0;
    int memory_bus_width     = 0;
#ifdef CUDA
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&max_memory_clock_kHz, cudaDevAttrMemoryClockRate, device_id));
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device_id));
#else
    GPU_ERR_CHECK(
        hipDeviceGetAttribute(&max_memory_clock_kHz, hipDeviceAttributeMemoryClockRate, device_id));
    GPU_ERR_CHECK(
        hipDeviceGetAttribute(&memory_bus_width, hipDeviceAttributeMemoryBusWidth, device_id));
#endif

    auto max_memory_clock_MHz = static_cast<float>(max_memory_clock_kHz) / 1000.0;
    // multiply by 2.0 because transfer is bidirectional
    // divide by 8.0 because bus width is in bits and we want bytes
    // divide by 1000 to convert MB to GB
    float result = (max_memory_clock_MHz * 2.0 * memory_bus_width / 8.0) / 1000.0;
    return result;
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
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   lds = reinterpret_cast<T*>(shmem_ptr);
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
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   lds = reinterpret_cast<T*>(shmem_ptr);
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
//   elem_per_thread : element number handled by each thread per row
//   rows            : tile width
//   padding         : padding element number at the end of each row
//   trial           : repeat times
//   verbose         : show details
//   grid_x          : -1 for auto-calc
//   block_x         : -1 for auto-calc
//
template <typename T>
float mem_access_test(const int   kernel_id,
                      const int   len,
                      const int   batch,
                      const int   elem_per_thread,
                      const int   rows,
                      const int   padding,
                      const float max_memory_bw,
                      const int   trial   = 10,
                      const bool  verbose = false,
                      const int   grid_x  = -1,
                      const int   block_x = -1)
{
    size_t i_total_size  = len * batch;
    size_t o_total_size  = len * batch;
    size_t i_total_bytes = i_total_size * sizeof(T);
    size_t o_total_bytes = o_total_size * sizeof(T);
    size_t lds_bytes     = len * rows * sizeof(T);

    size_t i_padded_bytes = (len + padding) * batch * sizeof(T);
    size_t o_padded_bytes = i_padded_bytes;

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
              << ", " << block.y << ", " << block.z
              << "\ntotal MB: " << (i_total_bytes + o_total_bytes) / 1e6 << std::endl;
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
    GPU_ERR_CHECK(cudaMalloc(&d_in, i_padded_bytes));
    GPU_ERR_CHECK(cudaMalloc(&d_out, o_padded_bytes));
    GPU_ERR_CHECK(cudaMemcpy(d_in, in.data(), i_padded_bytes, cudaMemcpyHostToDevice));
    GPU_ERR_CHECK(cudaMemcpy(d_out, out.data(), o_padded_bytes, cudaMemcpyHostToDevice));
#else
    GPU_ERR_CHECK(hipMalloc(&d_in, i_padded_bytes));
    GPU_ERR_CHECK(hipMalloc(&d_out, o_padded_bytes));
    GPU_ERR_CHECK(hipMemcpy(d_in, in.data(), i_padded_bytes, hipMemcpyHostToDevice));
    GPU_ERR_CHECK(hipMemcpy(d_out, out.data(), o_padded_bytes, hipMemcpyHostToDevice));
#endif

    for(auto i = 0; i < trial; i++)
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
        GPU_ERR_CHECK(cudaEventRecord(stop));
        GPU_ERR_CHECK(cudaEventSynchronize(stop));
        GPU_ERR_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
#else
        GPU_ERR_CHECK(hipEventRecord(stop));
        GPU_ERR_CHECK(hipEventSynchronize(stop));
        GPU_ERR_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
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
        GPU_ERR_CHECK(cudaEventDestroy(start));
        GPU_ERR_CHECK(cudaEventDestroy(stop));
#else
        GPU_ERR_CHECK(hipEventDestroy(start));
        GPU_ERR_CHECK(hipEventDestroy(stop));
#endif
    }

#ifdef CUDA
    GPU_ERR_CHECK(cudaMemcpy(out.data(), d_out, o_padded_bytes, cudaMemcpyDeviceToHost));
#else
    GPU_ERR_CHECK(hipMemcpy(out.data(), d_out, o_padded_bytes, hipMemcpyDeviceToHost));
#endif

    std::cout << "Verify output...";

    if(kernel_id == 2)
    {
        for(auto i = 0; i < batch / rows; i++)
        {
            auto base = i * len * rows;
            for(auto j = 0; j < len * rows; j++) // check rows * len transpose
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
    GPU_ERR_CHECK(cudaFree(d_in));
    GPU_ERR_CHECK(cudaFree(d_out));
#else
    GPU_ERR_CHECK(hipFree(d_in));
    GPU_ERR_CHECK(hipFree(d_out));
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
                            const int padding_max = 10,
                            const int device_id   = 0,
                            const int trial       = 20)
{
    typedef std::tuple<int, int, int, int, int> key_t;

    std::map<key_t, float> results[3];

#ifdef CUDA
    GPU_ERR_CHECK(cudaSetDevice(device_id));
#else
    GPU_ERR_CHECK(hipSetDevice(device_id));
#endif

    float max_memory_bw = max_memory_bandwidth_GB_per_s(device_id);

    int max_thread_per_block = 0;
    int max_lds_size         = 0;
#ifdef CUDA
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&max_thread_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    GPU_ERR_CHECK(
        cudaDeviceGetAttribute(&max_lds_size, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
#else
    GPU_ERR_CHECK(hipDeviceGetAttribute(
        &max_thread_per_block, hipDeviceAttributeMaxThreadsPerBlock, device_id));
    GPU_ERR_CHECK(
        hipDeviceGetAttribute(&max_lds_size, hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));
#endif
    // Todo: might check grid size if necessary

    int elem_per_thread_min = std::max(1, (len + max_thread_per_block - 1) / max_thread_per_block);
    int rows_min            = 1;
    int adjusted_rows_max   = rows_max;
    while(adjusted_rows_max * len * sizeof(T) > max_lds_size)
    {
        adjusted_rows_max--;
    }

    if(adjusted_rows_max < rows_min)
        std::cout << "Warning: adjusted_rows_max " << adjusted_rows_max
                  << " is too small, ignore the test.\n";

    if(elem_per_thread_max < elem_per_thread_min)
        std::cout << "Warning: elem_per_thread_max " << elem_per_thread_max
                  << " is too small, ignore the test.\n";

    for(auto padding = 0; padding <= padding_max; padding++)
        for(auto rows = rows_min; rows <= adjusted_rows_max; rows++)
            for(auto elem_per_thread = elem_per_thread_min; elem_per_thread <= elem_per_thread_max;
                elem_per_thread++)
                for(auto kernel_id = 0; kernel_id < 3; kernel_id++)
                {
                    // Handle divisible cases only
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
                                                 max_memory_bw,
                                                 trial,
                                                 true);
                    }
                }

    // Pop up the top 3 runs for each kernel and rerun with showing details
    for(int kernel_id = 0; kernel_id < 3; kernel_id++)
    {
        std::multimap<float, key_t> sorted_results = flip_map(results[kernel_id]);
        auto                        start          = sorted_results.crbegin();
        for(auto i = 0; i < std::min<int>(1, sorted_results.size()); i++)
        {
            auto  it  = std::next(start, i);
            float ret = mem_access_test<T>(kernel_id,
                                           std::get<0>(it->second),
                                           std::get<1>(it->second),
                                           std::get<2>(it->second),
                                           std::get<3>(it->second),
                                           std::get<4>(it->second),
                                           max_memory_bw,
                                           trial,
                                           true);
            std::cout << "Median original"
                      << ": " << it->first << ", rerun: " << ret << std::endl;
        }
    }
}

int main()
{

    std::cout << "Run case 108 ---------------------------------------\n";
    tuning_mem_access_test<float2>(108, 46656, 9, 12);

    std::cout << "Run case 200 ---------------------------------------\n";
    tuning_mem_access_test<float2>(200, 20200, 5, 10);

    std::cout << "Run case 256 ---------------------------------------\n";
    tuning_mem_access_test<double2>(256, 24696, 8, 8);

    std::cout << "Run case 336 ---------------------------------------\n";
    tuning_mem_access_test<double2>(336, 18816, 9, 6);

    std::cout << "Run case 4096 ---------------------------------------\n";
    tuning_mem_access_test<double2>(4096, 16384, 32, 8, 2);

    std::cout << "Run case 100 ---------------------------------------\n";
    tuning_mem_access_test<double2>(100, 1000000, 5, 10, 3);

    return 0;
}

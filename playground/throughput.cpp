#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <fftw3.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#include "timer.h"

static const int elements_per_thread = 4;
static const int warm_up = 0;

using dtype = double4;

struct CT4
{
    static const int n             = 4;
    static const int log2n         = 2;
    static const int shared_memory = 4;
};

struct CT8
{
    static const int n             = 8;
    static const int log2n         = 3;
    static const int shared_memory = 8;
};

struct CT16
{
    static const int n             = 16;
    static const int log2n         = 4;
    static const int shared_memory = 16;
};

struct CT32
{
    static const int n             = 32;
    static const int log2n         = 5;
    static const int shared_memory = 32;
};

struct CT64
{
    static const int n             = 64;
    static const int log2n         = 6;
    static const int shared_memory = 64;
};

struct CT128
{
    static const int n             = 128;
    static const int log2n         = 7;
    static const int shared_memory = 128;
};

struct CT256
{
    static const int n             = 256;
    static const int log2n         = 8;
    static const int shared_memory = 256;
};

struct CT512
{
    static const int n             = 512;
    static const int log2n         = 9;
    static const int shared_memory = 512;
};

struct CT1024
{
    static const int n             = 1024;
    static const int log2n         = 10;
    static const int shared_memory = 1024;
};

struct CT2048
{
    static const int n             = 2048;
    static const int log2n         = 11;
    static const int shared_memory = 2048;
};

#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return {};

using namespace std;

//
// Random inputs
//
template<typename T>
vector<T> random_vector(size_t n)
{
    vector<T>              x(n);
    random_device                     rd;
    mt19937                           gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
    for(size_t i = 0; i < n; ++i)
    {
        x[i].x = dis(gen);
    }
    return x;
}

template<typename T>
void __device__ throughput_kernel_work(T* x, int stride, int thread)
{
  #pragma unroll elements_per_thread
  for (int i=0; i < elements_per_thread; ++i)
    x[i*stride + thread].x *= x[i*stride + thread].y;
}

template<typename T, class params>
void __global__ __launch_bounds__(params::n/elements_per_thread) throughput_kernel(T* x)
{
    int offset = hipBlockIdx_x * hipBlockDim_x;
    int thread = hipThreadIdx_x;
    int stride = hipBlockDim_x / elements_per_thread;
    __shared__ T y[params::shared_memory];

    #pragma unroll elements_per_thread
    for (int i=0; i < elements_per_thread; ++i)
      y[i*stride + thread] = x[offset + i*stride + thread];
    __syncthreads();

    throughput_kernel_work<T>(y, stride, thread);
    __syncthreads();

    #pragma unroll elements_per_thread
    for (int i=0; i < elements_per_thread; ++i)
      x[offset + i*stride + thread] = y[i*stride + thread];
}

template<typename T>
inline void throughput_kernel_dispatch(int nbatch, int nx, T* X) {
  switch (nx) {
    case 4:
      throughput_kernel<T, CT4><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 8:
      throughput_kernel<T, CT8><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 16:
      throughput_kernel<T, CT16><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 32:
      throughput_kernel<T, CT32><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 64:
      throughput_kernel<T, CT64><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 128:
      throughput_kernel<T, CT128><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 256:
      throughput_kernel<T, CT256><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 512:
      throughput_kernel<T, CT512><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 1024:
      throughput_kernel<T, CT1024><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    case 2048:
      throughput_kernel<T, CT2048><<<nbatch, nx/elements_per_thread>>>(X);
      break;
    }
}

template<typename T>
float gpu_throughput(vector<T> const& x, int nx, int nbatch)
{
    T* X;
    HIP_CHECK(hipMalloc(&X, nx * nbatch * sizeof(T)));
    HIP_CHECK(hipMemcpy(X, x.data(), nx * nbatch * sizeof(T), hipMemcpyHostToDevice));

    GPUTimer timer;
    for (int i = 0; i < warm_up; ++i)
      throughput_kernel_dispatch<T>(nbatch, nx, X);
    timer.tic();
    throughput_kernel_dispatch<T>(nbatch, nx, X);
    timer.toc();

    HIP_CHECK(hipFree(X));

    return timer.elapsed();
}

template<typename T>
void test1d(size_t n, size_t nbatch)
{
    double GiB = double(n * nbatch * sizeof(T)) / 1024 / 1024 / 1024;
    cout << "# 1d test" << endl;
    cout << "1d input length:   " << n << " (" << nbatch << ")" << endl;
    cout << "1d input size:     " << GiB << "GiB" << endl;

    auto x = random_vector<T>(n * nbatch);
    auto dt = gpu_throughput<T>(x, n, nbatch);

    cout << "GPU kernel time:   " << dt << "ms" << endl;
    cout << "GPU throughput:    " << GiB * 1000 / dt << " GiB/s streaming" << endl;
}

int main(int argc, char* argv[])
{
    size_t length = 2048;
    size_t nbatch = 1;
    if(argc > 1)
        length = stoi(argv[1]);
    if(argc > 2)
        nbatch = stoi(argv[2]);

    test1d<dtype>(length, nbatch);
}

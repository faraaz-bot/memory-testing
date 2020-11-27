//
// Cooley-Tukey playground.
//
// Simple 1d, complex to complex, power of 2.
//

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

struct CT4
{
    static const int n             = 4;
    static const int log2n         = 2;
    static const int shared_memory = 4;
    static const int threads       = 2;
};

struct CT8
{
    static const int n             = 8;
    static const int log2n         = 3;
    static const int shared_memory = 8;
    static const int threads       = 4;
};

struct CT16
{
    static const int n             = 16;
    static const int log2n         = 4;
    static const int shared_memory = 16;
    static const int threads       = 8;
};

struct CT32
{
    static const int n             = 32;
    static const int log2n         = 5;
    static const int shared_memory = 32;
    static const int threads       = 16;
};

struct CT64
{
    static const int n             = 64;
    static const int log2n         = 6;
    static const int shared_memory = 64;
    static const int threads       = 32;
};

struct CT128
{
    static const int n             = 128;
    static const int log2n         = 7;
    static const int shared_memory = 128;
    static const int threads       = 64;
};

struct CT256
{
    static const int n             = 256;
    static const int log2n         = 8;
    static const int shared_memory = 256;
    static const int threads       = 128;
};

struct CT512
{
    static const int n             = 512;
    static const int log2n         = 9;
    static const int shared_memory = 512;
    static const int threads       = 256;
};

struct CT1024
{
    static const int n             = 1024;
    static const int log2n         = 10;
    static const int shared_memory = 1024;
    static const int threads       = 512;
};

struct CT2048
{
    static const int n             = 2048;
    static const int log2n         = 11;
    static const int shared_memory = 2048;
    static const int threads       = 1024;
};

#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return {};

using namespace std;
using gpu_result = pair<float, vector<fftw_complex>>;

//
// Random inputs
//
vector<fftw_complex> random_vector(size_t n)
{
    vector<fftw_complex>              x(n);
    random_device                     rd;
    mt19937                           gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
    for(size_t i = 0; i < n; ++i)
    {
        x[i][0] = dis(gen);
        x[i][1] = dis(gen);
    }
    return x;
}

//
// Copy helper for fftw_complex (which aren't assignable!)
//
vector<fftw_complex> copy(vector<fftw_complex> const& x)
{
    vector<fftw_complex> z(x.size());
    for(size_t i = 0; i < x.size(); ++i)
    {
        z[i][0] = x[i][0];
        z[i][1] = x[i][1];
    }
    return z;
}

//
// FFTW backed FFT
//
vector<fftw_complex> fft_fftw(vector<fftw_complex> const& x, int nx, int nbatch)
{
    auto z = copy(x);
    // clang-format off
    auto p = fftw_plan_many_dft(1, &nx, nbatch,
                                z.data(), nullptr, 1, nx,
                                z.data(), nullptr, 1, nx,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    fftw_execute(p);
    fftw_destroy_plan(p);
    return z;
}

vector<fftw_complex> fft_fftw_2d(vector<fftw_complex> const& x, int nx, int ny)
{
    auto z = copy(x);
    auto p = fftw_plan_dft_2d(nx, ny, z.data(), z.data(), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    return z;
}

vector<fftw_complex> fft_fftw_3d(vector<fftw_complex> const& x, int nx, int ny, int nz)
{
    auto z = copy(x);
    auto p = fftw_plan_dft_3d(nx, ny, nz, z.data(), z.data(), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    return z;
}

//
// Cooley-Tukey FFT
//

size_t bitreverse(size_t x, size_t n)
{
    size_t r = 0;
    for(size_t i = 0; i < n; ++i)
    {
        r <<= 1;
        r |= x & 1;
        x >>= 1;
    }
    return r;
}

//
// Cooley-Tukey, re-order first, FFT on the GPU
//

template <bool sync>
__device__ void cooley_tukey_dif_iter__(hipDoubleComplex* x, int thread, int N, int M)
{
    if constexpr(sync)
        __syncthreads();

    // convert thread to i in [0, N]; skip odd blocks
    int i = ((M - 1) & thread) + ((~(M - 1) & thread) << 1);
    int j = i + M;
    int m = i % M;

    double cost, sint;
    sincospi(-double(m) / M, &sint, &cost);

    hipDoubleComplex xi = x[i];
    hipDoubleComplex xj = x[j];
    hipDoubleComplex d;

    d.x = cost * xj.x - sint * xj.y;
    d.y = sint * xj.x + cost * xj.y;

    x[i] = xi + d;
    x[j] = xi - d;
}

__device__ void cooley_tukey_dif__(hipDoubleComplex* x, int thread, int N)
{
    int M = 1; // size of current block

    while(M < N)
    {
        if(M > 64)
            cooley_tukey_dif_iter__<true>(x, thread, N, M);
        else
            cooley_tukey_dif_iter__<false>(x, thread, N, M);
        M <<= 1;
    }
}

void __device__ cooley_tukey_dif_wtwiddles_iter__(
    hipDoubleComplex* __restrict__ x, hipDoubleComplex* __restrict__ T, int thread, int M, int P)
{
    // convert thread to i in [0, N]; skip odd blocks
    int i = ((M - 1) & thread) + ((~(M - 1) & thread) << 1);
    int j = i + M;
    int m = i % M;

    hipDoubleComplex t  = T[m * P];
    hipDoubleComplex xi = x[i];
    hipDoubleComplex xj = x[j];
    hipDoubleComplex d;

    d.x = t.x * xj.x - t.y * xj.y;
    d.y = t.y * xj.x + t.x * xj.y;

    x[i] = xi + d;
    x[j] = xi - d;
}

template <class params>
void __device__ cooley_tukey_dif_wtwiddles__(hipDoubleComplex* __restrict__ x,
                                             hipDoubleComplex* __restrict__ T,
                                             int thread)
{
    int M = 1;
    int P = params::n >> 1;

    static const int iters_no_sync = 6;
    static const int iters_no_sync = 7;

    if constexpr(params::log2n >= iters_no_sync)
    {
        for(int itr = 0; itr < iters_no_sync; ++itr)
        {
            cooley_tukey_dif_wtwiddles_iter__(x, T, thread, M, P);
            M <<= 1;
            P >>= 1;
        }

        for(int itr = iters_no_sync; itr < params::log2n; ++itr)
        {
            __syncthreads();
            cooley_tukey_dif_wtwiddles_iter__(x, T, thread, M, P);
            M <<= 1;
            P >>= 1;
        }
    }
    else
    {
        for(int itr = 0; itr < params::log2n; ++itr)
        {
            cooley_tukey_dif_wtwiddles_iter__(x, T, thread, M, P);
            M <<= 1;
            P >>= 1;
        }
    }
}

__device__ void reorder1(hipDoubleComplex* x, int p, int n)
{
    int q = __brev(p) >> (32 - n);
    if(p > q)
    {
        hipDoubleComplex t = x[p];
        x[p]               = x[q];
        x[q]               = t;
    }
}

__device__ void reorder(hipDoubleComplex* x, int i, int N, int log2n)
{
    if(i >= N / 2)
        return;

    reorder1(x, i, log2n);
    reorder1(x, i + N / 2, log2n);
}

__device__ void copy_and_reorder(
    hipDoubleComplex* x, hipDoubleComplex* x_, int i, int N, int offset, int tstride, int log2n)
{
    int p, q;

    p    = i;
    q    = __brev(p) >> (32 - log2n);
    x[p] = x_[offset + q * tstride];

    p    = i + N / 2;
    q    = __brev(p) >> (32 - log2n);
    x[p] = x_[offset + q * tstride];
}

__global__ void __launch_bounds__(1024)
    cooley_tukey_dif(hipDoubleComplex* x_, int N, int log2N, dim3 bstrides, int tstride)
{
    __shared__ hipDoubleComplex x[4096];

    int offset
        = hipBlockIdx_x * bstrides.x + hipBlockIdx_y * bstrides.y + hipBlockIdx_z * bstrides.z;
    int thread = hipThreadIdx_x;

    x[thread]         = x_[offset + thread * tstride];
    x[thread + N / 2] = x_[offset + (thread + N / 2) * tstride];
    __syncthreads();

    reorder(x, thread, N, log2N);
    __syncthreads();

    cooley_tukey_dif__(x, thread, N);
    __syncthreads();

    x_[offset + thread * tstride]           = x[thread];
    x_[offset + (thread + N / 2) * tstride] = x[thread + N / 2];
}

template <class params>
__global__ void __launch_bounds__(params::threads) cooley_tukey_dif_wtwiddles(hipDoubleComplex* x_,
                                                                              hipDoubleComplex* T,
                                                                              dim3 bstrides,
                                                                              int  tstride)

{
    __shared__ hipDoubleComplex x[params::shared_memory];

    int offset
        = hipBlockIdx_x * bstrides.x + hipBlockIdx_y * bstrides.y + hipBlockIdx_z * bstrides.z;
    int thread = hipThreadIdx_x;

    if(thread >= params::n / 2)
        return;

    x[thread]                 = x_[offset + thread * tstride];
    x[thread + params::n / 2] = x_[offset + (thread + params::n / 2) * tstride];
    __syncthreads();

    reorder(x, thread, params::n, params::log2n);
    __syncthreads();

    cooley_tukey_dif_wtwiddles__<params>(x, T, thread);
    __syncthreads();

    x_[offset + thread * tstride]                   = x[thread];
    x_[offset + (thread + params::n / 2) * tstride] = x[thread + params::n / 2];
}

__global__ void cooley_tukey_twiddles(hipDoubleComplex* T, int N)
{
    int m = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(m >= N)
        return;

    double cost, sint;
    sincospi(-double(m) / N, &sint, &cost);
    T[m].x = cost;
    T[m].y = sint;
}

gpu_result fft_gpu_ct_dif(vector<fftw_complex> const& x, int nx, int nbatch)
{
    auto z = copy(x);

    hipDoubleComplex* X;
    HIP_CHECK(hipMalloc(&X, nx * nbatch * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, z.data(), nx * nbatch * sizeof(fftw_complex), hipMemcpyHostToDevice));

    hipDoubleComplex* T;
    HIP_CHECK(hipMalloc(&T, nx / 2 * sizeof(fftw_complex)));

    GPUTimer timer;
    timer.tic();
    dim3 strides(nx);
    // cooley_tukey_dif<<<nbatch, nx / 2>>>(X, nx, log2(nx), strides, 1);
    cooley_tukey_twiddles<<<(nx / 2 + 255) / 256, 256>>>(T, nx / 2);
    // cooley_tukey_twiddles<<<1, nx / 2>>>(T, nx / 2);
    switch(nx)
    {
    case 4:
        cooley_tukey_dif_wtwiddles<CT4><<<nbatch, CT4::threads>>>(X, T, strides, 1);
        break;
    case 8:
        cooley_tukey_dif_wtwiddles<CT8><<<nbatch, CT8::threads>>>(X, T, strides, 1);
        break;
    case 16:
        cooley_tukey_dif_wtwiddles<CT16><<<nbatch, CT16::threads>>>(X, T, strides, 1);
        break;
    case 32:
        cooley_tukey_dif_wtwiddles<CT32><<<nbatch, CT32::threads>>>(X, T, strides, 1);
        break;
    case 64:
        cooley_tukey_dif_wtwiddles<CT64><<<nbatch, CT64::threads>>>(X, T, strides, 1);
        break;
    case 128:
        cooley_tukey_dif_wtwiddles<CT128><<<nbatch, CT128::threads>>>(X, T, strides, 1);
        break;
    case 256:
        cooley_tukey_dif_wtwiddles<CT256><<<nbatch, CT256::threads>>>(X, T, strides, 1);
        break;
    case 512:
        cooley_tukey_dif_wtwiddles<CT512><<<nbatch, CT512::threads>>>(X, T, strides, 1);
        break;
    case 1024:
        cooley_tukey_dif_wtwiddles<CT1024><<<nbatch, CT1024::threads>>>(X, T, strides, 1);
        break;
    case 2048:
        cooley_tukey_dif_wtwiddles<CT2048><<<nbatch, CT2048::threads>>>(X, T, strides, 1);
        break;
    }
    timer.toc();

    HIP_CHECK(hipMemcpy(z.data(), X, nx * nbatch * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(T));
    HIP_CHECK(hipFree(X));

    return {timer.elapsed(), move(z)};
}

gpu_result fft_gpu_ct_dif_2d(vector<fftw_complex> const& x, int nx, int ny)
{
    auto const N = x.size();

    auto z = copy(x);

    hipDoubleComplex* X;
    HIP_CHECK(hipMalloc(&X, N * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, z.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice));

    GPUTimer timer;
    timer.tic();
    cooley_tukey_dif<<<nx, ny / 2>>>(X, ny, log2(ny), dim3(ny), 1);
    cooley_tukey_dif<<<ny, nx / 2>>>(X, nx, log2(nx), dim3(1), ny);
    timer.toc();

    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(X));

    return {timer.elapsed(), move(z)};
}

gpu_result fft_gpu_ct_dif_3d(vector<fftw_complex> const& x, int nx, int ny, int nz)
{
    auto const N = x.size();

    auto z = copy(x);

    hipDoubleComplex* X;
    HIP_CHECK(hipMalloc(&X, N * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, z.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice));

    GPUTimer timer;
    timer.tic();
    cooley_tukey_dif<<<dim3(nx, ny), nz / 2>>>(X, nz, log2(nz), dim3(ny * nz, nz), 1);
    cooley_tukey_dif<<<dim3(nx, nz), ny / 2>>>(X, ny, log2(ny), dim3(ny * nz, 1), nz);
    cooley_tukey_dif<<<dim3(ny, nz), nx / 2>>>(X, nx, log2(nx), dim3(nz, 1), ny * nz);
    timer.toc();

    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(X));

    return {timer.elapsed(), move(z)};
}

//
// Relative difference
//
double compare(vector<fftw_complex> const& z1, vector<fftw_complex> const& z2)
{
    double d = 0.0;
    double r = 0.0;
    for(size_t n = 0; n < z1.size(); ++n)
    {
        double dx = z1[n][0] - z2[n][0];
        double dy = z1[n][1] - z2[n][1];
        // if(dx * dx + dy * dy > 1.e-7)
        // {
        //     cout << n << " " << sqrt(dx * dx + dy * dy) << " " << z1[n][0] << " " << z2[n][0]
        //          << endl;
        // }
        d += dx * dx + dy * dy;
        r += z1[n][0] * z1[n][0] + z1[n][1] * z1[n][1];
    }
    return sqrt(d) / sqrt(r);
}

//
// Some tests!
//
void test1d(size_t n, size_t nbatch)
{
    CPUTimer timer;

    double GiB = double(n * nbatch * 16) / 1024 / 1024 / 1024;
    cout << "# 1d test" << endl;
    cout << "1d input length: " << n << " (" << nbatch << ")" << endl;
    cout << "1d input size:   " << GiB << "GiB" << endl;

    auto x = random_vector(n * nbatch);

    timer.tic();
    auto z1 = fft_fftw(x, n, nbatch);
    timer.toc();
    cout << "FFTW time:       " << timer.elapsed() << "ms" << endl;

    auto [c4, z4] = fft_gpu_ct_dif(x, n, nbatch);
    auto [c5, z5] = fft_gpu_ct_dif(x, n, nbatch);
    auto [c6, z6] = fft_gpu_ct_dif(x, n, nbatch);
    cout << "GPU rel diff:    " << compare(z1, z4) << endl;
    cout << "GPU kernel time: " << c6 << "ms" << endl;
    cout << "GPU throughput:  " << GiB * 1000 / c6 << " GiB/s" << endl;
}

void test2d()
{
    size_t const n = (size_t)pow(2, 10);
    auto         x = random_vector(n * n);

    cout << "# 2d test" << endl;
    cout << "2d input length: " << n << "x" << n << endl;

    CPUTimer timer;

    timer.tic();
    auto z1 = fft_fftw_2d(x, n, n);
    timer.toc();
    cout << "FFTW time:       " << timer.elapsed() << "ms" << endl;

    auto [c2, z2] = fft_gpu_ct_dif_2d(x, n, n);
    cout << "GPU rel diff:    " << compare(z1, z2) << endl;
    cout << "GPU kernel time: " << c2 << "ms" << endl;
}

void test3d()
{
    size_t const n = (size_t)pow(2, 9);
    auto         x = random_vector(n * n * n);

    cout << "# 3d text" << endl;
    cout << "3d input length: " << n << "x" << n << "x" << n << endl;

    CPUTimer timer;

    timer.tic();
    auto z1 = fft_fftw_3d(x, n, n, n);
    timer.toc();
    cout << "FFTW time:       " << timer.elapsed() << "ms" << endl;

    auto [c2, z2] = fft_gpu_ct_dif_3d(x, n, n, n);
    cout << "GPU rel diff:    " << compare(z1, z2) << endl;
    cout << "GPU kernel:      " << c2 << "ms" << endl;
}

int main(int argc, char* argv[])
{
    size_t length = 2048;
    size_t nbatch = 1;
    if(argc > 1)
        length = stoi(argv[1]);
    if(argc > 2)
        nbatch = stoi(argv[2]);

    test1d(length, nbatch);
    //test2d();
    //test3d();
}

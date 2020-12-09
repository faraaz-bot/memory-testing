//
// Cooley-Tukey playground.
//
// Simple 1d, complex to complex, length 2048, variable batch.
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
using dtype      = hipDoubleComplex;
using fft_result = pair<float, vector<dtype>>;

//
// Random inputs
//
vector<dtype> random_vector(size_t n)
{
    vector<dtype>                     x(n);
    random_device                     rd;
    mt19937                           gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
    for(size_t i = 0; i < n; ++i)
    {
        x[i].x = dis(gen);
        x[i].y = dis(gen);
    }
    return x;
}

//
// Copy helper for dtype (used to be helpful for fftw_complex)...
//
template <typename T>
vector<T> copy(vector<T> const& x)
{
    vector<T> z(x.size());
    for(size_t i = 0; i < x.size(); ++i)
    {
        z[i] = x[i];
    }
    return z;
}

//
// FFTW backed FFT
//
pair<float, vector<hipDoubleComplex>>
    fft_fftw(vector<hipDoubleComplex> const& x, int nx, int nbatch)
{
    auto z = copy(x);
    // clang-format off
    auto p = fftw_plan_many_dft(1, &nx, nbatch,
                                (fftw_complex*) z.data(), nullptr, 1, nx,
                                (fftw_complex*) z.data(), nullptr, 1, nx,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    CPUTimer timer;
    timer.tic();
    fftw_execute(p);
    timer.toc();
    fftw_destroy_plan(p);
    return {timer.elapsed(), move(z)};
}

pair<float, vector<hipComplex>> fft_fftw(vector<hipComplex> const& x, int nx, int nbatch)
{
    auto z = copy(x);
    // clang-format off
    auto p = fftwf_plan_many_dft(1, &nx, nbatch,
                                (fftwf_complex*) z.data(), nullptr, 1, nx,
                                (fftwf_complex*) z.data(), nullptr, 1, nx,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    CPUTimer timer;
    timer.tic();
    fftwf_execute(p);
    timer.toc();
    fftwf_destroy_plan(p);
    return {timer.elapsed(), move(z)};
}

//
// Cooley-Tukey, re-order first
//
// The power-2 Cooley-Tukey algorithm updates two elements at a time.
// Therefore we only need to launch N/2 threads.
//
// During the p'th iteration of Cooley-Tukey, we update N/2 pairs of
// elements; denoted by indicies (i, j).  The i indicies are traversed
// in chunks of length M=2^p elements; and j is simply i + M.  The i
// indicies skip over odd chunks (those are the j elements).
//
// To go from a thread index to a Cooley-Tukey pair during the p'th
// iteration, we decompose the thread index into: which chunk we're
// in; and which element within that chunk we're at.  This is done
// using:
//
//  int i = ((M - 1) & thread) + ((~(M - 1) & thread) << 1);
//          ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
//             within chunk            chunk; skip odd
//

template <class params>
__device__ void
    cooley_tukey_dif_wtwiddles_shuffle__(dtype* __restrict__ x, dtype* __restrict__ T, int thread)
{
    int M = 1;
    int P = params::n >> 1;

    int   i, j, m;
    dtype t, z1, z2, d1, d2;

    constexpr int iters_no_sync = params::log2n > 6 ? 6 : params::log2n;

    // first iteration; elements (z1 and z2) interact
    d1 = x[thread * 2];
    d2 = x[thread * 2 + 1];
    z1 = d1 + d2;
    z2 = d1 - d2;
    M <<= 1;
    P >>= 1;

    // subsequent iterations up to wavefront; elements don't interact; no sync
    for(int itr = 1; itr < iters_no_sync; ++itr)
    {
        int lane  = thread % 64;
        int mask  = 1 << (itr - 1);
        int onoff = (lane & mask) >> (itr - 1);
        int pm    = 1 - 2 * onoff;
        int vmask = onoff * mask;
        int dmask = mask ^ vmask;

        int root = (2 * lane * P) % (params::n / 2);

        t    = T[root];
        d1.x = t.x * z1.x - t.y * z1.y;
        d1.y = t.y * z1.x + t.x * z1.y;
        t    = T[root + P];
        d2.x = t.x * z2.x - t.y * z2.y;
        d2.y = t.y * z2.x + t.x * z2.y;

        z1.x = __shfl_xor(z1.x, vmask) + pm * __shfl_xor(d1.x, dmask);
        z1.y = __shfl_xor(z1.y, vmask) + pm * __shfl_xor(d1.y, dmask);
        z2.x = __shfl_xor(z2.x, vmask) + pm * __shfl_xor(d2.x, dmask);
        z2.y = __shfl_xor(z2.y, vmask) + pm * __shfl_xor(d2.y, dmask);

        M <<= 1;
        P >>= 1;
    }

    x[thread * 2]     = z1;
    x[thread * 2 + 1] = z2;

    // remaining iterations; out of wavefront, use local storage and sync
    for(int itr = iters_no_sync; itr < params::log2n; ++itr)
    {
        __syncthreads();

        i = ((M - 1) & thread) + ((~(M - 1) & thread) << 1);
        j = i + M;
        m = i % M;

        z1 = x[i];
        z2 = x[j];

        t    = T[m * P];
        d1.x = t.x * z2.x - t.y * z2.y;
        d1.y = t.y * z2.x + t.x * z2.y;

        x[i] = z1 + d1;
        x[j] = z1 - d1;

        M <<= 1;
        P >>= 1;
    }
}

__device__ void reorder1(dtype* x, int p, int n)
{
    int q = __brev(p) >> (32 - n);
    if(p > q)
    {
        dtype t = x[p];
        x[p]    = x[q];
        x[q]    = t;
    }
}

__device__ void reorder(dtype* x, int i, int N, int log2n)
{
    reorder1(x, i, log2n);
    reorder1(x, i + N / 2, log2n);
}

template <class params>
__global__ void __launch_bounds__(params::threads) cooley_tukey_dif_wtwiddles(dtype* x_, dtype* T)

{
    __shared__ dtype x[params::shared_memory];

    int offset = hipBlockIdx_x * hipBlockDim_x * 2;
    int thread = hipThreadIdx_x;

    if(thread >= params::n / 2)
        return;

    x[thread]                 = x_[offset + thread];
    x[thread + params::n / 2] = x_[offset + (thread + params::n / 2)];
    __syncthreads();

    reorder(x, thread, params::n, params::log2n);
    __syncthreads();

    cooley_tukey_dif_wtwiddles_shuffle__<params>(x, T, thread);
    __syncthreads();

    x_[offset + thread]                   = x[thread];
    x_[offset + (thread + params::n / 2)] = x[thread + params::n / 2];
}

__global__ void cooley_tukey_twiddles(dtype* T, int N)
{
    int m = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(m >= N)
        return;

    double cost, sint;
    sincospi(-double(m) / N, &sint, &cost);
    T[m].x = cost;
    T[m].y = sint;
}

void cooley_tukey_cpu_twiddles(dtype* T, int N)
{
    for(int m = 0; m < N; ++m)
    {
        T[m].x = cos(-double(m) / N * M_PI);
        T[m].y = sin(-double(m) / N * M_PI);
    }
}

fft_result fft_gpu_ct_dif(vector<dtype> const& x, int nx, int nbatch)
{
    auto z = copy(x);

    dtype* X;
    HIP_CHECK(hipMalloc(&X, nx * nbatch * sizeof(dtype)));
    HIP_CHECK(hipMemcpy(X, z.data(), nx * nbatch * sizeof(dtype), hipMemcpyHostToDevice));

    dtype* T;
    HIP_CHECK(hipMalloc(&T, nx / 2 * sizeof(dtype)));
    //    cooley_tukey_twiddles<<<(nx / 2 + 255) / 256, 256>>>(T, nx / 2);

    vector<dtype> t(nx / 2);
    cooley_tukey_cpu_twiddles(t.data(), nx / 2);
    HIP_CHECK(hipMemcpy(T, t.data(), nx / 2 * sizeof(dtype), hipMemcpyHostToDevice));

    GPUTimer timer;
    timer.tic();
    cooley_tukey_dif_wtwiddles<CT2048><<<nbatch, CT2048::threads>>>(X, T);
    timer.toc();

    HIP_CHECK(hipMemcpy(z.data(), X, nx * nbatch * sizeof(dtype), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(T));
    HIP_CHECK(hipFree(X));

    return {timer.elapsed(), move(z)};
}

//
// Relative difference
//
double compare(vector<dtype> const& z1, vector<dtype> const& z2)
{
    double d = 0.0;
    double r = 0.0;
    for(size_t n = 0; n < z1.size(); ++n)
    {
        double dx = z1[n].x - z2[n].x;
        double dy = z1[n].y - z2[n].y;
        // if(dx * dx + dy * dy > 1.e-7)
        // {
        //     cout << n << " " << sqrt(dx * dx + dy * dy) << " " << z1[n][0] << " " << z2[n][0]
        //          << endl;
        // }
        d += dx * dx + dy * dy;
        r += z1[n].x * z1[n].x + z1[n].y * z1[n].y;
    }
    return sqrt(d) / sqrt(r);
}

//
// Some tests!
//
void test1d(size_t n, size_t nbatch)
{
    double GiB = double(n * nbatch * sizeof(dtype)) / 1024 / 1024 / 1024;
    cout << "# 1d test" << endl;
    cout << "1d input length: " << n << " (" << nbatch << ")" << endl;
    cout << "1d input size:   " << GiB << "GiB" << endl;

    auto x = random_vector(n * nbatch);

    auto [t1, z1] = fft_fftw(x, n, nbatch);
    cout << "FFTW time:       " << t1 << "ms" << endl;

    auto [t2, z2] = fft_gpu_ct_dif(x, n, nbatch);
    cout << "GPU rel diff:    " << compare(z1, z2) << endl;
    cout << "GPU kernel time: " << t2 << "ms" << endl;
    cout << "GPU throughput:  " << GiB * 1000 / t2 << " GiB/s" << endl;
}

int main(int argc, char* argv[])
{
    size_t length = 2048;
    size_t nbatch = 1;
    if(argc > 1)
        nbatch = stoi(argv[1]);

    test1d(length, nbatch);
}

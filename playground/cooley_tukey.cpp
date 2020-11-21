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
// Cooley-Tukey FFT on the GPU
//
//
// Recall
//
//    X(k) = sum(n=0..N-1) x(n) w(n k, N)
//
// where
//
//    w(k, N) = exp(-2 pi i k / N).
//
// We want to decompose this assuming that N is a power of 2.  First,
// break the sum into two (think we could also do this in four too).
// Essentially we break the n-index into n = n1 + n2 N/2
//
//    X(k) = sum(n1=0..N/2-1) sum(n2=0,1) x(n1 + n2 N/2) w((n1 + n2 N/2) k, N)
//         = sum(n1=0..N/2-1) sum(n2=0,1) x(n1 + n2 N/2) w((n1 + n2 N/2) k, N)
//         = sum(n1=0..N/2-1) w(n1 k, N) sum(n2=0,1) x(n1 + n2 N/2) w(n2 N k / 2, N)
//         = sum(n1=0..N/2-1) w(n1 k, N) [ x(n1) + (-1)^k x(n1+N/2) ]
//
// For even/odd k, we obtain
//
//    X(2k) = sum(n1=0..N/2-1) w(n1 k, N / 2) [ x(n1) + x(n1+N/2) ]
//    X(2k+1) = sum(n1=0..N/2-1) w(n1 k, N / 2) w(n1, N) [ x(n1) - x(n1+N/2) ]
//
// Note that these look like N/2 DFTs; so could recurse.  If we store
// cleverly, can re-write as iterative and do everything in-place.
// This is the idea behind Cooley-Tukey.
//

//
// Cooley-Tukey, re-order first, FFT on the GPU
//
// This version does a single element across all iterations;
// bit-reverse done first.
//
__device__ void cooley_tukey_dif__(hipDoubleComplex* x, int i0, int N)
{
    // note: i0 in [0, N/2]
    if(i0 >= N / 2)
        return;

    int M = 1; // size of current block

    while(M < N)
    {
        if(M > 64)
            __syncthreads();

        // convert i0 to i in [0, N]; skip odd blocks
        int i = ((M - 1) & i0) + ((~(M - 1) & i0) << 1);
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

        M <<= 1;
    }
}

template <bool sync>
__device__ void cooley_tukey_dif_wtwiddles_iter__(
    hipDoubleComplex* x, hipDoubleComplex* T, int i0, int M, int P)
{

    if constexpr(sync)
        __syncthreads();

    // convert i0 to i in [0, N]; skip odd blocks
    int i = ((M - 1) & i0) + ((~(M - 1) & i0) << 1);
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

__device__ void
    cooley_tukey_dif_wtwiddles__(hipDoubleComplex* x, hipDoubleComplex* T, int i0, int N)
{
    cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, 1, N / 2);

#define HELP_ME_UNDERSTAND
#ifdef HELP_ME_UNDERSTAND
    cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, 2, N / 4);
    cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, 4, N / 8);
    cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, 8, N / 16);
    cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, 16, N / 32);
    cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, 32, N / 64);
    cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, 64, N / 128);
    int P = N / 256;
    int M = 128;
#else
    int P = N / 4;
    int M = 2;
#endif

    while(M < N)
    {
        // if(M > 64)
        cooley_tukey_dif_wtwiddles_iter__<true>(x, T, i0, M, P);
        // else
        //     cooley_tukey_dif_wtwiddles_iter__<false>(x, T, i0, M, P, clks);
        M <<= 1;
        P >>= 1;
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

__global__ void cooley_tukey_dif(hipDoubleComplex* x_, int N, int log2N, dim3 bstrides, int tstride)
{
    __shared__ hipDoubleComplex x[2048];

    int offset
        = hipBlockIdx_x * bstrides.x + hipBlockIdx_y * bstrides.y + hipBlockIdx_z * bstrides.z;
    int i = hipThreadIdx_x;

    x[i]         = x_[offset + i * tstride];
    x[i + N / 2] = x_[offset + (i + N / 2) * tstride];
    __syncthreads();

    reorder(x, i, N, log2N);
    __syncthreads();

    cooley_tukey_dif__(x, i, N);
    __syncthreads();

    x_[offset + i * tstride]           = x[i];
    x_[offset + (i + N / 2) * tstride] = x[i + N / 2];
}

__global__ void cooley_tukey_dif_wtwiddles(
    hipDoubleComplex* x_, hipDoubleComplex* T, int N, int log2N, dim3 bstrides, int tstride)

{
    __shared__ hipDoubleComplex x[2048];

    int offset
        = hipBlockIdx_x * bstrides.x + hipBlockIdx_y * bstrides.y + hipBlockIdx_z * bstrides.z;
    int i = hipThreadIdx_x;

    if(i > N / 2)
        return;

        //#define USE_COMBINED_REORDER
#ifdef USE_COMBINED_REORDER
    copy_and_reorder(x, x_, i, N, offset, tstride, log2N);
    __syncthreads();
#else
    x[i]         = x_[offset + i * tstride];
    x[i + N / 2] = x_[offset + (i + N / 2) * tstride];
    __syncthreads();

    reorder(x, i, N, log2N);
    __syncthreads();
#endif

    cooley_tukey_dif_wtwiddles__(x, T, i, N);

    __syncthreads();
    x_[offset + i * tstride]           = x[i];
    x_[offset + (i + N / 2) * tstride] = x[i + N / 2];
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
    cooley_tukey_twiddles<<<(nx / 2 + 255) / 256, 256>>>(T, nx / 2);
    cooley_tukey_dif_wtwiddles<<<nbatch, nx / 2>>>(X, T, nx, log2(nx), strides, 1);
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
        // if (dx * dx + dy * dy > 1.e-7) {
        //   cout << n << " " << sqrt(dx * dx + dy * dy) << " " << z1[n][0] << " " << z2[n][0] << endl;
        // }
        d += dx * dx + dy * dy;
        r += z1[n][0] * z1[n][0] + z1[n][1] * z1[n][1];
    }
    return sqrt(d) / sqrt(r);
}

//
// Some tests!
//
void test1d(size_t n)
{
    //    size_t const nbatch = 65536;
    //size_t const nbatch = 16384;
    size_t const nbatch = 1;
    auto         x      = random_vector(n * nbatch);

    CPUTimer timer;

    cout << "# 1d test" << endl;
    cout << "1d input length: " << n << " (" << nbatch << ")" << endl;

    timer.tic();
    auto z1 = fft_fftw(x, n, nbatch);
    timer.toc();
    cout << "FFTW time:       " << timer.elapsed() << "ms" << endl;

    auto [c4, z4] = fft_gpu_ct_dif(x, n, nbatch);
    cout << "GPU rel diff:    " << compare(z1, z4) << endl;
    cout << "GPU kernel time: " << c4 << "ms" << endl;
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
    size_t n = 2048;
    if(argc > 1)
    {
        n = stoi(argv[1]);
    }

    test1d(n);
    //test2d();
    //test3d();
}

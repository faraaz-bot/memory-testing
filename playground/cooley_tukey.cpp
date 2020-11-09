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

#define PI 3.141592653589793238462643383279502884L
#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return {};

#ifdef USE_CLOCK_TIMERS
#define TIC(x) clks.tic(x)
#define TOC(x) clks.toc(x)
#else
#define TIC(x)
#define TOC(x)
#endif

using namespace std;
using gpu_result = pair<float, vector<fftw_complex>>;

enum CooleyTukeyClocks
{
    OUTER_PULL,
    OUTER_REORDER,
    OUTER_TRANSFORM,
    OUTER_PUSH,
    OUTER_TOTAL,
    INNER_SYNC,
    INNER_ARITH,
    INNER_PULL,
    INNER_BUTTERFLY,
    INNER_PUSH,
    INNER_TOTAL,
    SIZE
};

using CTTimer = ClockTimer<CooleyTukeyClocks>;

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
// Naive DFT
//
vector<fftw_complex> fft_naive(vector<fftw_complex> const& x)
{
    auto const N = x.size();

    vector<fftw_complex> z(N);
    for(size_t k = 0; k < N; ++k)
    {
        for(size_t n = 0; n < N; ++n)
        {
            double theta = -2 * PI * n * k / N;
            z[k][0] += cos(theta) * x[n][0] - sin(theta) * x[n][1];
            z[k][1] += cos(theta) * x[n][1] + sin(theta) * x[n][0];
        }
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

vector<fftw_complex> fft_ict(vector<fftw_complex> const& x)
{
    auto const N     = x.size();
    auto const log2N = (size_t)log2(N);

    auto z = copy(x);
    for(size_t s = 0; s < log2N; ++s)
    {
        auto a     = (size_t)pow(2, s);
        auto b     = N / a;
        auto halfb = b / 2;
        for(size_t l = 0; l < halfb; ++l)
        {
            for(size_t k = 0; k < a; ++k)
            {
                auto         p     = l + k * b;
                auto         q     = p + halfb;
                double       theta = -2 * PI * l / b;
                fftw_complex xp    = {z[p][0], z[p][1]};
                fftw_complex xq    = {z[q][0], z[q][1]};
                z[p][0]            = xp[0] + xq[0];
                z[p][1]            = xp[1] + xq[1];
                z[q][0]            = cos(theta) * (xp[0] - xq[0]) - sin(theta) * (xp[1] - xq[1]);
                z[q][1]            = cos(theta) * (xp[1] - xq[1]) + sin(theta) * (xp[0] - xq[0]);
            }
        }
    }

    for(size_t p = 0; p < N; ++p)
    {
        auto q = bitreverse(p, log2N);
        if(p > q)
            swap(z[p], z[q]);
    }

    return z;
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

//#define AVOID_BANK_CONFLICTS

#ifdef AVOID_BANK_CONFLICTS
// 256
#define LCLMASK 255
#define BNK(i) (((i) >> 8) * 257 + (i & LCLMASK))
// 128
// #define LCLMASK 127
// #define BNK(i) (((i)>>7)*129+(i&LCLMASK))
// 64
// #define LCLMASK 63
// #define BNK(i) (((i)>>6)*65+(i&LCLMASK))
// 32
// #define LCLMASK 31
// #define BNK(i) (((i) >> 5) * 33 + (i & LCLMASK))
#else
#define BNK(i) (i)
#endif

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

        hipDoubleComplex xi = x[BNK(i)];
        hipDoubleComplex xj = x[BNK(j)];
        hipDoubleComplex d;

        d.x = cost * xj.x - sint * xj.y;
        d.y = sint * xj.x + cost * xj.y;

        x[BNK(i)] = xi + d;
        x[BNK(j)] = xi - d;

        M <<= 1;
    }
}

__device__ void cooley_tukey_dif_wtwiddles__(
    hipDoubleComplex* x, hipDoubleComplex* T, int i0, int N, CTTimer& clks)
{
    TIC(INNER_TOTAL);

    int P = N >> 1;

    for(int M = 1; M < N; M <<= 1)
    {
        TIC(INNER_SYNC);
        if(M > 64)
            __syncthreads();
        TOC(INNER_SYNC);

        // convert i0 to i in [0, N]; skip odd blocks
        TIC(INNER_ARITH);
        int i = ((M - 1) & i0) + ((~(M - 1) & i0) << 1);
        int j = i + M;
        int m = i % M;
        TOC(INNER_ARITH);

        TIC(INNER_PULL);
        hipDoubleComplex t  = T[m * P];
        hipDoubleComplex xi = x[BNK(i)];
        hipDoubleComplex xj = x[BNK(j)];
        hipDoubleComplex d;
        TOC(INNER_PULL);

        TIC(INNER_BUTTERFLY);
        d.x = t.x * xj.x - t.y * xj.y;
        d.y = t.y * xj.x + t.x * xj.y;
        TOC(INNER_BUTTERFLY);

        TIC(INNER_PUSH);
        x[BNK(i)] = xi + d;
        x[BNK(j)] = xi - d;
        TOC(INNER_PUSH);

        P >>= 1;
    }

    TOC(INNER_TOTAL);
}

__device__ void reorder1(hipDoubleComplex* x, int p, int n)
{
    int q = __brev(p) >> (32 - n);
    if(p > q)
    {
        hipDoubleComplex t = x[BNK(p)];
        x[BNK(p)]          = x[BNK(q)];
        x[BNK(q)]          = t;
    }
}

__device__ void reorder(hipDoubleComplex* x, int i, int N, int log2n)
{
    if(i >= N / 2)
        return;

    reorder1(x, i, log2n);
    reorder1(x, i + N / 2, log2n);
}

__global__ void cooley_tukey_dif(
    hipDoubleComplex* x_, int N, int log2N, dim3 bstrides, int tstride, CTTimer* clksbuf)
{
    __shared__ hipDoubleComplex x[2048];

    int offset
        = hipBlockIdx_x * bstrides.x + hipBlockIdx_y * bstrides.y + hipBlockIdx_z * bstrides.z;
    int i = hipThreadIdx_x;

    CTTimer clks;
    if(offset == 0 && i == 0)
        clks.accumulate = true;

    TIC(OUTER_TOTAL);

    // bool const set_clock = false;
    TIC(OUTER_PULL);
    x[BNK(i)]         = x_[offset + i * tstride];
    x[BNK(i + N / 2)] = x_[offset + (i + N / 2) * tstride];
    __syncthreads();
    TOC(OUTER_PULL);

    TIC(OUTER_REORDER);
    reorder(x, i, N, log2N);
    __syncthreads();
    TOC(OUTER_REORDER);

    TIC(OUTER_TRANSFORM);
    cooley_tukey_dif__(x, i, N);
    __syncthreads();
    TOC(OUTER_TRANSFORM);

    TIC(OUTER_PUSH);
    x_[offset + i * tstride]           = x[BNK(i)];
    x_[offset + (i + N / 2) * tstride] = x[BNK(i + N / 2)];
    TOC(OUTER_PUSH);

    TOC(OUTER_TOTAL);

    *clksbuf = clks;
}

__global__ void cooley_tukey_dif_wtwiddles(hipDoubleComplex* x_,
                                           hipDoubleComplex* T,
                                           int               N,
                                           int               log2N,
                                           dim3              bstrides,
                                           int               tstride,
                                           CTTimer*          clksbuf)
{
    __shared__ hipDoubleComplex x[2048];

    int offset
        = hipBlockIdx_x * bstrides.x + hipBlockIdx_y * bstrides.y + hipBlockIdx_z * bstrides.z;
    int i = hipThreadIdx_x;

    if(i > N / 2)
        return;

    CTTimer clks;
    if(offset == 0 && i == 0)
        clks.accumulate = true;

    TIC(OUTER_TOTAL);

    TIC(OUTER_PULL);
    x[BNK(i)]         = x_[offset + i * tstride];
    x[BNK(i + N / 2)] = x_[offset + (i + N / 2) * tstride];
    __syncthreads();
    TOC(OUTER_PULL);

    TIC(OUTER_REORDER);
    reorder(x, i, N, log2N);
    __syncthreads();
    TOC(OUTER_REORDER);

    TIC(OUTER_TRANSFORM);
    cooley_tukey_dif_wtwiddles__(x, T, i, N, clks);
    TOC(OUTER_TRANSFORM);

    TIC(OUTER_PUSH);
    __syncthreads();
    x_[offset + i * tstride]           = x[BNK(i)];
    x_[offset + (i + N / 2) * tstride] = x[BNK(i + N / 2)];
    TOC(OUTER_PUSH);

    TOC(OUTER_TOTAL);

    if(clks.accumulate)
        *clksbuf = clks;
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

    CTTimer* d_clocks;
    HIP_CHECK(hipMalloc(&d_clocks, sizeof(CTTimer)));

    GPUTimer timer;
    timer.tic();
    dim3 strides(nx);
    cooley_tukey_twiddles<<<(nx / 2 + 255) / 256, 256>>>(T, nx / 2);
    cooley_tukey_dif_wtwiddles<<<nbatch, nx / 2>>>(X, T, nx, log2(nx), strides, 1, d_clocks);
    timer.toc();

    CTTimer clocks;
    HIP_CHECK(hipMemcpy(&clocks, d_clocks, sizeof(CTTimer), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(z.data(), X, nx * nbatch * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_clocks));
    HIP_CHECK(hipFree(T));
    HIP_CHECK(hipFree(X));

#ifdef USE_CLOCK_TIMERS
    cout << "OUTER_PULL      " << clocks.total[OUTER_PULL] << endl;
    cout << "OUTER_REORDER   " << clocks.total[OUTER_REORDER] << endl;
    cout << "OUTER_TRANSFORM " << clocks.total[OUTER_TRANSFORM] << endl;
    cout << "OUTER_PUSH      " << clocks.total[OUTER_PUSH] << endl;
    cout << "OUTER_TOTAL     " << clocks.total[OUTER_TOTAL] << endl;
    cout << "INNER_SYNC      " << clocks.total[INNER_SYNC] << endl;
    cout << "INNER_ARITH     " << clocks.total[INNER_ARITH] << endl;
    cout << "INNER_PULL      " << clocks.total[INNER_PULL] << endl;
    cout << "INNER_BUTTERFLY " << clocks.total[INNER_BUTTERFLY] << endl;
    cout << "INNER_PUSH      " << clocks.total[INNER_PUSH] << endl;
    cout << "INNER_TOTAL     " << clocks.total[INNER_TOTAL] << endl;
#endif

    return {timer.elapsed(), move(z)};
}

gpu_result fft_gpu_ct_dif_2d(vector<fftw_complex> const& x, int nx, int ny)
{
    auto const N = x.size();

    auto z = copy(x);

    hipDoubleComplex* X;
    HIP_CHECK(hipMalloc(&X, N * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, z.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice));

    CTTimer* d_clocks;
    HIP_CHECK(hipMalloc(&d_clocks, sizeof(CTTimer)));

    GPUTimer timer;
    timer.tic();
    cooley_tukey_dif<<<nx, ny / 2>>>(X, ny, log2(ny), dim3(ny), 1, d_clocks);
    cooley_tukey_dif<<<ny, nx / 2>>>(X, nx, log2(nx), dim3(1), ny, d_clocks);
    timer.toc();

    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_clocks));
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

    CTTimer* d_clocks;
    HIP_CHECK(hipMalloc(&d_clocks, sizeof(CTTimer)));

    GPUTimer timer;
    timer.tic();
    cooley_tukey_dif<<<dim3(nx, ny), nz / 2>>>(X, nz, log2(nz), dim3(ny * nz, nz), 1, d_clocks);
    cooley_tukey_dif<<<dim3(nx, nz), ny / 2>>>(X, ny, log2(ny), dim3(ny * nz, 1), nz, d_clocks);
    cooley_tukey_dif<<<dim3(ny, nz), nx / 2>>>(X, nx, log2(nx), dim3(nz, 1), ny * nz, d_clocks);
    timer.toc();

    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_clocks));
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
void test1d()
{
    size_t const n      = (size_t)pow(2, 11);
    size_t const nbatch = 4096;
    auto         x      = random_vector(n * nbatch);

    CPUTimer timer;

    cout << "# 1d test" << endl;
    cout << "1d input length: " << n << " (" << nbatch << ")" << endl;

    timer.tic();
    auto z1 = fft_fftw(x, n, nbatch);
    timer.toc();
    cout << "FFTW time:       " << timer.elapsed() << "ms" << endl;

    // timer.tic();
    // auto z2 = fft_ict(x);
    // timer.toc();
    // cout << "ICT time:        " << timer.elapsed() << "ms" << endl;
    // cout << "ICT rel diff:    " << compare(z1, z2) << endl;

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
#ifdef AVOID_BANK_CONFLICTS
    cout << "TRYING TO AVOID BANK CONFLICTS" << endl;
#endif
    test1d();
    //test2d();
    //test3d();
}

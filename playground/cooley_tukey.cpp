//
// Cooley-Tukey playground.
//
// Simple 1d, complex to complex, power of 2.
//

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <fftw3.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#define PI 3.141592653589793238462643383279502884L
#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return {};

using namespace std;
using gpu_result = pair<size_t, vector<fftw_complex>>;

//
// Random inputs
//
vector<fftw_complex> random_vector(size_t n)
{
    vector<fftw_complex>              x(n);
    random_device                     rd;
    mt19937                           gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
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
vector<fftw_complex> fft_fftw(vector<fftw_complex> const& x)
{
    auto z = copy(x);
    auto p = fftw_plan_dft_1d(z.size(), z.data(), z.data(), FFTW_FORWARD, FFTW_ESTIMATE);
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

//
// This version does a single element across all iterations.  It
// doesn't synchronise properly so will fail for large N.
//
__global__ void cooley_tukey_dit_01(int N, hipDoubleComplex* x)
{
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i >= N)
        return;

    int P = 1; // current number of blocks; doubles every iteration
    int M = N; // size of current block; halves every iteration

    while(M > 1)
    {
        M >>= 1;
        P <<= 1;

        if((i / M) % 2 != 0)
            continue;

        int j = i + M;
        int m = i % M;

        double theta = -PI * m / M;
        double cost  = cos(theta);
        double sint  = sin(theta);

        hipDoubleComplex xi = x[i];
        hipDoubleComplex xj = x[j];

        x[i]   = xi + xj;
        x[j].x = cost * (xi.x - xj.x) - sint * (xi.y - xj.y);
        x[j].y = cost * (xi.y - xj.y) + sint * (xi.x - xj.x);
    }
}

gpu_result fft_gpu_ct_dit_01(vector<fftw_complex> const& x)
{
    auto const N     = x.size();
    auto const log2N = (size_t)log2(N);

    void* X;
    HIP_CHECK(hipMalloc(&X, N * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, x.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice));

    auto tic     = clock();
    int  threads = N;
    int  blocks  = 1;
    cooley_tukey_dit_01<<<blocks, threads>>>(N, (hipDoubleComplex*)X);
    HIP_CHECK(hipDeviceSynchronize());
    auto toc = clock();

    vector<fftw_complex> z(N);
    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(X));

    for(size_t p = 0; p < N; ++p)
    {
        auto q = bitreverse(p, log2N);
        if(p > q)
            swap(z[p], z[q]);
    }

    return {toc - tic, move(z)};
}

//
// Cooley-Tukey, re-order first, FFT on the GPU
//
// This version does a single element across all iterations;
// bit-reverse done first.
//
__device__ void cooley_tukey_dit_02_(hipDoubleComplex* x, int i0, int N)
{
    // note: i0 in [0, N/2]
    if(i0 >= N / 2)
        return;

    int P = N; // current number of blocks
    int M = 1; // size of current block

    while(M < N)
    {
        if(M > 64)
            __syncthreads();

        // convert i0 to i in [0, N]; skip odd blocks
        int i = ((M - 1) & i0) + ((~(M - 1) & i0) << 1);
        int j = i + M;
        int m = i % M;

        double theta = -PI * m / M;
        double cost  = cos(theta);
        double sint  = sin(theta);

        hipDoubleComplex xi = x[i];
        hipDoubleComplex xj = x[j];
        hipDoubleComplex d;

        d.x = cost * xj.x - sint * xj.y;
        d.y = sint * xj.x + cost * xj.y;

        x[i] = xi + d;
        x[j] = xi - d;

        M <<= 1;
        P >>= 1;
    }
}

__device__ void reorder1(hipDoubleComplex* x, int p, int n)
{
    int q = p;
    int r = 0;
    for(int i = 0; i < n; ++i)
    {
        r <<= 1;
        r |= q & 1;
        q >>= 1;
    }
    q = r;

    if(p > q)
    {
        hipDoubleComplex t = x[p];
        x[p]               = x[q];
        x[q]               = t;
    }
}

__device__ void reorder(hipDoubleComplex* x, int i, int N)
{
    if(i >= N / 2)
        return;

    int log2n = (int)log2(N);
    reorder1(x, i, log2n);
    reorder1(x, i + N / 2, log2n);
}

__global__ void cooley_tukey_dit_02(hipDoubleComplex* x_, int N, dim3 bstrides, int tstride)
{
    __shared__ hipDoubleComplex x[2048+64];

    int offset = hipBlockIdx_x * bstrides.x + hipBlockIdx_y * bstrides.y + hipBlockIdx_z * bstrides.z;
    int i = hipThreadIdx_x;

    x[i]         = x_[offset + i * tstride];
    x[i + N / 2] = x_[offset + (i + N / 2) * tstride];
    __syncthreads();
    reorder(x, i, N);
    __syncthreads();
    cooley_tukey_dit_02_(x, i, N);
    __syncthreads();
    x_[offset + i * tstride]           = x[i];
    x_[offset + (i + N / 2) * tstride] = x[i + N / 2];
}

gpu_result fft_gpu_ct_dit_02(vector<fftw_complex> const& x)
{
    auto const N = x.size();

    auto z = copy(x);

    void* X;
    HIP_CHECK(hipMalloc(&X, N * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, z.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice));

    auto tic = clock();

    dim3 strides(N);
    cooley_tukey_dit_02<<<1, N/2>>>((hipDoubleComplex*)X, N, strides, 1);
    HIP_CHECK(hipDeviceSynchronize());

    auto toc = clock();

    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(X));

    return {toc - tic, move(z)};
}

gpu_result fft_gpu_ct_dit_2d_02(vector<fftw_complex> const& x, int nx, int ny)
{
    auto const N = x.size();

    auto z = copy(x);

    void* X;
    HIP_CHECK(hipMalloc(&X, N * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, z.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice));

    auto tic = clock();
    cooley_tukey_dit_02<<<nx, ny/2>>>((hipDoubleComplex*)X, ny, dim3(ny), 1);
    cooley_tukey_dit_02<<<ny, nx/2>>>((hipDoubleComplex*)X, nx, dim3(1), ny);
    HIP_CHECK(hipDeviceSynchronize());
    auto toc = clock();

    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(X));

    return {toc - tic, move(z)};
}

gpu_result fft_gpu_ct_dit_3d_02(vector<fftw_complex> const& x, int nx, int ny, int nz)
{
    auto const N = x.size();

    auto z = copy(x);

    void* X;
    HIP_CHECK(hipMalloc(&X, N * sizeof(fftw_complex)));
    HIP_CHECK(hipMemcpy(X, z.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice));

    auto tic = clock();
    cooley_tukey_dit_02<<<dim3(nx,ny), nz/2>>>((hipDoubleComplex*)X, nz, dim3(ny*nz, nz), 1);
    cooley_tukey_dit_02<<<dim3(nx,nz), ny/2>>>((hipDoubleComplex*)X, ny, dim3(ny*nz, 1), nz);
    cooley_tukey_dit_02<<<dim3(ny,nz), nx/2>>>((hipDoubleComplex*)X, nx, dim3(nz, 1), ny*nz);
    HIP_CHECK(hipDeviceSynchronize());
    auto toc = clock();

    HIP_CHECK(hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(X));

    return {toc - tic, move(z)};
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
        d += sqrt(dx * dx + dy * dy);
        r += sqrt(z1[n][0] * z1[n][0] + z1[n][1] * z1[n][1]);
    }
    return d / r;
}

//
// Some tests!
//
void test1d()
{
    clock_t tic, toc;

    size_t const n = (size_t)pow(2, 11);
    auto         x = random_vector(n);

    cout << "1d input length: " << n << endl;

    tic     = clock();
    auto z1 = fft_fftw(x);
    toc     = clock();
    cout << "FFTW cycles: " << toc - tic << endl;

    tic     = clock();
    auto z2 = fft_ict(x);
    toc     = clock();
    cout << "ICT cycles: " << toc - tic << endl;
    cout << "ICT rel diff " << compare(z1, z2) << endl;

    tic           = clock();
    auto [c3, z3] = fft_gpu_ct_dit_01(x); // this fails for n > 64
    toc           = clock();
    cout << "GPU cycles: " << toc - tic << endl;
    cout << "GPU CT DIT 01 rel diff " << compare(z1, z3) << endl;
    cout << "GPU CT DIT 01 time: " << c3 << "; " << double(c3) / CLOCKS_PER_SEC * 1000 << "ms"
         << "; " << toc - tic << endl;

    tic           = clock();
    auto [c4, z4] = fft_gpu_ct_dit_02(x); // this fails for n > 1024
    toc           = clock();
    cout << "GPU cycles: " << toc - tic << endl;
    cout << "GPU CT DIT 02 rel diff " << compare(z1, z4) << endl;
    cout << "GPU CT DIT 02 time: " << c4 << "; " << double(c4) / CLOCKS_PER_SEC * 1000 << "ms"
         << "; " << toc - tic << endl;
}

void test2d()
{
    clock_t tic, toc;

    size_t const n = (size_t)pow(2, 10);
    auto         x = random_vector(n * n);

    cout << "2d input length: " << n << "x" << n << endl;

    tic     = clock();
    auto z1 = fft_fftw_2d(x, n, n);
    toc     = clock();
    cout << "FFTW cycles: " << toc - tic << endl;

    tic           = clock();
    auto [c2, z2] = fft_gpu_ct_dit_2d_02(x, n, n);
    toc           = clock();
    cout << "GPU cycles: " << toc - tic << endl;
    cout << "GPU CT DIT 02 rel diff " << compare(z1, z2) << endl;
    cout << "GPU CT DIT 02 time: " << c2 << "; " << double(c2) / CLOCKS_PER_SEC * 1000 << "ms"
         << "; " << toc - tic << endl;
}

void test3d()
{
    clock_t tic, toc;

    size_t const n = (size_t)pow(2, 9);
    auto         x = random_vector(n * n * n);

    cout << "3d input length: " << n << "x" << n << "x" << n << endl;

    tic     = clock();
    auto z1 = fft_fftw_3d(x, n, n, n);
    toc     = clock();
    cout << "FFTW cycles: " << toc - tic << endl;

    tic           = clock();
    auto [c2, z2] = fft_gpu_ct_dit_3d_02(x, n, n, n);
    toc           = clock();
    cout << "GPU cycles: " << toc - tic << endl;
    cout << "GPU CT DIT 02 rel diff " << compare(z1, z2) << endl;
    cout << "GPU CT DIT 02 time: " << c2 << "; " << double(c2) / CLOCKS_PER_SEC * 1000 << "ms"
         << "; " << toc - tic << endl;
}

int main(int argc, char* argv[])
{
    test1d();
    test2d();
    test3d();
}

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

using namespace std;

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
// This version does one iteration at a time...
//
__global__ void cooley_tukey1(int a, int b, hipDoubleComplex* x)
{
    int l = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int k = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(l >= b / 2)
        return;
    if(k >= a)
        return;

    double theta = -2 * PI * l / b;
    double cost  = cos(theta);
    double sint  = sin(theta);

    int p = l + k * b;
    int q = p + b / 2;

    hipDoubleComplex xp = x[p];
    hipDoubleComplex xq = x[q];

    x[p]   = xp + xq;
    x[q].x = cost * (xp.x - xq.x) - sint * (xp.y - xq.y);
    x[q].y = cost * (xp.y - xq.y) + sint * (xp.x - xq.x);
}

vector<fftw_complex> fft_gpu1(vector<fftw_complex> const& x)
{
    auto const N     = x.size();
    auto const log2N = (size_t)log2(N);

    void* X;
    hipMalloc(&X, N * sizeof(fftw_complex));
    hipMemcpy(X, x.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice);

    auto tic = clock();
    for(int s = 0; s < log2N; ++s)
    {
        auto a = (size_t)pow(2, s);
        auto b = N / a;
        dim3 threads(16, 16);
        dim3 blocks(max(1, b / 32), max(1, a / 16));
        cooley_tukey1<<<blocks, threads>>>(a, b, (hipDoubleComplex*)X);
    }
    hipDeviceSynchronize();
    auto toc = clock();

    cout << "GPU time (inner): " << double(toc - tic) / CLOCKS_PER_SEC * 1000 << "ms"
         << endl; // echoing this here is kinda gross...

    vector<fftw_complex> z(N);
    hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost);
    hipFree(X);

    for(size_t p = 0; p < N; ++p)
    {
        auto q = bitreverse(p, log2N);
        if(p > q)
            swap(z[p], z[q]);
    }

    return z;
}

//
// This version does a single element across all iterations.  It
// doesn't synchronise properly so will fail for large N.
//
__global__ void cooley_tukey2(int N, hipDoubleComplex* x)
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

vector<fftw_complex> fft_gpu2(vector<fftw_complex> const& x)
{
    auto const N     = x.size();
    auto const log2N = (size_t)log2(N);

    void* X;
    hipMalloc(&X, N * sizeof(fftw_complex));
    hipMemcpy(X, x.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice);

    auto tic     = clock();
    int  threads = N;
    int  blocks  = 1;
    cooley_tukey2<<<blocks, threads>>>(N, (hipDoubleComplex*)X);
    hipDeviceSynchronize();
    auto toc = clock();

    cout << "GPU time (inner): " << double(toc - tic) / CLOCKS_PER_SEC * 1000 << "ms"
         << endl; // echoing this here is kinda gross...

    vector<fftw_complex> z(N);
    hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost);
    hipFree(X);

    for(size_t p = 0; p < N; ++p)
    {
        auto q = bitreverse(p, log2N);
        if(p > q)
            swap(z[p], z[q]);
    }

    return z;
}

//
// This version does a single element across all iterations;
// bit-reverse done first.
//
__device__ void cooley_tukey3_(int N, hipDoubleComplex* x)
{

    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i >= N)
        return;

    int P = N; // current number of blocks
    int M = 1; // size of current block

    while(M < N)
    {
        if(M > 64)
            __syncthreads();

        if((i / M) % 2 != 0)
        {
            M <<= 1;
            P >>= 1;
            continue;
        }

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

__global__ void cooley_tukey3(int N, hipDoubleComplex* x_)
{
    int                         i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    __shared__ hipDoubleComplex x[1024];
    x[i] = x_[i];
    cooley_tukey3_(N, x);
    __syncthreads();
    x_[i] = x[i];
}

vector<fftw_complex> fft_gpu3(vector<fftw_complex> const& x)
{
    auto const N     = x.size();
    auto const log2N = (size_t)log2(N);

    auto z = copy(x);
    for(size_t p = 0; p < N; ++p)
    {
        auto q = bitreverse(p, log2N);
        if(p > q)
            swap(z[p], z[q]);
    }

    void* X;
    hipMalloc(&X, N * sizeof(fftw_complex));
    hipMemcpy(X, z.data(), N * sizeof(fftw_complex), hipMemcpyHostToDevice);

    auto tic     = clock();
    int  threads = N;
    int  blocks  = 1;
    cooley_tukey3<<<blocks, threads>>>(N, (hipDoubleComplex*)X);
    hipDeviceSynchronize();
    auto toc = clock();

    cout << "GPU time (inner): " << double(toc - tic) / CLOCKS_PER_SEC * 1000 << "ms"
         << endl; // echoing this here is kinda gross...

    hipMemcpy(z.data(), X, N * sizeof(fftw_complex), hipMemcpyDeviceToHost);
    hipFree(X);

    return z;
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
void test_ct()
{
    clock_t tic, toc;

    size_t const n = (size_t)pow(2, 10);
    auto         x = random_vector(n);

    cout << "1d input length: " << n << endl;

    tic     = clock();
    auto z1 = fft_fftw(x);
    toc     = clock();
    cout << "FFTW cycles: " << toc - tic << endl;

    // tic = clock();
    // auto z2 = fft_naive(x);
    // toc = clock();
    // cout << "NAIVE time: " << toc - tic << endl;

    tic     = clock();
    auto z3 = fft_ict(x);
    toc     = clock();
    cout << "ICT cycles: " << toc - tic << endl;

    tic     = clock();
    auto z4 = fft_gpu1(x);
    toc     = clock();
    cout << "GPU cycles: " << toc - tic << endl;

    tic     = clock();
    auto z5 = fft_gpu2(x); // this fails for large n
    toc     = clock();
    cout << "GPU cycles: " << toc - tic << endl;

    tic     = clock();
    auto z6 = fft_gpu3(x); // this fails for large n
    toc     = clock();
    cout << "GPU cycles: " << toc - tic << endl;

    cout << "ICT rel diff " << compare(z1, z3) << endl;
    cout << "GPU1 rel diff " << compare(z1, z4) << endl;
    cout << "GPU2 rel diff " << compare(z1, z5) << endl;
    cout << "GPU3 rel diff " << compare(z1, z6) << endl;
}

int main(int argc, char* argv[])
{
    test_ct();
}

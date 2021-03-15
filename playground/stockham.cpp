/// build:
///    /opt/rocm/bin/hipcc st_256_quick_test.cpp  -o st_256_quick_test -lfftw3f

#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>

#include <fftw3.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#include "timer.h"

#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return {};

using namespace std;

template <class T>
struct real_type;

template <>
struct real_type<hipComplex>
{
    typedef float type;
};

template <>
struct real_type<hipDoubleComplex>
{
    typedef double type;
};

template <class T>
using real_type_t = typename real_type<T>::type;

#include "butterfly_template.h"

//
// Random inputs
//
template <typename T>
vector<T> random_vector(size_t n)
{
    vector<T>                         x(n);
    random_device                     rd;
    mt19937                           gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
    for(size_t i = 0; i < n; ++i)
    {
        // always use double for dis(gen), save as real type
        x[i].x = dis(gen);
        x[i].y = dis(gen);
    }
    return x;
}

//
// Copy helper for dtype (which aren't assignable!)
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

float average(vector<float> x)
{
    return accumulate(x.cbegin(), x.cend(), 0.0) / x.size();
}

//
// FFTW backed FFT
//
pair<float, vector<hipDoubleComplex>>
    fft_fftw(vector<hipDoubleComplex> const& x, int nx, int nbatch)
{
    // std::cout << "Complex Double" << std::endl;
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
    // std::cout << "Complex Single" << std::endl;
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
// Stockham
//
template <typename T>
__global__ void stockham_twiddles(int ntwiddles, T* twiddles, int nfactors, int* factors)
{
    int m = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(m >= ntwiddles)
        return;

    int n       = 0;
    int factor  = factors[0];
    int nroots  = factor;
    int nraccum = factor - 1;
    while(m > nraccum - 1)
    {
        factor = factors[++n];
        nroots *= factor;
        nraccum += (factor - 1) * (nroots / factor);
    }

    int m0 = nraccum - (nroots / factor) * (factor - 1);
    int j  = (m - m0) % (factor - 1) + 1;
    int k  = (m - m0) / (factor - 1);

    // always use double for sine cosine, cast to real type when saving to twiddles
    double cost, sint;
    sincospi(-2 * double(j) * k / nroots, &sint, &cost);
    twiddles[m].x = real_type_t<T>(cost);
    twiddles[m].y = real_type_t<T>(sint);
}

#define TWIDDLE_MUL_FWD(TWIDDLES, INDEX, REG)   \
    {                                           \
        T              W = TWIDDLES[INDEX]; \
        real_type_t<T> TR, TI;              \
        TR    = (W.x * REG.x) - (W.y * REG.y);  \
        TI    = (W.y * REG.x) + (W.x * REG.y);  \
        REG.x = TR;                             \
        REG.y = TI;                             \
    }

template <typename T>
__global__ void fft_256_fwd(T* gb, T* twiddles)
{
    __shared__ T lds[256];

    int ioOffset = 256 * blockIdx.x;
    T*  lwb      = gb + ioOffset;
    int me       = threadIdx.x;

    T X[4];

    X[0] = lwb[me + 0];
    X[1] = lwb[me + 64];
    X[2] = lwb[me + 128];
    X[3] = lwb[me + 192];

    FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

    lds[me * 4 + 0] = X[0];
    lds[me * 4 + 1] = X[1];
    lds[me * 4 + 2] = X[2];
    lds[me * 4 + 3] = X[3];

    X[0] = lds[me + 0];
    X[1] = lds[me + 64];
    X[2] = lds[me + 128];
    X[3] = lds[me + 192];

    TWIDDLE_MUL_FWD(twiddles, 3 + 3 * (me % 4) + 0, X[1])
    TWIDDLE_MUL_FWD(twiddles, 3 + 3 * (me % 4) + 1, X[2])
    TWIDDLE_MUL_FWD(twiddles, 3 + 3 * (me % 4) + 2, X[3])

    FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

    lds[(me / 4) * 16 + me % 4 + 0]  = X[0];
    lds[(me / 4) * 16 + me % 4 + 4]  = X[1];
    lds[(me / 4) * 16 + me % 4 + 8]  = X[2];
    lds[(me / 4) * 16 + me % 4 + 12] = X[3];

    X[0] = lds[me + 0];
    X[1] = lds[me + 64];
    X[2] = lds[me + 128];
    X[3] = lds[me + 192];

    TWIDDLE_MUL_FWD(twiddles, 15 + 3 * (me % 16) + 0, X[1])
    TWIDDLE_MUL_FWD(twiddles, 15 + 3 * (me % 16) + 1, X[2])
    TWIDDLE_MUL_FWD(twiddles, 15 + 3 * (me % 16) + 2, X[3])

    FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

    lds[(me / 16) * 64 + me % 16 + 0]  = X[0];
    lds[(me / 16) * 64 + me % 16 + 16] = X[1];
    lds[(me / 16) * 64 + me % 16 + 32] = X[2];
    lds[(me / 16) * 64 + me % 16 + 48] = X[3];

    X[0] = lds[me + 0];
    X[1] = lds[me + 64];
    X[2] = lds[me + 128];
    X[3] = lds[me + 192];

    TWIDDLE_MUL_FWD(twiddles, 63 + 3 * me + 0, X[1])
    TWIDDLE_MUL_FWD(twiddles, 63 + 3 * me + 1, X[2])
    TWIDDLE_MUL_FWD(twiddles, 63 + 3 * me + 2, X[3])

    FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

    lwb[me + 0]   = X[0];
    lwb[me + 64]  = X[1];
    lwb[me + 128] = X[2];
    lwb[me + 192] = X[3];
}

//
// gb - global buffer
// twiddles - twiddle table
// nbpt - number of batches per thread
//
#define GLBIDX(b, i) (256 * nbpt * blockIdx.x + 256 * b + i)
#define LCLIDX(b, i) (256 * b + i)

template <int nbpt, typename T>
__global__ void fft_256_fwd_batchfirst(T* gb, T* twiddles)
{
    T __shared__ lds[256 * nbpt];
    T            X[4], W[4];
    T            t;

    int me = threadIdx.x;
    int idx;

    for(int b = 0; b < nbpt; ++b)
    {
        idx  = GLBIDX(b, me);
        X[0] = gb[idx + 0];
        X[1] = gb[idx + 64];
        X[2] = gb[idx + 128];
        X[3] = gb[idx + 192];

        FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

        idx          = LCLIDX(b, me * 4);
        lds[idx + 0] = X[0];
        lds[idx + 1] = X[1];
        lds[idx + 2] = X[2];
        lds[idx + 3] = X[3];
    }

    idx  = 3 + 3 * (me % 4);
    W[1] = twiddles[idx + 0];
    W[2] = twiddles[idx + 1];
    W[3] = twiddles[idx + 2];

    for(int b = 0; b < nbpt; ++b)
    {
        idx  = LCLIDX(b, me);
        X[0] = lds[idx + 0];
        X[1] = lds[idx + 64];
        X[2] = lds[idx + 128];
        X[3] = lds[idx + 192];

        t.x  = W[1].x * X[1].x - W[1].y * X[1].y;
        t.y  = W[1].y * X[1].x + W[1].x * X[1].y;
        X[1] = t;

        t.x  = W[2].x * X[2].x - W[2].y * X[2].y;
        t.y  = W[2].y * X[2].x + W[2].x * X[2].y;
        X[2] = t;

        t.x  = W[3].x * X[3].x - W[3].y * X[3].y;
        t.y  = W[3].y * X[3].x + W[3].x * X[3].y;
        X[3] = t;

        FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

        idx           = LCLIDX(b, (me / 4) * 16 + me % 4);
        lds[idx + 0]  = X[0];
        lds[idx + 4]  = X[1];
        lds[idx + 8]  = X[2];
        lds[idx + 12] = X[3];
    }

    idx  = 15 + 3 * (me % 16);
    W[1] = twiddles[idx + 0];
    W[2] = twiddles[idx + 1];
    W[3] = twiddles[idx + 2];

    for(int b = 0; b < nbpt; ++b)
    {
        idx  = LCLIDX(b, me);
        X[0] = lds[idx + 0];
        X[1] = lds[idx + 64];
        X[2] = lds[idx + 128];
        X[3] = lds[idx + 192];

        t.x  = W[1].x * X[1].x - W[1].y * X[1].y;
        t.y  = W[1].y * X[1].x + W[1].x * X[1].y;
        X[1] = t;

        t.x  = W[2].x * X[2].x - W[2].y * X[2].y;
        t.y  = W[2].y * X[2].x + W[2].x * X[2].y;
        X[2] = t;

        t.x  = W[3].x * X[3].x - W[3].y * X[3].y;
        t.y  = W[3].y * X[3].x + W[3].x * X[3].y;
        X[3] = t;

        FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

        idx           = LCLIDX(b, (me / 16) * 64 + me % 16);
        lds[idx + 0]  = X[0];
        lds[idx + 16] = X[1];
        lds[idx + 32] = X[2];
        lds[idx + 48] = X[3];
    }

    idx  = 63 + 3 * me;
    W[1] = twiddles[idx + 0];
    W[2] = twiddles[idx + 1];
    W[3] = twiddles[idx + 2];

    for(int b = 0; b < nbpt; ++b)
    {
        idx  = LCLIDX(b, me);
        X[0] = lds[idx + 0];
        X[1] = lds[idx + 64];
        X[2] = lds[idx + 128];
        X[3] = lds[idx + 192];

        t.x  = W[1].x * X[1].x - W[1].y * X[1].y;
        t.y  = W[1].y * X[1].x + W[1].x * X[1].y;
        X[1] = t;

        t.x  = W[2].x * X[2].x - W[2].y * X[2].y;
        t.y  = W[2].y * X[2].x + W[2].x * X[2].y;
        X[2] = t;

        t.x  = W[3].x * X[3].x - W[3].y * X[3].y;
        t.y  = W[3].y * X[3].x + W[3].x * X[3].y;
        X[3] = t;

        FwdRad4B1(&X[0], &X[1], &X[2], &X[3]);

        idx           = GLBIDX(b, me);
        gb[idx + 0]   = X[0];
        gb[idx + 64]  = X[1];
        gb[idx + 128] = X[2];
        gb[idx + 192] = X[3];
    }
}

#ifdef USE_GENERATED
#include "stockham_generated_kernel.h"
#endif

template <typename T>
tuple<float, float, vector<T>> fft_stockham_gpu(vector<T> const& x, int nx, int nbatch, int nbpt)
{
    vector<float> times;
    int           ntrials = 10;

    auto z = copy(x);

    T* X;
    HIP_CHECK(hipMalloc(&X, nx * nbatch * sizeof(T)));
    HIP_CHECK(hipMemcpy(X, z.data(), nx * nbatch * sizeof(T), hipMemcpyHostToDevice));

    T* twiddles;
    HIP_CHECK(hipMalloc(&twiddles, (nx - 1) * sizeof(T)));

    GPUTimer total;
    total.tic();
    vector<int> factors;

#ifndef USE_GENERATED
    // 256
    factors = {4, 4, 4, 4};
#else
    factors = GENERATED_FACTORS;
#endif

    int* d_factors;
    HIP_CHECK(hipMalloc(&d_factors, factors.size() * sizeof(int)));
    HIP_CHECK(
        hipMemcpy(d_factors, factors.data(), factors.size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, nx - 1>>>(nx - 1, twiddles, factors.size(), d_factors);
    HIP_CHECK(hipFree(d_factors));

    if(false)
    {
        vector<T> t(nx - 1);
        HIP_CHECK(hipMemcpy(t.data(), twiddles, t.size() * sizeof(T), hipMemcpyDeviceToHost));
        for(int i = 0; i < nx - 1; ++i)
            cout << i << " " << t[i].x << " " << t[i].y << endl;
    }

    GPUTimer timer;
    for(int n = 0; n <= ntrials; ++n)
    {
        timer.tic();
#ifndef USE_GENERATED
        if(nx == 256)
        {
            if(nbpt > 1)
            {
                switch(nbpt)
                {
                case 2:
                    fft_256_fwd_batchfirst<2><<<nbatch / 2, 64>>>(X, twiddles);
                    break;
                case 4:
                    fft_256_fwd_batchfirst<4><<<nbatch / 4, 64>>>(X, twiddles);
                    break;
                case 8:
                    fft_256_fwd_batchfirst<8><<<nbatch / 8, 64>>>(X, twiddles);
                    break;
                case 16:
                    fft_256_fwd_batchfirst<16><<<nbatch / 16, 64>>>(X, twiddles);
                    break;
                default:
                    cout << "INVALID NBPT" << endl;
                }
            }
            else
            {
                fft_256_fwd<<<nbatch, 64>>>(X, twiddles);
            }
        }
#else
        // 1,1 means unit-stride (TODO: arbitrary stride)
        GENERATED_KERNEL_LAUNCH(X, nbatch, twiddles, 1, 1);
#endif

        timer.toc();
        if(n > 0)
            times.push_back(timer.elapsed());
        if(n == 0)
            HIP_CHECK(hipMemcpy(z.data(), X, nx * nbatch * sizeof(T), hipMemcpyDeviceToHost));
    }
    total.toc();

    HIP_CHECK(hipFree(twiddles));
    HIP_CHECK(hipFree(X));

    return {average(times), total.elapsed(), move(z)};
}

//
// Relative difference
//
template <typename T>
double compare(vector<T> const& z1, vector<T> const& z2)
{
    double d = 0.0;
    double r = 0.0;
    for(size_t n = 0; n < z1.size(); ++n)
    {
        double dx = z1[n].x - z2[n].x;
        double dy = z1[n].y - z2[n].y;
        // if(dx * dx + dy * dy > 1.e-7)
        // {
        //     cout << n << " " << sqrt(dx * dx + dy * dy) << " " << z1[n].x << " " << z2[n].x << endl;
        // }
        d += dx * dx + dy * dy;
        r += z1[n].x * z1[n].x + z1[n].y * z1[n].y;
    }
    return sqrt(d) / sqrt(r);
}

//
// Some tests!
//
template <typename T>
void test1d(size_t n, size_t nbatch, size_t nbpt)
{
    double GiB = double(n * nbatch * sizeof(T)) / 1024 / 1024 / 1024;
    cout << "# 1d test" << endl;
    cout << "1d input length: " << n << " (" << nbatch << ")" << endl;
    cout << "1d input size:   " << GiB << "GiB" << endl;

    auto x = random_vector<T>(n * nbatch);

    auto [t1, z1] = fft_fftw(x, n, nbatch);
    cout << "FFTW time:       " << t1 << "ms" << endl;

    auto [t2, t2t, z2] = fft_stockham_gpu(x, n, nbatch, nbpt);

    cout << "GPU rel diff:    " << compare(z1, z2) << endl;
    cout << "GPU kernel time: " << t2 << "ms"
         << " / " << t2t << "ms" << endl;
    cout << "GPU throughput:  " << GiB * 1000 / t2 << " GiB/s" << endl;
}

int main(int argc, char* argv[])
{
    size_t length = 256;
    size_t nbatch = 1;
    size_t nbpt   = 1;
    size_t single = 1;
    if(argc > 1)
        length = stoi(argv[1]);
    if(argc > 2)
        nbatch = stoi(argv[2]);
    if(argc > 3)
        single = stoi(argv[3]);
    if(argc > 4)
        nbpt = stoi(argv[4]);

    if (single)
        test1d<hipComplex>(length, nbatch, nbpt);
    else
        test1d<hipDoubleComplex>(length, nbatch, nbpt);
}

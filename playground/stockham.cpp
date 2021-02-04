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
using dtype       = hipDoubleComplex;
using fft_result  = pair<float, vector<dtype>>;
using fft_result2 = tuple<float, float, vector<dtype>>;

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
// Stockham
//

__global__ void stockham_twiddles(int ntwiddles, dtype* twiddles, int nfactors, int* factors)
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

    real_type_t<dtype> cost, sint;
    sincospi(-2 * real_type_t<dtype>(j) * k / nroots, &sint, &cost);
    twiddles[m].x = cost;
    twiddles[m].y = sint;
}

#define TWIDDLE_MUL_FWD(TWIDDLES, INDEX, REG)   \
    {                                           \
        dtype              W = TWIDDLES[INDEX]; \
        real_type_t<dtype> TR, TI;              \
        TR    = (W.x * REG.x) - (W.y * REG.y);  \
        TI    = (W.y * REG.x) + (W.x * REG.y);  \
        REG.x = TR;                             \
        REG.y = TI;                             \
    }

__device__ void FwdRad2B1(dtype* R0, dtype* R1)
{

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

__device__ void FwdRad4(dtype* R0, dtype* R2, dtype* R1, dtype* R3)
{
    dtype T;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0f * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0f * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0f * (*R0) - (*R2);
    (*R3) = (*R1) + dtype(-(*R3).y, (*R3).x);
    (*R1) = 2.0f * (*R1) - (*R3);

    T     = (*R1);
    (*R1) = (*R2);
    (*R2) = T;
}

// butterfly radix-7 constants
#define C7Q1 -1.16666666666666651863693004997913
#define C7Q2 0.79015646852540022404554065360571
#define C7Q3 0.05585426728964774240049351305970
#define C7Q4 0.73430220123575240531721419756650
#define C7Q5 0.44095855184409837868031445395900
#define C7Q6 0.34087293062393136944265847887436
#define C7Q7 -0.53396936033772524066165487965918
#define C7Q8 0.87484229096165666561546458979137

__device__ void
    FwdRad7B1(dtype* R0, dtype* R1, dtype* R2, dtype* R3, dtype* R4, dtype* R5, dtype* R6)
{

    dtype p0;
    dtype p1;
    dtype p2;
    dtype p3;
    dtype p4;
    dtype p5;
    dtype p6;
    dtype p7;
    dtype p8;
    dtype p9;
    dtype q0;
    dtype q1;
    dtype q2;
    dtype q3;
    dtype q4;
    dtype q5;
    dtype q6;
    dtype q7;
    dtype q8;
    /*FFT7 Forward Complex */

    p0 = *R1 + *R6;
    p1 = *R1 - *R6;
    p2 = *R2 + *R5;
    p3 = *R2 - *R5;
    p4 = *R4 + *R3;
    p5 = *R4 - *R3;

    p6 = p2 + p0;
    q4 = p2 - p0;
    q2 = p0 - p4;
    q3 = p4 - p2;
    p7 = p5 + p3;
    q7 = p5 - p3;
    q6 = p1 - p5;
    q8 = p3 - p1;
    q1 = p6 + p4;
    q5 = p7 + p1;
    q0 = *R0 + q1;

    q1 = C7Q1 * q1;
    q2 = C7Q2 * q2;
    q3 = C7Q3 * q3;
    q4 = C7Q4 * q4;

    q5 = C7Q5 * q5;
    q6 = C7Q6 * q6;
    q7 = C7Q7 * q7;
    q8 = C7Q8 * q8;

    p0 = q0 + q1;
    p1 = q2 + q3;
    p2 = q4 - q3;
    p3 = -q2 - q4;
    p4 = q6 + q7;
    p5 = q8 - q7;
    p6 = -q8 - q6;
    p7 = p0 + p1;
    p8 = p0 + p2;
    p9 = p0 + p3;
    q6 = p4 + q5;
    q7 = p5 + q5;
    q8 = p6 + q5;

    *R0     = q0;
    (*R1).x = p7.x + q6.y;
    (*R1).y = p7.y - q6.x;
    (*R2).x = p9.x + q8.y;
    (*R2).y = p9.y - q8.x;
    (*R3).x = p8.x - q7.y;
    (*R3).y = p8.y + q7.x;
    (*R4).x = p8.x + q7.y;
    (*R4).y = p8.y - q7.x;
    (*R5).x = p9.x - q8.y;
    (*R5).y = p9.y + q8.x;
    (*R6).x = p7.x - q6.y;
    (*R6).y = p7.y + q6.x;
}

__global__ void fft_256_fwd(dtype* gb, dtype* twiddles)
{
    __shared__ dtype lds[256];

    int    ioOffset = 256 * blockIdx.x;
    dtype* lwb      = gb + ioOffset;
    int    me       = threadIdx.x;

    dtype X[4];

    X[0] = lwb[me + 0];
    X[1] = lwb[me + 64];
    X[2] = lwb[me + 128];
    X[3] = lwb[me + 192];

    FwdRad4(&X[0], &X[1], &X[2], &X[3]);

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

    FwdRad4(&X[0], &X[1], &X[2], &X[3]);

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

    FwdRad4(&X[0], &X[1], &X[2], &X[3]);

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

    FwdRad4(&X[0], &X[1], &X[2], &X[3]);

    lwb[me + 0]   = X[0];
    lwb[me + 64]  = X[1];
    lwb[me + 128] = X[2];
    lwb[me + 192] = X[3];
}

template <typename scalar_type>
__device__ void forward_length56_pass0(scalar_type*       input,
                                       scalar_type*       output,
                                       unsigned int       rw,
                                       unsigned int       thread,
                                       const scalar_type* twiddles,
                                       const size_t       stride_in,
                                       const size_t       stride_out,
                                       unsigned int       offset_in,
                                       unsigned int       offset_out,
                                       scalar_type*       R0,
                                       scalar_type*       R1,
                                       scalar_type*       R2,
                                       scalar_type*       R3,
                                       scalar_type*       R4,
                                       scalar_type*       R5,
                                       scalar_type*       R6,
                                       scalar_type*       R7,
                                       scalar_type*       R8,
                                       scalar_type*       R9,
                                       scalar_type*       R10,
                                       scalar_type*       R11,
                                       scalar_type*       R12,
                                       scalar_type*       R13)
{
    if(rw)
    {
        (*R0)  = input[offset_in + (2 * thread + 0) * stride_in];
        (*R7)  = input[offset_in + (2 * thread + 1) * stride_in];
        (*R1)  = input[offset_in + (2 * thread + 8) * stride_in];
        (*R8)  = input[offset_in + (2 * thread + 9) * stride_in];
        (*R2)  = input[offset_in + (2 * thread + 16) * stride_in];
        (*R9)  = input[offset_in + (2 * thread + 17) * stride_in];
        (*R3)  = input[offset_in + (2 * thread + 24) * stride_in];
        (*R10) = input[offset_in + (2 * thread + 25) * stride_in];
        (*R4)  = input[offset_in + (2 * thread + 32) * stride_in];
        (*R11) = input[offset_in + (2 * thread + 33) * stride_in];
        (*R5)  = input[offset_in + (2 * thread + 40) * stride_in];
        (*R12) = input[offset_in + (2 * thread + 41) * stride_in];
        (*R6)  = input[offset_in + (2 * thread + 48) * stride_in];
        (*R13) = input[offset_in + (2 * thread + 49) * stride_in];
    }
    FwdRad7B1(R0, R1, R2, R3, R4, R5, R6);
    FwdRad7B1(R7, R8, R9, R10, R11, R12, R13);
    if(rw)
    {
        output[offset_out + (((2 * thread + 0) / 1) * 7 + (2 * thread + 0) % 1 + 0) * stride_out]
            = (*R0);
        output[offset_out + (((2 * thread + 0) / 1) * 7 + (2 * thread + 0) % 1 + 1) * stride_out]
            = (*R1);
        output[offset_out + (((2 * thread + 0) / 1) * 7 + (2 * thread + 0) % 1 + 2) * stride_out]
            = (*R2);
        output[offset_out + (((2 * thread + 0) / 1) * 7 + (2 * thread + 0) % 1 + 3) * stride_out]
            = (*R3);
        output[offset_out + (((2 * thread + 0) / 1) * 7 + (2 * thread + 0) % 1 + 4) * stride_out]
            = (*R4);
        output[offset_out + (((2 * thread + 0) / 1) * 7 + (2 * thread + 0) % 1 + 5) * stride_out]
            = (*R5);
        output[offset_out + (((2 * thread + 0) / 1) * 7 + (2 * thread + 0) % 1 + 6) * stride_out]
            = (*R6);
        output[offset_out + (((2 * thread + 1) / 1) * 7 + (2 * thread + 1) % 1 + 0) * stride_out]
            = (*R7);
        output[offset_out + (((2 * thread + 1) / 1) * 7 + (2 * thread + 1) % 1 + 1) * stride_out]
            = (*R8);
        output[offset_out + (((2 * thread + 1) / 1) * 7 + (2 * thread + 1) % 1 + 2) * stride_out]
            = (*R9);
        output[offset_out + (((2 * thread + 1) / 1) * 7 + (2 * thread + 1) % 1 + 3) * stride_out]
            = (*R10);
        output[offset_out + (((2 * thread + 1) / 1) * 7 + (2 * thread + 1) % 1 + 4) * stride_out]
            = (*R11);
        output[offset_out + (((2 * thread + 1) / 1) * 7 + (2 * thread + 1) % 1 + 5) * stride_out]
            = (*R12);
        output[offset_out + (((2 * thread + 1) / 1) * 7 + (2 * thread + 1) % 1 + 6) * stride_out]
            = (*R13);
    }
    if(rw)
    {
        (*R0)  = output[offset_out + 7 * thread + 0];
        (*R2)  = output[offset_out + 7 * thread + 1];
        (*R4)  = output[offset_out + 7 * thread + 2];
        (*R6)  = output[offset_out + 7 * thread + 3];
        (*R8)  = output[offset_out + 7 * thread + 4];
        (*R10) = output[offset_out + 7 * thread + 5];
        (*R12) = output[offset_out + 7 * thread + 6];
        (*R1)  = output[offset_out + 7 * thread + 28];
        (*R3)  = output[offset_out + 7 * thread + 29];
        (*R5)  = output[offset_out + 7 * thread + 30];
        (*R7)  = output[offset_out + 7 * thread + 31];
        (*R9)  = output[offset_out + 7 * thread + 32];
        (*R11) = output[offset_out + 7 * thread + 33];
        (*R13) = output[offset_out + 7 * thread + 34];
    }
}
template <typename scalar_type>
__device__ void forward_length56_pass1(scalar_type*       input,
                                       scalar_type*       output,
                                       unsigned int       rw,
                                       unsigned int       thread,
                                       const scalar_type* twiddles,
                                       const size_t       stride_in,
                                       const size_t       stride_out,
                                       unsigned int       offset_in,
                                       unsigned int       offset_out,
                                       scalar_type*       R0,
                                       scalar_type*       R1,
                                       scalar_type*       R2,
                                       scalar_type*       R3,
                                       scalar_type*       R4,
                                       scalar_type*       R5,
                                       scalar_type*       R6,
                                       scalar_type*       R7,
                                       scalar_type*       R8,
                                       scalar_type*       R9,
                                       scalar_type*       R10,
                                       scalar_type*       R11,
                                       scalar_type*       R12,
                                       scalar_type*       R13)
{
    scalar_type W;
    scalar_type t;
    W      = twiddles[6 + 1 * ((7 * thread + 0) % 7)];
    t.x    = W.x * (*R1).x - W.y * (*R1).y;
    t.y    = W.y * (*R1).x + W.x * (*R1).y;
    (*R1)  = t;
    W      = twiddles[6 + 1 * ((7 * thread + 1) % 7)];
    t.x    = W.x * (*R3).x - W.y * (*R3).y;
    t.y    = W.y * (*R3).x + W.x * (*R3).y;
    (*R3)  = t;
    W      = twiddles[6 + 1 * ((7 * thread + 2) % 7)];
    t.x    = W.x * (*R5).x - W.y * (*R5).y;
    t.y    = W.y * (*R5).x + W.x * (*R5).y;
    (*R5)  = t;
    W      = twiddles[6 + 1 * ((7 * thread + 3) % 7)];
    t.x    = W.x * (*R7).x - W.y * (*R7).y;
    t.y    = W.y * (*R7).x + W.x * (*R7).y;
    (*R7)  = t;
    W      = twiddles[6 + 1 * ((7 * thread + 4) % 7)];
    t.x    = W.x * (*R9).x - W.y * (*R9).y;
    t.y    = W.y * (*R9).x + W.x * (*R9).y;
    (*R9)  = t;
    W      = twiddles[6 + 1 * ((7 * thread + 5) % 7)];
    t.x    = W.x * (*R11).x - W.y * (*R11).y;
    t.y    = W.y * (*R11).x + W.x * (*R11).y;
    (*R11) = t;
    W      = twiddles[6 + 1 * ((7 * thread + 6) % 7)];
    t.x    = W.x * (*R13).x - W.y * (*R13).y;
    t.y    = W.y * (*R13).x + W.x * (*R13).y;
    (*R13) = t;
    FwdRad2B1(R0, R1);
    FwdRad2B1(R2, R3);
    FwdRad2B1(R4, R5);
    FwdRad2B1(R6, R7);
    FwdRad2B1(R8, R9);
    FwdRad2B1(R10, R11);
    FwdRad2B1(R12, R13);
    if(rw)
    {
        output[offset_out + (((7 * thread + 0) / 7) * 14 + (7 * thread + 0) % 7 + 0) * stride_out]
            = (*R0);
        output[offset_out + (((7 * thread + 0) / 7) * 14 + (7 * thread + 0) % 7 + 7) * stride_out]
            = (*R1);
        output[offset_out + (((7 * thread + 1) / 7) * 14 + (7 * thread + 1) % 7 + 0) * stride_out]
            = (*R2);
        output[offset_out + (((7 * thread + 1) / 7) * 14 + (7 * thread + 1) % 7 + 7) * stride_out]
            = (*R3);
        output[offset_out + (((7 * thread + 2) / 7) * 14 + (7 * thread + 2) % 7 + 0) * stride_out]
            = (*R4);
        output[offset_out + (((7 * thread + 2) / 7) * 14 + (7 * thread + 2) % 7 + 7) * stride_out]
            = (*R5);
        output[offset_out + (((7 * thread + 3) / 7) * 14 + (7 * thread + 3) % 7 + 0) * stride_out]
            = (*R6);
        output[offset_out + (((7 * thread + 3) / 7) * 14 + (7 * thread + 3) % 7 + 7) * stride_out]
            = (*R7);
        output[offset_out + (((7 * thread + 4) / 7) * 14 + (7 * thread + 4) % 7 + 0) * stride_out]
            = (*R8);
        output[offset_out + (((7 * thread + 4) / 7) * 14 + (7 * thread + 4) % 7 + 7) * stride_out]
            = (*R9);
        output[offset_out + (((7 * thread + 5) / 7) * 14 + (7 * thread + 5) % 7 + 0) * stride_out]
            = (*R10);
        output[offset_out + (((7 * thread + 5) / 7) * 14 + (7 * thread + 5) % 7 + 7) * stride_out]
            = (*R11);
        output[offset_out + (((7 * thread + 6) / 7) * 14 + (7 * thread + 6) % 7 + 0) * stride_out]
            = (*R12);
        output[offset_out + (((7 * thread + 6) / 7) * 14 + (7 * thread + 6) % 7 + 7) * stride_out]
            = (*R13);
    }
    if(rw)
    {
        (*R0)  = output[offset_out + 7 * thread + 0];
        (*R2)  = output[offset_out + 7 * thread + 1];
        (*R4)  = output[offset_out + 7 * thread + 2];
        (*R6)  = output[offset_out + 7 * thread + 3];
        (*R8)  = output[offset_out + 7 * thread + 4];
        (*R10) = output[offset_out + 7 * thread + 5];
        (*R12) = output[offset_out + 7 * thread + 6];
        (*R1)  = output[offset_out + 7 * thread + 28];
        (*R3)  = output[offset_out + 7 * thread + 29];
        (*R5)  = output[offset_out + 7 * thread + 30];
        (*R7)  = output[offset_out + 7 * thread + 31];
        (*R9)  = output[offset_out + 7 * thread + 32];
        (*R11) = output[offset_out + 7 * thread + 33];
        (*R13) = output[offset_out + 7 * thread + 34];
    }
}
template <typename scalar_type>
__device__ void forward_length56_pass2(scalar_type*       input,
                                       scalar_type*       output,
                                       unsigned int       rw,
                                       unsigned int       thread,
                                       const scalar_type* twiddles,
                                       const size_t       stride_in,
                                       const size_t       stride_out,
                                       unsigned int       offset_in,
                                       unsigned int       offset_out,
                                       scalar_type*       R0,
                                       scalar_type*       R1,
                                       scalar_type*       R2,
                                       scalar_type*       R3,
                                       scalar_type*       R4,
                                       scalar_type*       R5,
                                       scalar_type*       R6,
                                       scalar_type*       R7,
                                       scalar_type*       R8,
                                       scalar_type*       R9,
                                       scalar_type*       R10,
                                       scalar_type*       R11,
                                       scalar_type*       R12,
                                       scalar_type*       R13)
{
    scalar_type W;
    scalar_type t;
    W      = twiddles[13 + 1 * ((7 * thread + 0) % 14)];
    t.x    = W.x * (*R1).x - W.y * (*R1).y;
    t.y    = W.y * (*R1).x + W.x * (*R1).y;
    (*R1)  = t;
    W      = twiddles[13 + 1 * ((7 * thread + 1) % 14)];
    t.x    = W.x * (*R3).x - W.y * (*R3).y;
    t.y    = W.y * (*R3).x + W.x * (*R3).y;
    (*R3)  = t;
    W      = twiddles[13 + 1 * ((7 * thread + 2) % 14)];
    t.x    = W.x * (*R5).x - W.y * (*R5).y;
    t.y    = W.y * (*R5).x + W.x * (*R5).y;
    (*R5)  = t;
    W      = twiddles[13 + 1 * ((7 * thread + 3) % 14)];
    t.x    = W.x * (*R7).x - W.y * (*R7).y;
    t.y    = W.y * (*R7).x + W.x * (*R7).y;
    (*R7)  = t;
    W      = twiddles[13 + 1 * ((7 * thread + 4) % 14)];
    t.x    = W.x * (*R9).x - W.y * (*R9).y;
    t.y    = W.y * (*R9).x + W.x * (*R9).y;
    (*R9)  = t;
    W      = twiddles[13 + 1 * ((7 * thread + 5) % 14)];
    t.x    = W.x * (*R11).x - W.y * (*R11).y;
    t.y    = W.y * (*R11).x + W.x * (*R11).y;
    (*R11) = t;
    W      = twiddles[13 + 1 * ((7 * thread + 6) % 14)];
    t.x    = W.x * (*R13).x - W.y * (*R13).y;
    t.y    = W.y * (*R13).x + W.x * (*R13).y;
    (*R13) = t;
    FwdRad2B1(R0, R1);
    FwdRad2B1(R2, R3);
    FwdRad2B1(R4, R5);
    FwdRad2B1(R6, R7);
    FwdRad2B1(R8, R9);
    FwdRad2B1(R10, R11);
    FwdRad2B1(R12, R13);
    if(rw)
    {
        output[offset_out + (((7 * thread + 0) / 14) * 28 + (7 * thread + 0) % 14 + 0) * stride_out]
            = (*R0);
        output[offset_out
               + (((7 * thread + 0) / 14) * 28 + (7 * thread + 0) % 14 + 14) * stride_out]
            = (*R1);
        output[offset_out + (((7 * thread + 1) / 14) * 28 + (7 * thread + 1) % 14 + 0) * stride_out]
            = (*R2);
        output[offset_out
               + (((7 * thread + 1) / 14) * 28 + (7 * thread + 1) % 14 + 14) * stride_out]
            = (*R3);
        output[offset_out + (((7 * thread + 2) / 14) * 28 + (7 * thread + 2) % 14 + 0) * stride_out]
            = (*R4);
        output[offset_out
               + (((7 * thread + 2) / 14) * 28 + (7 * thread + 2) % 14 + 14) * stride_out]
            = (*R5);
        output[offset_out + (((7 * thread + 3) / 14) * 28 + (7 * thread + 3) % 14 + 0) * stride_out]
            = (*R6);
        output[offset_out
               + (((7 * thread + 3) / 14) * 28 + (7 * thread + 3) % 14 + 14) * stride_out]
            = (*R7);
        output[offset_out + (((7 * thread + 4) / 14) * 28 + (7 * thread + 4) % 14 + 0) * stride_out]
            = (*R8);
        output[offset_out
               + (((7 * thread + 4) / 14) * 28 + (7 * thread + 4) % 14 + 14) * stride_out]
            = (*R9);
        output[offset_out + (((7 * thread + 5) / 14) * 28 + (7 * thread + 5) % 14 + 0) * stride_out]
            = (*R10);
        output[offset_out
               + (((7 * thread + 5) / 14) * 28 + (7 * thread + 5) % 14 + 14) * stride_out]
            = (*R11);
        output[offset_out + (((7 * thread + 6) / 14) * 28 + (7 * thread + 6) % 14 + 0) * stride_out]
            = (*R12);
        output[offset_out
               + (((7 * thread + 6) / 14) * 28 + (7 * thread + 6) % 14 + 14) * stride_out]
            = (*R13);
    }
    if(rw)
    {
        (*R0)  = output[offset_out + 7 * thread + 0];
        (*R2)  = output[offset_out + 7 * thread + 1];
        (*R4)  = output[offset_out + 7 * thread + 2];
        (*R6)  = output[offset_out + 7 * thread + 3];
        (*R8)  = output[offset_out + 7 * thread + 4];
        (*R10) = output[offset_out + 7 * thread + 5];
        (*R12) = output[offset_out + 7 * thread + 6];
        (*R1)  = output[offset_out + 7 * thread + 28];
        (*R3)  = output[offset_out + 7 * thread + 29];
        (*R5)  = output[offset_out + 7 * thread + 30];
        (*R7)  = output[offset_out + 7 * thread + 31];
        (*R9)  = output[offset_out + 7 * thread + 32];
        (*R11) = output[offset_out + 7 * thread + 33];
        (*R13) = output[offset_out + 7 * thread + 34];
    }
}
template <typename scalar_type>
__device__ void forward_length56_pass3(scalar_type*       input,
                                       scalar_type*       output,
                                       unsigned int       rw,
                                       unsigned int       thread,
                                       const scalar_type* twiddles,
                                       const size_t       stride_in,
                                       const size_t       stride_out,
                                       unsigned int       offset_in,
                                       unsigned int       offset_out,
                                       scalar_type*       R0,
                                       scalar_type*       R1,
                                       scalar_type*       R2,
                                       scalar_type*       R3,
                                       scalar_type*       R4,
                                       scalar_type*       R5,
                                       scalar_type*       R6,
                                       scalar_type*       R7,
                                       scalar_type*       R8,
                                       scalar_type*       R9,
                                       scalar_type*       R10,
                                       scalar_type*       R11,
                                       scalar_type*       R12,
                                       scalar_type*       R13)
{
    scalar_type W;
    scalar_type t;
    W      = twiddles[27 + 1 * ((7 * thread + 0) % 28)];
    t.x    = W.x * (*R1).x - W.y * (*R1).y;
    t.y    = W.y * (*R1).x + W.x * (*R1).y;
    (*R1)  = t;
    W      = twiddles[27 + 1 * ((7 * thread + 1) % 28)];
    t.x    = W.x * (*R3).x - W.y * (*R3).y;
    t.y    = W.y * (*R3).x + W.x * (*R3).y;
    (*R3)  = t;
    W      = twiddles[27 + 1 * ((7 * thread + 2) % 28)];
    t.x    = W.x * (*R5).x - W.y * (*R5).y;
    t.y    = W.y * (*R5).x + W.x * (*R5).y;
    (*R5)  = t;
    W      = twiddles[27 + 1 * ((7 * thread + 3) % 28)];
    t.x    = W.x * (*R7).x - W.y * (*R7).y;
    t.y    = W.y * (*R7).x + W.x * (*R7).y;
    (*R7)  = t;
    W      = twiddles[27 + 1 * ((7 * thread + 4) % 28)];
    t.x    = W.x * (*R9).x - W.y * (*R9).y;
    t.y    = W.y * (*R9).x + W.x * (*R9).y;
    (*R9)  = t;
    W      = twiddles[27 + 1 * ((7 * thread + 5) % 28)];
    t.x    = W.x * (*R11).x - W.y * (*R11).y;
    t.y    = W.y * (*R11).x + W.x * (*R11).y;
    (*R11) = t;
    W      = twiddles[27 + 1 * ((7 * thread + 6) % 28)];
    t.x    = W.x * (*R13).x - W.y * (*R13).y;
    t.y    = W.y * (*R13).x + W.x * (*R13).y;
    (*R13) = t;
    FwdRad2B1(R0, R1);
    FwdRad2B1(R2, R3);
    FwdRad2B1(R4, R5);
    FwdRad2B1(R6, R7);
    FwdRad2B1(R8, R9);
    FwdRad2B1(R10, R11);
    FwdRad2B1(R12, R13);
    if(rw)
    {
        output[offset_out + (7 * thread + 0) * stride_out]  = (*R0);
        output[offset_out + (7 * thread + 1) * stride_out]  = (*R2);
        output[offset_out + (7 * thread + 2) * stride_out]  = (*R4);
        output[offset_out + (7 * thread + 3) * stride_out]  = (*R6);
        output[offset_out + (7 * thread + 4) * stride_out]  = (*R8);
        output[offset_out + (7 * thread + 5) * stride_out]  = (*R10);
        output[offset_out + (7 * thread + 6) * stride_out]  = (*R12);
        output[offset_out + (7 * thread + 28) * stride_out] = (*R1);
        output[offset_out + (7 * thread + 29) * stride_out] = (*R3);
        output[offset_out + (7 * thread + 30) * stride_out] = (*R5);
        output[offset_out + (7 * thread + 31) * stride_out] = (*R7);
        output[offset_out + (7 * thread + 32) * stride_out] = (*R9);
        output[offset_out + (7 * thread + 33) * stride_out] = (*R11);
        output[offset_out + (7 * thread + 34) * stride_out] = (*R13);
    }
}

__global__ void fft_56_fwd(dtype* gb, dtype* twiddles)
{
    dtype __shared__ lds[56];

    int batch  = blockIdx.x;
    int thread = threadIdx.x;
    if(thread >= 4)
        return;

    unsigned int offset = batch * 56;

    dtype R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13;

    forward_length56_pass0<dtype>(gb,
                                  lds,
                                  1,
                                  thread,
                                  twiddles,
                                  1,
                                  1,
                                  offset,
                                  0,
                                  &R0,
                                  &R1,
                                  &R2,
                                  &R3,
                                  &R4,
                                  &R5,
                                  &R6,
                                  &R7,
                                  &R8,
                                  &R9,
                                  &R10,
                                  &R11,
                                  &R12,
                                  &R13);

    forward_length56_pass1<dtype>(lds,
                                  lds,
                                  1,
                                  thread,
                                  twiddles,
                                  1,
                                  1,
                                  0,
                                  0,
                                  &R0,
                                  &R1,
                                  &R2,
                                  &R3,
                                  &R4,
                                  &R5,
                                  &R6,
                                  &R7,
                                  &R8,
                                  &R9,
                                  &R10,
                                  &R11,
                                  &R12,
                                  &R13);

    forward_length56_pass2<dtype>(lds,
                                  lds,
                                  1,
                                  thread,
                                  twiddles,
                                  1,
                                  1,
                                  0,
                                  0,
                                  &R0,
                                  &R1,
                                  &R2,
                                  &R3,
                                  &R4,
                                  &R5,
                                  &R6,
                                  &R7,
                                  &R8,
                                  &R9,
                                  &R10,
                                  &R11,
                                  &R12,
                                  &R13);

    forward_length56_pass3<dtype>(lds,
                                  gb,
                                  1,
                                  thread,
                                  twiddles,
                                  1,
                                  1,
                                  0,
                                  offset,
                                  &R0,
                                  &R1,
                                  &R2,
                                  &R3,
                                  &R4,
                                  &R5,
                                  &R6,
                                  &R7,
                                  &R8,
                                  &R9,
                                  &R10,
                                  &R11,
                                  &R12,
                                  &R13);
}

template<typename scalar_type>
__global__ void fft_56_fwd_fat(scalar_type*       inout,
                               //                    bool               rw,
                               //                    int                thread,
                    const scalar_type* twiddles,
                    int                stride_in,
                    int                stride_out,
                    int                offset_in,
                    int                offset_out,
                    int                offset_lds)
{
    int thread = threadIdx.x;
    if(thread >= 4)
        return;

    __shared__ scalar_type lds[1024];
    scalar_type            R[14];
    scalar_type            W;
    scalar_type            t;
    R[0] = inout[offset_in + (2 * thread + 0) * stride_in];
    R[1] = inout[offset_in + (2 * thread + 8) * stride_in];
    R[2] = inout[offset_in + (2 * thread + 16) * stride_in];
    R[3] = inout[offset_in + (2 * thread + 24) * stride_in];
    R[4] = inout[offset_in + (2 * thread + 32) * stride_in];
    R[5] = inout[offset_in + (2 * thread + 40) * stride_in];
    R[6] = inout[offset_in + (2 * thread + 48) * stride_in];
    FwdRad7B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6]);
    lds[offset_lds + ((2 * thread + 0)) * 7 + 0] = R[0];
    lds[offset_lds + ((2 * thread + 0)) * 7 + 1] = R[1];
    lds[offset_lds + ((2 * thread + 0)) * 7 + 2] = R[2];
    lds[offset_lds + ((2 * thread + 0)) * 7 + 3] = R[3];
    lds[offset_lds + ((2 * thread + 0)) * 7 + 4] = R[4];
    lds[offset_lds + ((2 * thread + 0)) * 7 + 5] = R[5];
    lds[offset_lds + ((2 * thread + 0)) * 7 + 6] = R[6];
    R[7]                                         = inout[offset_in + (2 * thread + 1) * stride_in];
    R[8]                                         = inout[offset_in + (2 * thread + 9) * stride_in];
    R[9]                                         = inout[offset_in + (2 * thread + 17) * stride_in];
    R[10]                                        = inout[offset_in + (2 * thread + 25) * stride_in];
    R[11]                                        = inout[offset_in + (2 * thread + 33) * stride_in];
    R[12]                                        = inout[offset_in + (2 * thread + 41) * stride_in];
    R[13]                                        = inout[offset_in + (2 * thread + 49) * stride_in];
    FwdRad7B1(&R[7], &R[8], &R[9], &R[10], &R[11], &R[12], &R[13]);
    lds[offset_lds + ((2 * thread + 1)) * 7 + 0] = R[7];
    lds[offset_lds + ((2 * thread + 1)) * 7 + 1] = R[8];
    lds[offset_lds + ((2 * thread + 1)) * 7 + 2] = R[9];
    lds[offset_lds + ((2 * thread + 1)) * 7 + 3] = R[10];
    lds[offset_lds + ((2 * thread + 1)) * 7 + 4] = R[11];
    lds[offset_lds + ((2 * thread + 1)) * 7 + 5] = R[12];
    lds[offset_lds + ((2 * thread + 1)) * 7 + 6] = R[13];
    R[0]                                         = lds[offset_lds + 7 * thread + 0];
    R[1]                                         = lds[offset_lds + 7 * thread + 28];
    W                                            = twiddles[6 + 1 * ((7 * thread + 0) % 7)];
    t.x                                          = W.x * R[1].x - W.y * R[1].y;
    t.y                                          = W.y * R[1].x + W.x * R[1].y;
    R[1]                                         = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 0) / 7) * 14 + (7 * thread + 0) % 7 + 0)] = R[0];
    lds[offset_lds + (((7 * thread + 0) / 7) * 14 + (7 * thread + 0) % 7 + 7)] = R[1];
    R[2] = lds[offset_lds + 7 * thread + 1];
    R[3] = lds[offset_lds + 7 * thread + 29];
    W    = twiddles[6 + 1 * ((7 * thread + 1) % 7)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    FwdRad2B1(&R[2], &R[3]);
    lds[offset_lds + (((7 * thread + 1) / 7) * 14 + (7 * thread + 1) % 7 + 0)] = R[2];
    lds[offset_lds + (((7 * thread + 1) / 7) * 14 + (7 * thread + 1) % 7 + 7)] = R[3];
    R[0] = lds[offset_lds + 7 * thread + 2];
    R[1] = lds[offset_lds + 7 * thread + 30];
    W    = twiddles[6 + 1 * ((7 * thread + 2) % 7)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 2) / 7) * 14 + (7 * thread + 2) % 7 + 0)] = R[0];
    lds[offset_lds + (((7 * thread + 2) / 7) * 14 + (7 * thread + 2) % 7 + 7)] = R[1];
    R[2] = lds[offset_lds + 7 * thread + 3];
    R[3] = lds[offset_lds + 7 * thread + 31];
    W    = twiddles[6 + 1 * ((7 * thread + 3) % 7)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    FwdRad2B1(&R[2], &R[3]);
    lds[offset_lds + (((7 * thread + 3) / 7) * 14 + (7 * thread + 3) % 7 + 0)] = R[2];
    lds[offset_lds + (((7 * thread + 3) / 7) * 14 + (7 * thread + 3) % 7 + 7)] = R[3];
    R[0] = lds[offset_lds + 7 * thread + 4];
    R[1] = lds[offset_lds + 7 * thread + 32];
    W    = twiddles[6 + 1 * ((7 * thread + 4) % 7)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 4) / 7) * 14 + (7 * thread + 4) % 7 + 0)] = R[0];
    lds[offset_lds + (((7 * thread + 4) / 7) * 14 + (7 * thread + 4) % 7 + 7)] = R[1];
    R[2] = lds[offset_lds + 7 * thread + 5];
    R[3] = lds[offset_lds + 7 * thread + 33];
    W    = twiddles[6 + 1 * ((7 * thread + 5) % 7)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    FwdRad2B1(&R[2], &R[3]);
    lds[offset_lds + (((7 * thread + 5) / 7) * 14 + (7 * thread + 5) % 7 + 0)] = R[2];
    lds[offset_lds + (((7 * thread + 5) / 7) * 14 + (7 * thread + 5) % 7 + 7)] = R[3];
    R[0] = lds[offset_lds + 7 * thread + 6];
    R[1] = lds[offset_lds + 7 * thread + 34];
    W    = twiddles[6 + 1 * ((7 * thread + 6) % 7)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 6) / 7) * 14 + (7 * thread + 6) % 7 + 0)] = R[0];
    lds[offset_lds + (((7 * thread + 6) / 7) * 14 + (7 * thread + 6) % 7 + 7)] = R[1];
    R[0] = lds[offset_lds + 7 * thread + 0];
    R[1] = lds[offset_lds + 7 * thread + 28];
    W    = twiddles[13 + 1 * ((7 * thread + 0) % 14)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 0) / 14) * 28 + (7 * thread + 0) % 14 + 0)]  = R[0];
    lds[offset_lds + (((7 * thread + 0) / 14) * 28 + (7 * thread + 0) % 14 + 14)] = R[1];
    R[2] = lds[offset_lds + 7 * thread + 1];
    R[3] = lds[offset_lds + 7 * thread + 29];
    W    = twiddles[13 + 1 * ((7 * thread + 1) % 14)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    FwdRad2B1(&R[2], &R[3]);
    lds[offset_lds + (((7 * thread + 1) / 14) * 28 + (7 * thread + 1) % 14 + 0)]  = R[2];
    lds[offset_lds + (((7 * thread + 1) / 14) * 28 + (7 * thread + 1) % 14 + 14)] = R[3];
    R[0] = lds[offset_lds + 7 * thread + 2];
    R[1] = lds[offset_lds + 7 * thread + 30];
    W    = twiddles[13 + 1 * ((7 * thread + 2) % 14)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 2) / 14) * 28 + (7 * thread + 2) % 14 + 0)]  = R[0];
    lds[offset_lds + (((7 * thread + 2) / 14) * 28 + (7 * thread + 2) % 14 + 14)] = R[1];
    R[2] = lds[offset_lds + 7 * thread + 3];
    R[3] = lds[offset_lds + 7 * thread + 31];
    W    = twiddles[13 + 1 * ((7 * thread + 3) % 14)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    FwdRad2B1(&R[2], &R[3]);
    lds[offset_lds + (((7 * thread + 3) / 14) * 28 + (7 * thread + 3) % 14 + 0)]  = R[2];
    lds[offset_lds + (((7 * thread + 3) / 14) * 28 + (7 * thread + 3) % 14 + 14)] = R[3];
    R[0] = lds[offset_lds + 7 * thread + 4];
    R[1] = lds[offset_lds + 7 * thread + 32];
    W    = twiddles[13 + 1 * ((7 * thread + 4) % 14)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 4) / 14) * 28 + (7 * thread + 4) % 14 + 0)]  = R[0];
    lds[offset_lds + (((7 * thread + 4) / 14) * 28 + (7 * thread + 4) % 14 + 14)] = R[1];
    R[2] = lds[offset_lds + 7 * thread + 5];
    R[3] = lds[offset_lds + 7 * thread + 33];
    W    = twiddles[13 + 1 * ((7 * thread + 5) % 14)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    FwdRad2B1(&R[2], &R[3]);
    lds[offset_lds + (((7 * thread + 5) / 14) * 28 + (7 * thread + 5) % 14 + 0)]  = R[2];
    lds[offset_lds + (((7 * thread + 5) / 14) * 28 + (7 * thread + 5) % 14 + 14)] = R[3];
    R[0] = lds[offset_lds + 7 * thread + 6];
    R[1] = lds[offset_lds + 7 * thread + 34];
    W    = twiddles[13 + 1 * ((7 * thread + 6) % 14)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    lds[offset_lds + (((7 * thread + 6) / 14) * 28 + (7 * thread + 6) % 14 + 0)]  = R[0];
    lds[offset_lds + (((7 * thread + 6) / 14) * 28 + (7 * thread + 6) % 14 + 14)] = R[1];
    R[0] = lds[offset_lds + 7 * thread + 0];
    R[1] = lds[offset_lds + 7 * thread + 28];
    W    = twiddles[27 + 1 * ((7 * thread + 0) % 28)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    FwdRad2B1(&R[0], &R[1]);
    inout[offset_out + (7 * thread + 0) * stride_out]  = R[0];
    inout[offset_out + (7 * thread + 28) * stride_out] = R[1];
    R[2]                                               = lds[offset_lds + 7 * thread + 1];
    R[3]                                               = lds[offset_lds + 7 * thread + 29];
    W                                                  = twiddles[27 + 1 * ((7 * thread + 1) % 28)];
    t.x                                                = W.x * R[3].x - W.y * R[3].y;
    t.y                                                = W.y * R[3].x + W.x * R[3].y;
    R[3]                                               = t;
    FwdRad2B1(&R[2], &R[3]);
    inout[offset_out + (7 * thread + 1) * stride_out]  = R[2];
    inout[offset_out + (7 * thread + 29) * stride_out] = R[3];
    R[0]                                               = lds[offset_lds + 7 * thread + 2];
    R[1]                                               = lds[offset_lds + 7 * thread + 30];
    W                                                  = twiddles[27 + 1 * ((7 * thread + 2) % 28)];
    t.x                                                = W.x * R[1].x - W.y * R[1].y;
    t.y                                                = W.y * R[1].x + W.x * R[1].y;
    R[1]                                               = t;
    FwdRad2B1(&R[0], &R[1]);
    inout[offset_out + (7 * thread + 2) * stride_out]  = R[0];
    inout[offset_out + (7 * thread + 30) * stride_out] = R[1];
    R[2]                                               = lds[offset_lds + 7 * thread + 3];
    R[3]                                               = lds[offset_lds + 7 * thread + 31];
    W                                                  = twiddles[27 + 1 * ((7 * thread + 3) % 28)];
    t.x                                                = W.x * R[3].x - W.y * R[3].y;
    t.y                                                = W.y * R[3].x + W.x * R[3].y;
    R[3]                                               = t;
    FwdRad2B1(&R[2], &R[3]);
    inout[offset_out + (7 * thread + 3) * stride_out]  = R[2];
    inout[offset_out + (7 * thread + 31) * stride_out] = R[3];
    R[0]                                               = lds[offset_lds + 7 * thread + 4];
    R[1]                                               = lds[offset_lds + 7 * thread + 32];
    W                                                  = twiddles[27 + 1 * ((7 * thread + 4) % 28)];
    t.x                                                = W.x * R[1].x - W.y * R[1].y;
    t.y                                                = W.y * R[1].x + W.x * R[1].y;
    R[1]                                               = t;
    FwdRad2B1(&R[0], &R[1]);
    inout[offset_out + (7 * thread + 4) * stride_out]  = R[0];
    inout[offset_out + (7 * thread + 32) * stride_out] = R[1];
    R[2]                                               = lds[offset_lds + 7 * thread + 5];
    R[3]                                               = lds[offset_lds + 7 * thread + 33];
    W                                                  = twiddles[27 + 1 * ((7 * thread + 5) % 28)];
    t.x                                                = W.x * R[3].x - W.y * R[3].y;
    t.y                                                = W.y * R[3].x + W.x * R[3].y;
    R[3]                                               = t;
    FwdRad2B1(&R[2], &R[3]);
    inout[offset_out + (7 * thread + 5) * stride_out]  = R[2];
    inout[offset_out + (7 * thread + 33) * stride_out] = R[3];
    R[0]                                               = lds[offset_lds + 7 * thread + 6];
    R[1]                                               = lds[offset_lds + 7 * thread + 34];
    W                                                  = twiddles[27 + 1 * ((7 * thread + 6) % 28)];
    t.x                                                = W.x * R[1].x - W.y * R[1].y;
    t.y                                                = W.y * R[1].x + W.x * R[1].y;
    R[1]                                               = t;
    FwdRad2B1(&R[0], &R[1]);
    inout[offset_out + (7 * thread + 6) * stride_out]  = R[0];
    inout[offset_out + (7 * thread + 34) * stride_out] = R[1];
}

#define C8Q 0.70710678118654752440084436210485

__device__ void FwdRad8B1(dtype* R0, dtype* R4, dtype* R2, dtype* R6, dtype* R1, dtype* R5, dtype* R3, dtype* R7)
{

    dtype res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + dtype{-(*R3).y, (*R3).x};
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + dtype{-(*R7).y, (*R7).x};

    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) - C8Q * dtype{(*R5).y, -(*R5).x};
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + dtype{-(*R6).y, (*R6).x};
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) - C8Q * dtype{(*R7).y, -(*R7).x};
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename scalar_type>
__global__ void           fft_56_fwd_fat56(scalar_type*       inout,
                          //                          bool               rw,
                          //                          int                thread,
                          const scalar_type* twiddles,
                          int                stride_in,
                          int                stride_out,
                          int                offset_in,
                          int                offset_out,
                          int                offset_lds)
{
    int thread = threadIdx.x;
    if(thread >= 1)
        return;

    __shared__ scalar_type lds[1024];
    scalar_type            R[16];
    scalar_type            W;
    scalar_type            t;

    R[0] = inout[offset_in + (7 * thread + 0) * stride_in];
    R[1] = inout[offset_in + (7 * thread + 7) * stride_in];
    R[2] = inout[offset_in + (7 * thread + 14) * stride_in];
    R[3] = inout[offset_in + (7 * thread + 21) * stride_in];
    R[4] = inout[offset_in + (7 * thread + 28) * stride_in];
    R[5] = inout[offset_in + (7 * thread + 35) * stride_in];
    R[6] = inout[offset_in + (7 * thread + 42) * stride_in];
    R[7] = inout[offset_in + (7 * thread + 49) * stride_in];
    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
    lds[offset_lds + ((7 * thread + 0)) * 8 + 0] = R[0];
    lds[offset_lds + ((7 * thread + 0)) * 8 + 1] = R[1];
    lds[offset_lds + ((7 * thread + 0)) * 8 + 2] = R[2];
    lds[offset_lds + ((7 * thread + 0)) * 8 + 3] = R[3];
    lds[offset_lds + ((7 * thread + 0)) * 8 + 4] = R[4];
    lds[offset_lds + ((7 * thread + 0)) * 8 + 5] = R[5];
    lds[offset_lds + ((7 * thread + 0)) * 8 + 6] = R[6];
    lds[offset_lds + ((7 * thread + 0)) * 8 + 7] = R[7];

    R[8]  = inout[offset_in + (7 * thread + 1) * stride_in];
    R[9]  = inout[offset_in + (7 * thread + 8) * stride_in];
    R[10] = inout[offset_in + (7 * thread + 15) * stride_in];
    R[11] = inout[offset_in + (7 * thread + 22) * stride_in];
    R[12] = inout[offset_in + (7 * thread + 29) * stride_in];
    R[13] = inout[offset_in + (7 * thread + 36) * stride_in];
    R[14] = inout[offset_in + (7 * thread + 43) * stride_in];
    R[15] = inout[offset_in + (7 * thread + 50) * stride_in];
    FwdRad8B1(&R[8], &R[9], &R[10], &R[11], &R[12], &R[13], &R[14], &R[15]);
    lds[offset_lds + ((7 * thread + 1)) * 8 + 0] = R[8];
    lds[offset_lds + ((7 * thread + 1)) * 8 + 1] = R[9];
    lds[offset_lds + ((7 * thread + 1)) * 8 + 2] = R[10];
    lds[offset_lds + ((7 * thread + 1)) * 8 + 3] = R[11];
    lds[offset_lds + ((7 * thread + 1)) * 8 + 4] = R[12];
    lds[offset_lds + ((7 * thread + 1)) * 8 + 5] = R[13];
    lds[offset_lds + ((7 * thread + 1)) * 8 + 6] = R[14];
    lds[offset_lds + ((7 * thread + 1)) * 8 + 7] = R[15];

    R[0] = inout[offset_in + (7 * thread + 2) * stride_in];
    R[1] = inout[offset_in + (7 * thread + 9) * stride_in];
    R[2] = inout[offset_in + (7 * thread + 16) * stride_in];
    R[3] = inout[offset_in + (7 * thread + 23) * stride_in];
    R[4] = inout[offset_in + (7 * thread + 30) * stride_in];
    R[5] = inout[offset_in + (7 * thread + 37) * stride_in];
    R[6] = inout[offset_in + (7 * thread + 44) * stride_in];
    R[7] = inout[offset_in + (7 * thread + 51) * stride_in];
    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
    lds[offset_lds + ((7 * thread + 2)) * 8 + 0] = R[0];
    lds[offset_lds + ((7 * thread + 2)) * 8 + 1] = R[1];
    lds[offset_lds + ((7 * thread + 2)) * 8 + 2] = R[2];
    lds[offset_lds + ((7 * thread + 2)) * 8 + 3] = R[3];
    lds[offset_lds + ((7 * thread + 2)) * 8 + 4] = R[4];
    lds[offset_lds + ((7 * thread + 2)) * 8 + 5] = R[5];
    lds[offset_lds + ((7 * thread + 2)) * 8 + 6] = R[6];
    lds[offset_lds + ((7 * thread + 2)) * 8 + 7] = R[7];

    R[8]  = inout[offset_in + (7 * thread + 3) * stride_in];
    R[9]  = inout[offset_in + (7 * thread + 10) * stride_in];
    R[10] = inout[offset_in + (7 * thread + 17) * stride_in];
    R[11] = inout[offset_in + (7 * thread + 24) * stride_in];
    R[12] = inout[offset_in + (7 * thread + 31) * stride_in];
    R[13] = inout[offset_in + (7 * thread + 38) * stride_in];
    R[14] = inout[offset_in + (7 * thread + 45) * stride_in];
    R[15] = inout[offset_in + (7 * thread + 52) * stride_in];
    FwdRad8B1(&R[8], &R[9], &R[10], &R[11], &R[12], &R[13], &R[14], &R[15]);
    lds[offset_lds + ((7 * thread + 3)) * 8 + 0] = R[8];
    lds[offset_lds + ((7 * thread + 3)) * 8 + 1] = R[9];
    lds[offset_lds + ((7 * thread + 3)) * 8 + 2] = R[10];
    lds[offset_lds + ((7 * thread + 3)) * 8 + 3] = R[11];
    lds[offset_lds + ((7 * thread + 3)) * 8 + 4] = R[12];
    lds[offset_lds + ((7 * thread + 3)) * 8 + 5] = R[13];
    lds[offset_lds + ((7 * thread + 3)) * 8 + 6] = R[14];
    lds[offset_lds + ((7 * thread + 3)) * 8 + 7] = R[15];

    R[0] = inout[offset_in + (7 * thread + 4) * stride_in];
    R[1] = inout[offset_in + (7 * thread + 11) * stride_in];
    R[2] = inout[offset_in + (7 * thread + 18) * stride_in];
    R[3] = inout[offset_in + (7 * thread + 25) * stride_in];
    R[4] = inout[offset_in + (7 * thread + 32) * stride_in];
    R[5] = inout[offset_in + (7 * thread + 39) * stride_in];
    R[6] = inout[offset_in + (7 * thread + 46) * stride_in];
    R[7] = inout[offset_in + (7 * thread + 53) * stride_in];
    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
    lds[offset_lds + ((7 * thread + 4)) * 8 + 0] = R[0];
    lds[offset_lds + ((7 * thread + 4)) * 8 + 1] = R[1];
    lds[offset_lds + ((7 * thread + 4)) * 8 + 2] = R[2];
    lds[offset_lds + ((7 * thread + 4)) * 8 + 3] = R[3];
    lds[offset_lds + ((7 * thread + 4)) * 8 + 4] = R[4];
    lds[offset_lds + ((7 * thread + 4)) * 8 + 5] = R[5];
    lds[offset_lds + ((7 * thread + 4)) * 8 + 6] = R[6];
    lds[offset_lds + ((7 * thread + 4)) * 8 + 7] = R[7];

    R[8]  = inout[offset_in + (7 * thread + 5) * stride_in];
    R[9]  = inout[offset_in + (7 * thread + 12) * stride_in];
    R[10] = inout[offset_in + (7 * thread + 19) * stride_in];
    R[11] = inout[offset_in + (7 * thread + 26) * stride_in];
    R[12] = inout[offset_in + (7 * thread + 33) * stride_in];
    R[13] = inout[offset_in + (7 * thread + 40) * stride_in];
    R[14] = inout[offset_in + (7 * thread + 47) * stride_in];
    R[15] = inout[offset_in + (7 * thread + 54) * stride_in];
    FwdRad8B1(&R[8], &R[9], &R[10], &R[11], &R[12], &R[13], &R[14], &R[15]);
    lds[offset_lds + ((7 * thread + 5)) * 8 + 0] = R[8];
    lds[offset_lds + ((7 * thread + 5)) * 8 + 1] = R[9];
    lds[offset_lds + ((7 * thread + 5)) * 8 + 2] = R[10];
    lds[offset_lds + ((7 * thread + 5)) * 8 + 3] = R[11];
    lds[offset_lds + ((7 * thread + 5)) * 8 + 4] = R[12];
    lds[offset_lds + ((7 * thread + 5)) * 8 + 5] = R[13];
    lds[offset_lds + ((7 * thread + 5)) * 8 + 6] = R[14];
    lds[offset_lds + ((7 * thread + 5)) * 8 + 7] = R[15];

    R[0] = inout[offset_in + (7 * thread + 6) * stride_in];
    R[1] = inout[offset_in + (7 * thread + 13) * stride_in];
    R[2] = inout[offset_in + (7 * thread + 20) * stride_in];
    R[3] = inout[offset_in + (7 * thread + 27) * stride_in];
    R[4] = inout[offset_in + (7 * thread + 34) * stride_in];
    R[5] = inout[offset_in + (7 * thread + 41) * stride_in];
    R[6] = inout[offset_in + (7 * thread + 48) * stride_in];
    R[7] = inout[offset_in + (7 * thread + 55) * stride_in];
    FwdRad8B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7]);
    lds[offset_lds + ((7 * thread + 6)) * 8 + 0] = R[0];
    lds[offset_lds + ((7 * thread + 6)) * 8 + 1] = R[1];
    lds[offset_lds + ((7 * thread + 6)) * 8 + 2] = R[2];
    lds[offset_lds + ((7 * thread + 6)) * 8 + 3] = R[3];
    lds[offset_lds + ((7 * thread + 6)) * 8 + 4] = R[4];
    lds[offset_lds + ((7 * thread + 6)) * 8 + 5] = R[5];
    lds[offset_lds + ((7 * thread + 6)) * 8 + 6] = R[6];
    lds[offset_lds + ((7 * thread + 6)) * 8 + 7] = R[7];

    R[0] = lds[offset_lds + 8 * thread + 0];
    R[1] = lds[offset_lds + 8 * thread + 8];
    R[2] = lds[offset_lds + 8 * thread + 16];
    R[3] = lds[offset_lds + 8 * thread + 24];
    R[4] = lds[offset_lds + 8 * thread + 32];
    R[5] = lds[offset_lds + 8 * thread + 40];
    R[6] = lds[offset_lds + 8 * thread + 48];
    W    = twiddles[7 + 6 * ((8 * thread + 0) % 8)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    W    = twiddles[8 + 6 * ((8 * thread + 0) % 8)];
    t.x  = W.x * R[2].x - W.y * R[2].y;
    t.y  = W.y * R[2].x + W.x * R[2].y;
    R[2] = t;
    W    = twiddles[9 + 6 * ((8 * thread + 0) % 8)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    W    = twiddles[10 + 6 * ((8 * thread + 0) % 8)];
    t.x  = W.x * R[4].x - W.y * R[4].y;
    t.y  = W.y * R[4].x + W.x * R[4].y;
    R[4] = t;
    W    = twiddles[11 + 6 * ((8 * thread + 0) % 8)];
    t.x  = W.x * R[5].x - W.y * R[5].y;
    t.y  = W.y * R[5].x + W.x * R[5].y;
    R[5] = t;
    W    = twiddles[12 + 6 * ((8 * thread + 0) % 8)];
    t.x  = W.x * R[6].x - W.y * R[6].y;
    t.y  = W.y * R[6].x + W.x * R[6].y;
    R[6] = t;
    FwdRad7B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6]);
    inout[offset_out + (8 * thread + 0) * stride_out]  = R[0];
    inout[offset_out + (8 * thread + 8) * stride_out]  = R[1];
    inout[offset_out + (8 * thread + 16) * stride_out] = R[2];
    inout[offset_out + (8 * thread + 24) * stride_out] = R[3];
    inout[offset_out + (8 * thread + 32) * stride_out] = R[4];
    inout[offset_out + (8 * thread + 40) * stride_out] = R[5];
    inout[offset_out + (8 * thread + 48) * stride_out] = R[6];

    R[7]  = lds[offset_lds + 8 * thread + 1];
    R[8]  = lds[offset_lds + 8 * thread + 9];
    R[9]  = lds[offset_lds + 8 * thread + 17];
    R[10] = lds[offset_lds + 8 * thread + 25];
    R[11] = lds[offset_lds + 8 * thread + 33];
    R[12] = lds[offset_lds + 8 * thread + 41];
    R[13] = lds[offset_lds + 8 * thread + 49];
    W     = twiddles[7 + 6 * ((8 * thread + 1) % 8)];
    t.x   = W.x * R[8].x - W.y * R[8].y;
    t.y   = W.y * R[8].x + W.x * R[8].y;
    R[8]  = t;
    W     = twiddles[8 + 6 * ((8 * thread + 1) % 8)];
    t.x   = W.x * R[9].x - W.y * R[9].y;
    t.y   = W.y * R[9].x + W.x * R[9].y;
    R[9]  = t;
    W     = twiddles[9 + 6 * ((8 * thread + 1) % 8)];
    t.x   = W.x * R[10].x - W.y * R[10].y;
    t.y   = W.y * R[10].x + W.x * R[10].y;
    R[10] = t;
    W     = twiddles[10 + 6 * ((8 * thread + 1) % 8)];
    t.x   = W.x * R[11].x - W.y * R[11].y;
    t.y   = W.y * R[11].x + W.x * R[11].y;
    R[11] = t;
    W     = twiddles[11 + 6 * ((8 * thread + 1) % 8)];
    t.x   = W.x * R[12].x - W.y * R[12].y;
    t.y   = W.y * R[12].x + W.x * R[12].y;
    R[12] = t;
    W     = twiddles[12 + 6 * ((8 * thread + 1) % 8)];
    t.x   = W.x * R[13].x - W.y * R[13].y;
    t.y   = W.y * R[13].x + W.x * R[13].y;
    R[13] = t;
    FwdRad7B1(&R[7], &R[8], &R[9], &R[10], &R[11], &R[12], &R[13]);
    inout[offset_out + (8 * thread + 1) * stride_out]  = R[7];
    inout[offset_out + (8 * thread + 9) * stride_out]  = R[8];
    inout[offset_out + (8 * thread + 17) * stride_out] = R[9];
    inout[offset_out + (8 * thread + 25) * stride_out] = R[10];
    inout[offset_out + (8 * thread + 33) * stride_out] = R[11];
    inout[offset_out + (8 * thread + 41) * stride_out] = R[12];
    inout[offset_out + (8 * thread + 49) * stride_out] = R[13];

    R[0] = lds[offset_lds + 8 * thread + 2];
    R[1] = lds[offset_lds + 8 * thread + 10];
    R[2] = lds[offset_lds + 8 * thread + 18];
    R[3] = lds[offset_lds + 8 * thread + 26];
    R[4] = lds[offset_lds + 8 * thread + 34];
    R[5] = lds[offset_lds + 8 * thread + 42];
    R[6] = lds[offset_lds + 8 * thread + 50];
    W    = twiddles[7 + 6 * ((8 * thread + 2) % 8)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    W    = twiddles[8 + 6 * ((8 * thread + 2) % 8)];
    t.x  = W.x * R[2].x - W.y * R[2].y;
    t.y  = W.y * R[2].x + W.x * R[2].y;
    R[2] = t;
    W    = twiddles[9 + 6 * ((8 * thread + 2) % 8)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    W    = twiddles[10 + 6 * ((8 * thread + 2) % 8)];
    t.x  = W.x * R[4].x - W.y * R[4].y;
    t.y  = W.y * R[4].x + W.x * R[4].y;
    R[4] = t;
    W    = twiddles[11 + 6 * ((8 * thread + 2) % 8)];
    t.x  = W.x * R[5].x - W.y * R[5].y;
    t.y  = W.y * R[5].x + W.x * R[5].y;
    R[5] = t;
    W    = twiddles[12 + 6 * ((8 * thread + 2) % 8)];
    t.x  = W.x * R[6].x - W.y * R[6].y;
    t.y  = W.y * R[6].x + W.x * R[6].y;
    R[6] = t;
    FwdRad7B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6]);
    inout[offset_out + (8 * thread + 2) * stride_out]  = R[0];
    inout[offset_out + (8 * thread + 10) * stride_out] = R[1];
    inout[offset_out + (8 * thread + 18) * stride_out] = R[2];
    inout[offset_out + (8 * thread + 26) * stride_out] = R[3];
    inout[offset_out + (8 * thread + 34) * stride_out] = R[4];
    inout[offset_out + (8 * thread + 42) * stride_out] = R[5];
    inout[offset_out + (8 * thread + 50) * stride_out] = R[6];

    R[7]  = lds[offset_lds + 8 * thread + 3];
    R[8]  = lds[offset_lds + 8 * thread + 11];
    R[9]  = lds[offset_lds + 8 * thread + 19];
    R[10] = lds[offset_lds + 8 * thread + 27];
    R[11] = lds[offset_lds + 8 * thread + 35];
    R[12] = lds[offset_lds + 8 * thread + 43];
    R[13] = lds[offset_lds + 8 * thread + 51];
    W     = twiddles[7 + 6 * ((8 * thread + 3) % 8)];
    t.x   = W.x * R[8].x - W.y * R[8].y;
    t.y   = W.y * R[8].x + W.x * R[8].y;
    R[8]  = t;
    W     = twiddles[8 + 6 * ((8 * thread + 3) % 8)];
    t.x   = W.x * R[9].x - W.y * R[9].y;
    t.y   = W.y * R[9].x + W.x * R[9].y;
    R[9]  = t;
    W     = twiddles[9 + 6 * ((8 * thread + 3) % 8)];
    t.x   = W.x * R[10].x - W.y * R[10].y;
    t.y   = W.y * R[10].x + W.x * R[10].y;
    R[10] = t;
    W     = twiddles[10 + 6 * ((8 * thread + 3) % 8)];
    t.x   = W.x * R[11].x - W.y * R[11].y;
    t.y   = W.y * R[11].x + W.x * R[11].y;
    R[11] = t;
    W     = twiddles[11 + 6 * ((8 * thread + 3) % 8)];
    t.x   = W.x * R[12].x - W.y * R[12].y;
    t.y   = W.y * R[12].x + W.x * R[12].y;
    R[12] = t;
    W     = twiddles[12 + 6 * ((8 * thread + 3) % 8)];
    t.x   = W.x * R[13].x - W.y * R[13].y;
    t.y   = W.y * R[13].x + W.x * R[13].y;
    R[13] = t;
    FwdRad7B1(&R[7], &R[8], &R[9], &R[10], &R[11], &R[12], &R[13]);
    inout[offset_out + (8 * thread + 3) * stride_out]  = R[7];
    inout[offset_out + (8 * thread + 11) * stride_out] = R[8];
    inout[offset_out + (8 * thread + 19) * stride_out] = R[9];
    inout[offset_out + (8 * thread + 27) * stride_out] = R[10];
    inout[offset_out + (8 * thread + 35) * stride_out] = R[11];
    inout[offset_out + (8 * thread + 43) * stride_out] = R[12];
    inout[offset_out + (8 * thread + 51) * stride_out] = R[13];

    R[0] = lds[offset_lds + 8 * thread + 4];
    R[1] = lds[offset_lds + 8 * thread + 12];
    R[2] = lds[offset_lds + 8 * thread + 20];
    R[3] = lds[offset_lds + 8 * thread + 28];
    R[4] = lds[offset_lds + 8 * thread + 36];
    R[5] = lds[offset_lds + 8 * thread + 44];
    R[6] = lds[offset_lds + 8 * thread + 52];
    W    = twiddles[7 + 6 * ((8 * thread + 4) % 8)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    W    = twiddles[8 + 6 * ((8 * thread + 4) % 8)];
    t.x  = W.x * R[2].x - W.y * R[2].y;
    t.y  = W.y * R[2].x + W.x * R[2].y;
    R[2] = t;
    W    = twiddles[9 + 6 * ((8 * thread + 4) % 8)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    W    = twiddles[10 + 6 * ((8 * thread + 4) % 8)];
    t.x  = W.x * R[4].x - W.y * R[4].y;
    t.y  = W.y * R[4].x + W.x * R[4].y;
    R[4] = t;
    W    = twiddles[11 + 6 * ((8 * thread + 4) % 8)];
    t.x  = W.x * R[5].x - W.y * R[5].y;
    t.y  = W.y * R[5].x + W.x * R[5].y;
    R[5] = t;
    W    = twiddles[12 + 6 * ((8 * thread + 4) % 8)];
    t.x  = W.x * R[6].x - W.y * R[6].y;
    t.y  = W.y * R[6].x + W.x * R[6].y;
    R[6] = t;
    FwdRad7B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6]);
    inout[offset_out + (8 * thread + 4) * stride_out]  = R[0];
    inout[offset_out + (8 * thread + 12) * stride_out] = R[1];
    inout[offset_out + (8 * thread + 20) * stride_out] = R[2];
    inout[offset_out + (8 * thread + 28) * stride_out] = R[3];
    inout[offset_out + (8 * thread + 36) * stride_out] = R[4];
    inout[offset_out + (8 * thread + 44) * stride_out] = R[5];
    inout[offset_out + (8 * thread + 52) * stride_out] = R[6];

    R[7]  = lds[offset_lds + 8 * thread + 5];
    R[8]  = lds[offset_lds + 8 * thread + 13];
    R[9]  = lds[offset_lds + 8 * thread + 21];
    R[10] = lds[offset_lds + 8 * thread + 29];
    R[11] = lds[offset_lds + 8 * thread + 37];
    R[12] = lds[offset_lds + 8 * thread + 45];
    R[13] = lds[offset_lds + 8 * thread + 53];
    W     = twiddles[7 + 6 * ((8 * thread + 5) % 8)];
    t.x   = W.x * R[8].x - W.y * R[8].y;
    t.y   = W.y * R[8].x + W.x * R[8].y;
    R[8]  = t;
    W     = twiddles[8 + 6 * ((8 * thread + 5) % 8)];
    t.x   = W.x * R[9].x - W.y * R[9].y;
    t.y   = W.y * R[9].x + W.x * R[9].y;
    R[9]  = t;
    W     = twiddles[9 + 6 * ((8 * thread + 5) % 8)];
    t.x   = W.x * R[10].x - W.y * R[10].y;
    t.y   = W.y * R[10].x + W.x * R[10].y;
    R[10] = t;
    W     = twiddles[10 + 6 * ((8 * thread + 5) % 8)];
    t.x   = W.x * R[11].x - W.y * R[11].y;
    t.y   = W.y * R[11].x + W.x * R[11].y;
    R[11] = t;
    W     = twiddles[11 + 6 * ((8 * thread + 5) % 8)];
    t.x   = W.x * R[12].x - W.y * R[12].y;
    t.y   = W.y * R[12].x + W.x * R[12].y;
    R[12] = t;
    W     = twiddles[12 + 6 * ((8 * thread + 5) % 8)];
    t.x   = W.x * R[13].x - W.y * R[13].y;
    t.y   = W.y * R[13].x + W.x * R[13].y;
    R[13] = t;
    FwdRad7B1(&R[7], &R[8], &R[9], &R[10], &R[11], &R[12], &R[13]);
    inout[offset_out + (8 * thread + 5) * stride_out]  = R[7];
    inout[offset_out + (8 * thread + 13) * stride_out] = R[8];
    inout[offset_out + (8 * thread + 21) * stride_out] = R[9];
    inout[offset_out + (8 * thread + 29) * stride_out] = R[10];
    inout[offset_out + (8 * thread + 37) * stride_out] = R[11];
    inout[offset_out + (8 * thread + 45) * stride_out] = R[12];
    inout[offset_out + (8 * thread + 53) * stride_out] = R[13];

    R[0] = lds[offset_lds + 8 * thread + 6];
    R[1] = lds[offset_lds + 8 * thread + 14];
    R[2] = lds[offset_lds + 8 * thread + 22];
    R[3] = lds[offset_lds + 8 * thread + 30];
    R[4] = lds[offset_lds + 8 * thread + 38];
    R[5] = lds[offset_lds + 8 * thread + 46];
    R[6] = lds[offset_lds + 8 * thread + 54];
    W    = twiddles[7 + 6 * ((8 * thread + 6) % 8)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    W    = twiddles[8 + 6 * ((8 * thread + 6) % 8)];
    t.x  = W.x * R[2].x - W.y * R[2].y;
    t.y  = W.y * R[2].x + W.x * R[2].y;
    R[2] = t;
    W    = twiddles[9 + 6 * ((8 * thread + 6) % 8)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    W    = twiddles[10 + 6 * ((8 * thread + 6) % 8)];
    t.x  = W.x * R[4].x - W.y * R[4].y;
    t.y  = W.y * R[4].x + W.x * R[4].y;
    R[4] = t;
    W    = twiddles[11 + 6 * ((8 * thread + 6) % 8)];
    t.x  = W.x * R[5].x - W.y * R[5].y;
    t.y  = W.y * R[5].x + W.x * R[5].y;
    R[5] = t;
    W    = twiddles[12 + 6 * ((8 * thread + 6) % 8)];
    t.x  = W.x * R[6].x - W.y * R[6].y;
    t.y  = W.y * R[6].x + W.x * R[6].y;
    R[6] = t;
    FwdRad7B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6]);
    inout[offset_out + (8 * thread + 6) * stride_out]  = R[0];
    inout[offset_out + (8 * thread + 14) * stride_out] = R[1];
    inout[offset_out + (8 * thread + 22) * stride_out] = R[2];
    inout[offset_out + (8 * thread + 30) * stride_out] = R[3];
    inout[offset_out + (8 * thread + 38) * stride_out] = R[4];
    inout[offset_out + (8 * thread + 46) * stride_out] = R[5];
    inout[offset_out + (8 * thread + 54) * stride_out] = R[6];

    R[7]  = lds[offset_lds + 8 * thread + 7];
    R[8]  = lds[offset_lds + 8 * thread + 15];
    R[9]  = lds[offset_lds + 8 * thread + 23];
    R[10] = lds[offset_lds + 8 * thread + 31];
    R[11] = lds[offset_lds + 8 * thread + 39];
    R[12] = lds[offset_lds + 8 * thread + 47];
    R[13] = lds[offset_lds + 8 * thread + 55];
    W     = twiddles[7 + 6 * ((8 * thread + 7) % 8)];
    t.x   = W.x * R[8].x - W.y * R[8].y;
    t.y   = W.y * R[8].x + W.x * R[8].y;
    R[8]  = t;
    W     = twiddles[8 + 6 * ((8 * thread + 7) % 8)];
    t.x   = W.x * R[9].x - W.y * R[9].y;
    t.y   = W.y * R[9].x + W.x * R[9].y;
    R[9]  = t;
    W     = twiddles[9 + 6 * ((8 * thread + 7) % 8)];
    t.x   = W.x * R[10].x - W.y * R[10].y;
    t.y   = W.y * R[10].x + W.x * R[10].y;
    R[10] = t;
    W     = twiddles[10 + 6 * ((8 * thread + 7) % 8)];
    t.x   = W.x * R[11].x - W.y * R[11].y;
    t.y   = W.y * R[11].x + W.x * R[11].y;
    R[11] = t;
    W     = twiddles[11 + 6 * ((8 * thread + 7) % 8)];
    t.x   = W.x * R[12].x - W.y * R[12].y;
    t.y   = W.y * R[12].x + W.x * R[12].y;
    R[12] = t;
    W     = twiddles[12 + 6 * ((8 * thread + 7) % 8)];
    t.x   = W.x * R[13].x - W.y * R[13].y;
    t.y   = W.y * R[13].x + W.x * R[13].y;
    R[13] = t;
    FwdRad7B1(&R[7], &R[8], &R[9], &R[10], &R[11], &R[12], &R[13]);
    inout[offset_out + (8 * thread + 7) * stride_out]  = R[7];
    inout[offset_out + (8 * thread + 15) * stride_out] = R[8];
    inout[offset_out + (8 * thread + 23) * stride_out] = R[9];
    inout[offset_out + (8 * thread + 31) * stride_out] = R[10];
    inout[offset_out + (8 * thread + 39) * stride_out] = R[11];
    inout[offset_out + (8 * thread + 47) * stride_out] = R[12];
    inout[offset_out + (8 * thread + 55) * stride_out] = R[13];
}

//
// gb - global buffer
// twiddles - twiddle table
// nbpt - number of batches per thread
//
#define GLBIDX(b, i) (256 * nbpt * blockIdx.x + 256 * b + i)
#define LCLIDX(b, i) (256 * b + i)

template <int nbpt>
__global__ void fft_256_fwd_batchfirst(dtype* gb, dtype* twiddles)
{
    dtype __shared__ lds[256 * nbpt];
    dtype            X[4], W[4];
    dtype            t;

    int me = threadIdx.x;
    int idx;

    for(int b = 0; b < nbpt; ++b)
    {
        idx  = GLBIDX(b, me);
        X[0] = gb[idx + 0];
        X[1] = gb[idx + 64];
        X[2] = gb[idx + 128];
        X[3] = gb[idx + 192];

        FwdRad4(&X[0], &X[1], &X[2], &X[3]);

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

        FwdRad4(&X[0], &X[1], &X[2], &X[3]);

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

        FwdRad4(&X[0], &X[1], &X[2], &X[3]);

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

        FwdRad4(&X[0], &X[1], &X[2], &X[3]);

        idx           = GLBIDX(b, me);
        gb[idx + 0]   = X[0];
        gb[idx + 64]  = X[1];
        gb[idx + 128] = X[2];
        gb[idx + 192] = X[3];
    }
}

fft_result2 fft_stockham_gpu(vector<dtype> const& x, int nx, int nbatch, int nbpt)
{
    vector<float> times;
    int           ntrials = 10;

    auto z = copy(x);

    dtype* X;
    HIP_CHECK(hipMalloc(&X, nx * nbatch * sizeof(dtype)));
    HIP_CHECK(hipMemcpy(X, z.data(), nx * nbatch * sizeof(dtype), hipMemcpyHostToDevice));

    dtype* twiddles;
    HIP_CHECK(hipMalloc(&twiddles, (nx - 1) * sizeof(dtype)));

    GPUTimer total;
    total.tic();
    vector<int> factors;
    if(nx == 256)
    {
        factors = {4, 4, 4, 4};
    }
    else if(nx == 56)
    {
      //        factors = {7, 2, 2, 2};
      factors = {8, 7};
    }

    int* d_factors;
    HIP_CHECK(hipMalloc(&d_factors, factors.size() * sizeof(int)));
    HIP_CHECK(
        hipMemcpy(d_factors, factors.data(), factors.size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, nx - 1>>>(nx - 1, twiddles, factors.size(), d_factors);
    HIP_CHECK(hipFree(d_factors));

    if(false)
    {
        vector<dtype> t(nx - 1);
        HIP_CHECK(hipMemcpy(t.data(), twiddles, t.size() * sizeof(dtype), hipMemcpyDeviceToHost));
        for(int i = 0; i < nx - 1; ++i)
            cout << i << " " << t[i].x << " " << t[i].y << endl;
    }

    GPUTimer timer;
    for(int n = 0; n <= ntrials; ++n)
    {
        timer.tic();
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
        else if(nx == 56)
        {
          //            fft_56_fwd<<<nbatch, 4>>>(X, twiddles);
          //          fft_56_fwd_fat<<<nbatch, 4>>>(X, twiddles, 1, 1, 0, 0, 0);
          fft_56_fwd_fat56<<<nbatch, 4>>>(X, twiddles, 1, 1, 0, 0, 0);
        }
        timer.toc();
        if(n > 0)
            times.push_back(timer.elapsed());
        if(n == 0)
            HIP_CHECK(hipMemcpy(z.data(), X, nx * nbatch * sizeof(dtype), hipMemcpyDeviceToHost));
    }
    total.toc();

    HIP_CHECK(hipFree(twiddles));
    HIP_CHECK(hipFree(X));

    return {average(times), total.elapsed(), move(z)};
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
        if(dx * dx + dy * dy > 1.e-7)
        {
            cout << n << " " << sqrt(dx * dx + dy * dy) << " " << z1[n].x << " " << z2[n].x << endl;
        }
        d += dx * dx + dy * dy;
        r += z1[n].x * z1[n].x + z1[n].y * z1[n].y;
    }
    return sqrt(d) / sqrt(r);
}

//
// Some tests!
//
void test1d(size_t n, size_t nbatch, size_t nbpt)
{
    double GiB = double(n * nbatch * sizeof(dtype)) / 1024 / 1024 / 1024;
    cout << "# 1d test" << endl;
    cout << "1d input length: " << n << " (" << nbatch << ")" << endl;
    cout << "1d input size:   " << GiB << "GiB" << endl;

    auto x = random_vector(n * nbatch);

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
    if(argc > 1)
        length = stoi(argv[1]);
    if(argc > 2)
        nbatch = stoi(argv[2]);
    if(argc > 3)
        nbpt = stoi(argv[3]);

    test1d(length, nbatch, nbpt);
}

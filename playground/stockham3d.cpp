#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>

#include <fftw3.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#include <rocrand/rocrand.hpp>

#include "timer.h"

#define HIP_CHECK(r)    \
    if(r != hipSuccess) \
        return {};

using namespace std;

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

template <class T>
struct real_type;

template <>
struct real_type<float4>
{
    typedef float type;
};

template <>
struct real_type<double4>
{
    typedef double type;
};

template <>
struct real_type<float2>
{
    typedef float type;
};

template <>
struct real_type<double2>
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
    vector<T> x(n);

    rocrand_cpp::random_device                             rd;
    rocrand_cpp::mtgp32                                    engine(rd());
    rocrand_cpp::normal_distribution<float>                dist(0.0, 1.5);
    rocrand_cpp::uniform_real_distribution<real_type_t<T>> distribution;

    real_type_t<T>* rbuf;
    HIP_CHECK(hipMalloc(&rbuf, 2 * n * sizeof(real_type_t<T>)));
    distribution(engine, rbuf, 2 * n);
    HIP_CHECK(hipMemcpy(x.data(), rbuf, 2 * n * sizeof(real_type_t<T>), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(rbuf));

    return x;
}

//
// Copy helper
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
    fft_fftw(vector<hipDoubleComplex> const& x, size_t nx, size_t nbatch)
{
    auto z = copy(x);
    // clang-format off
    int n[1] = { int(nx) };
    auto p = fftw_plan_many_dft(1, n, nbatch,
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

pair<float, vector<hipComplex>> fft_fftw(vector<hipComplex> const& x, size_t nx, size_t nbatch)
{
    auto z = copy(x);
    // clang-format off
    int n[1] = { int(nx) };
    auto p = fftwf_plan_many_dft(1, n, nbatch,
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

pair<float, vector<hipDoubleComplex>>
    fft_fftw(vector<hipDoubleComplex> const& x, const size_t nx, size_t ny, size_t nbatch)
{
    auto z = copy(x);
    // clang-format off
    int n[2] = { int(ny), int(nx) };
    auto p = fftw_plan_many_dft(2, n, nbatch,
                                (fftw_complex*) z.data(), nullptr, 1, nx*ny,
                                (fftw_complex*) z.data(), nullptr, 1, nx*ny,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    CPUTimer timer;
    timer.tic();
    fftw_execute(p);
    timer.toc();
    fftw_destroy_plan(p);
    return {timer.elapsed(), move(z)};
}

pair<float, vector<hipComplex>>
    fft_fftw(vector<hipComplex> const& x, size_t nx, size_t ny, size_t nbatch)
{
    auto z = copy(x);
    // clang-format off
    int n[2] = { int(ny), int(nx) };
    auto p = fftwf_plan_many_dft(2, n, nbatch,
                                (fftwf_complex*) z.data(), nullptr, 1, nx*ny,
                                (fftwf_complex*) z.data(), nullptr, 1, nx*ny,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    CPUTimer timer;
    timer.tic();
    fftwf_execute(p);
    timer.toc();
    fftwf_destroy_plan(p);
    return {timer.elapsed(), move(z)};
}

pair<float, vector<hipDoubleComplex>>
    fft_fftw(vector<hipDoubleComplex> const& x, size_t nx, size_t ny, size_t nz, size_t nbatch)
{
    auto z = copy(x);
    // clang-format off
    int n[3] = { int(nz), int(ny), int(nx) };
    auto p = fftw_plan_many_dft(3, n, nbatch,
                                (fftw_complex*) z.data(), nullptr, 1, nx*ny*nz,
                                (fftw_complex*) z.data(), nullptr, 1, nx*ny*nz,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    // clang-format on
    CPUTimer timer;
    timer.tic();
    fftw_execute(p);
    timer.toc();
    fftw_destroy_plan(p);
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

template <typename scalar_type, StrideBin sb>
__device__ void forward_length75_SBRR_device(real_type_t<scalar_type>* __restrict__ lds,
                                             const scalar_type* __restrict__ twiddles,
                                             size_t       stride_lds,
                                             unsigned int offset_lds,
                                             bool         write,
                                             scalar_type* __restrict__ buf_in,
                                             scalar_type* __restrict__ buf_out,
                                             size_t stride_in,
                                             size_t stride_out,
                                             size_t offset_in,
                                             size_t offset_out)
{
    unsigned int thread;
    scalar_type  R[15];
    scalar_type  W;
    scalar_type  t;
    const size_t lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    thread               = threadIdx.x % 25;

    // pass 0, width 5
    // using 25 threads we need to do 15 radix-5 butterflies
    // therefore each thread will do 0.6 butterflies
    __syncthreads();
    if(write && thread < 15)
    {
        R[0] = buf_in[offset_in + (((thread + 0 + 0) + 0)) * stride_in];
        R[1] = buf_in[offset_in + (((thread + 0 + 0) + 15)) * stride_in];
        R[2] = buf_in[offset_in + (((thread + 0 + 0) + 30)) * stride_in];
        R[3] = buf_in[offset_in + (((thread + 0 + 0) + 45)) * stride_in];
        R[4] = buf_in[offset_in + (((thread + 0 + 0) + 60)) * stride_in];
    }

    FwdRad5B1(&R[0], &R[1], &R[2], &R[3], &R[4]);

    // more than enough threads, some do nothing
    if(write && thread < 15)
    {
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 0) * lstride]
            = R[0].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 1) * lstride]
            = R[1].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 2) * lstride]
            = R[2].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 3) * lstride]
            = R[3].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 4) * lstride]
            = R[4].x;
    }

    __syncthreads();
    // more than enough threads, some do nothing
    if(write && thread < 15)
    {
        R[0].x = lds[offset_lds + ((thread + 0 + 0) + 0) * lstride];
        R[1].x = lds[offset_lds + ((thread + 0 + 0) + 15) * lstride];
        R[2].x = lds[offset_lds + ((thread + 0 + 0) + 30) * lstride];
        R[3].x = lds[offset_lds + ((thread + 0 + 0) + 45) * lstride];
        R[4].x = lds[offset_lds + ((thread + 0 + 0) + 60) * lstride];
    }

    __syncthreads();
    // more than enough threads, some do nothing
    if(write && thread < 15)
    {
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 0) * lstride]
            = R[0].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 1) * lstride]
            = R[1].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 2) * lstride]
            = R[2].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 3) * lstride]
            = R[3].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 4) * lstride]
            = R[4].y;
    }

    __syncthreads();
    // more than enough threads, some do nothing
    if(write && thread < 15)
    {
        R[0].y = lds[offset_lds + ((thread + 0 + 0) + 0) * lstride];
        R[1].y = lds[offset_lds + ((thread + 0 + 0) + 15) * lstride];
        R[2].y = lds[offset_lds + ((thread + 0 + 0) + 30) * lstride];
        R[3].y = lds[offset_lds + ((thread + 0 + 0) + 45) * lstride];
        R[4].y = lds[offset_lds + ((thread + 0 + 0) + 60) * lstride];
    }

    __syncthreads();

    // pass 1, width 5
    // using 25 threads we need to do 15 radix-5 butterflies
    // therefore each thread will do 0.6 butterflies
    __syncthreads();
    W    = twiddles[4 + 4 * ((thread + 0 + 0) % 5)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    W    = twiddles[5 + 4 * ((thread + 0 + 0) % 5)];
    t.x  = W.x * R[2].x - W.y * R[2].y;
    t.y  = W.y * R[2].x + W.x * R[2].y;
    R[2] = t;
    W    = twiddles[6 + 4 * ((thread + 0 + 0) % 5)];
    t.x  = W.x * R[3].x - W.y * R[3].y;
    t.y  = W.y * R[3].x + W.x * R[3].y;
    R[3] = t;
    W    = twiddles[7 + 4 * ((thread + 0 + 0) % 5)];
    t.x  = W.x * R[4].x - W.y * R[4].y;
    t.y  = W.y * R[4].x + W.x * R[4].y;
    R[4] = t;

    FwdRad5B1(&R[0], &R[1], &R[2], &R[3], &R[4]);

    // more than enough threads, some do nothing
    if(write && thread < 15)
    {
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 0) * lstride]
            = R[0].x;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 5) * lstride]
            = R[1].x;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 10) * lstride]
            = R[2].x;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 15) * lstride]
            = R[3].x;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 20) * lstride]
            = R[4].x;
    }

    __syncthreads();
    R[0].x = lds[offset_lds + ((thread + 0 + 0) + 0) * lstride];
    R[1].x = lds[offset_lds + ((thread + 0 + 0) + 25) * lstride];
    R[2].x = lds[offset_lds + ((thread + 0 + 0) + 50) * lstride];

    __syncthreads();
    // more than enough threads, some do nothing
    if(write && thread < 15)
    {
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 0) * lstride]
            = R[0].y;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 5) * lstride]
            = R[1].y;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 10) * lstride]
            = R[2].y;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 15) * lstride]
            = R[3].y;
        lds[offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 20) * lstride]
            = R[4].y;
    }

    __syncthreads();
    R[0].y = lds[offset_lds + ((thread + 0 + 0) + 0) * lstride];
    R[1].y = lds[offset_lds + ((thread + 0 + 0) + 25) * lstride];
    R[2].y = lds[offset_lds + ((thread + 0 + 0) + 50) * lstride];

    __syncthreads();

    // pass 2, width 3
    // using 25 threads we need to do 25 radix-3 butterflies
    // therefore each thread will do 1.0 butterflies
    __syncthreads();
    W    = twiddles[24 + 2 * ((thread + 0 + 0) % 25)];
    t.x  = W.x * R[1].x - W.y * R[1].y;
    t.y  = W.y * R[1].x + W.x * R[1].y;
    R[1] = t;
    W    = twiddles[25 + 2 * ((thread + 0 + 0) % 25)];
    t.x  = W.x * R[2].x - W.y * R[2].y;
    t.y  = W.y * R[2].x + W.x * R[2].y;
    R[2] = t;

    FwdRad3B1(&R[0], &R[1], &R[2]);

    if(write)
    {
        // buf_out[offset_out
        //         + (((thread + 0 + 0) / 25) * 75 + (thread + 0 + 0) % 25 + 0) * stride_out]
        //     = R[0];
        // buf_out[offset_out
        //         + (((thread + 0 + 0) / 25) * 75 + (thread + 0 + 0) % 25 + 25) * stride_out]
        //     = R[1];
        // buf_out[offset_out
        //         + (((thread + 0 + 0) / 25) * 75 + (thread + 0 + 0) % 25 + 50) * stride_out]
        //     = R[2];
        buf_out[offset_out + (thread + 0) * stride_out]  = R[0];
        buf_out[offset_out + (thread + 25) * stride_out] = R[1];
        buf_out[offset_out + (thread + 50) * stride_out] = R[2];
    }
}

template <typename scalar_type>
__global__
    __launch_bounds__(250) void forward_length75_SBRR(const scalar_type* __restrict__ twiddles,
                                                      const int dim,
                                                      const int trans_dim,
                                                      const size_t* __restrict__ lengths,
                                                      const size_t* __restrict__ stride,
                                                      const size_t* __restrict__ permute,
                                                      const size_t nbatch,
                                                      scalar_type* __restrict__ buf_in,
                                                      scalar_type* __restrict__ buf_out)
{
    // this kernel:
    //   uses 25 threads per transform
    //   does 10 transforms per thread block
    // therefore it should be called with 250 threads per thread block
    extern __shared__ unsigned char __align__(sizeof(scalar_type)) lds_uchar[];
    real_type_t<scalar_type>* __restrict__ lds
        = reinterpret_cast<real_type_t<scalar_type>*>(lds_uchar);
    size_t       offset_in  = 0;
    size_t       offset_out = 0;
    size_t       stride_in;
    size_t       stride_out;
    unsigned int offset_lds = 0;
    size_t       stride_lds = 1;
    size_t       transform;
    bool         write;

    int indexes[8];

    // offsets
    size_t remaining;
    transform = blockIdx.x * 10 + threadIdx.x / 25;
    remaining = transform;
    for(int d = 0; d < dim; ++d)
    {
        if(d == trans_dim)
            continue;
        indexes[d] = remaining % lengths[d];
        remaining  = remaining / lengths[d];
    }
    indexes[trans_dim] = 0;
    indexes[dim]       = remaining;

    offset_in  = 0;
    offset_out = 0;
    for(int d = 0; d <= dim; ++d)
    {
        offset_in += indexes[d] * stride[d];
        offset_out += indexes[d] * stride[permute[d]];
    }
    stride_in  = stride[trans_dim];
    stride_out = stride[permute[trans_dim]];

    offset_lds = 75 * (transform % 10);

    write = true;
    if(indexes[dim] >= nbatch)
    {
        write = false;
    }

    // transform
    forward_length75_SBRR_device<scalar_type, SB_UNIT>(lds,
                                                       twiddles,
                                                       stride_lds,
                                                       offset_lds,
                                                       write,
                                                       buf_in,
                                                       buf_out,
                                                       stride_in,
                                                       stride_out,
                                                       offset_in,
                                                       offset_out);
}

template <typename scalar_type>
void forward_length75_launch(scalar_type*       buf_in,
                             scalar_type*       buf_out,
                             const int          dim,
                             const int          trans_dim,
                             size_t*            lengths,
                             size_t             nbatch,
                             const scalar_type* twiddles,
                             size_t*            kargs)
{
    size_t nblocks;
    size_t ntransforms = 1;
    for(int d = 0; d < dim; ++d)
    {
        if(d == trans_dim)
            continue;
        ntransforms *= lengths[d];
    }
    ntransforms *= nbatch;
    nblocks = (ntransforms + 9) / 10;
    forward_length75_SBRR<scalar_type><<<nblocks, 250, 8000>>>(
        twiddles, dim, trans_dim, kargs, kargs + dim, kargs + 2 * dim + 1, nbatch, buf_in, buf_out);
}

template <typename scalar_type, StrideBin sb>
__device__ void forward_length55_SBRR_device(real_type_t<scalar_type>* __restrict__ lds,
                                             const scalar_type* __restrict__ twiddles,
                                             size_t       stride_lds,
                                             unsigned int offset_lds,
                                             bool         write,
                                             scalar_type* __restrict__ buf_in,
                                             scalar_type* __restrict__ buf_out,
                                             const size_t stride_in,
                                             const size_t stride_out,
                                             size_t       offset_in,
                                             size_t       offset_out)
{
    unsigned int thread;
    scalar_type  R[33];
    scalar_type  W;
    scalar_type  t;
    const size_t lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    thread               = threadIdx.x % 25;

    // pass 0, width 5
    // using 25 threads we need to do 11 radix-5 butterflies
    // therefore each thread will do 0.44 butterflies
    __syncthreads();

    if(write && thread < 11)
    {
        R[0] = buf_in[offset_in + (((thread + 0 + 0) + 0)) * stride_in];
        R[1] = buf_in[offset_in + (((thread + 0 + 0) + 11)) * stride_in];
        R[2] = buf_in[offset_in + (((thread + 0 + 0) + 22)) * stride_in];
        R[3] = buf_in[offset_in + (((thread + 0 + 0) + 33)) * stride_in];
        R[4] = buf_in[offset_in + (((thread + 0 + 0) + 44)) * stride_in];
    }

    FwdRad5B1(&R[0], &R[1], &R[2], &R[3], &R[4]);

    if(write && thread < 11)
    {
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 0) * lstride]
            = R[0].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 1) * lstride]
            = R[1].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 2) * lstride]
            = R[2].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 3) * lstride]
            = R[3].x;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 4) * lstride]
            = R[4].x;
    }

    __syncthreads();
    if(write && thread < 5)
    {
        R[0].x  = lds[offset_lds + ((thread + 0 + 0) + 0) * lstride];
        R[1].x  = lds[offset_lds + ((thread + 0 + 0) + 5) * lstride];
        R[2].x  = lds[offset_lds + ((thread + 0 + 0) + 10) * lstride];
        R[3].x  = lds[offset_lds + ((thread + 0 + 0) + 15) * lstride];
        R[4].x  = lds[offset_lds + ((thread + 0 + 0) + 20) * lstride];
        R[5].x  = lds[offset_lds + ((thread + 0 + 0) + 25) * lstride];
        R[6].x  = lds[offset_lds + ((thread + 0 + 0) + 30) * lstride];
        R[7].x  = lds[offset_lds + ((thread + 0 + 0) + 35) * lstride];
        R[8].x  = lds[offset_lds + ((thread + 0 + 0) + 40) * lstride];
        R[9].x  = lds[offset_lds + ((thread + 0 + 0) + 45) * lstride];
        R[10].x = lds[offset_lds + ((thread + 0 + 0) + 50) * lstride];
    }

    __syncthreads();

    if(write && thread < 11)
    {
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 0) * lstride]
            = R[0].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 1) * lstride]
            = R[1].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 2) * lstride]
            = R[2].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 3) * lstride]
            = R[3].y;
        lds[offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 4) * lstride]
            = R[4].y;
    }

    __syncthreads();
    // more than enough threads, some do nothing
    if(write && thread < 5)
    {
        R[0].y  = lds[offset_lds + ((thread + 0 + 0) + 0) * lstride];
        R[1].y  = lds[offset_lds + ((thread + 0 + 0) + 5) * lstride];
        R[2].y  = lds[offset_lds + ((thread + 0 + 0) + 10) * lstride];
        R[3].y  = lds[offset_lds + ((thread + 0 + 0) + 15) * lstride];
        R[4].y  = lds[offset_lds + ((thread + 0 + 0) + 20) * lstride];
        R[5].y  = lds[offset_lds + ((thread + 0 + 0) + 25) * lstride];
        R[6].y  = lds[offset_lds + ((thread + 0 + 0) + 30) * lstride];
        R[7].y  = lds[offset_lds + ((thread + 0 + 0) + 35) * lstride];
        R[8].y  = lds[offset_lds + ((thread + 0 + 0) + 40) * lstride];
        R[9].y  = lds[offset_lds + ((thread + 0 + 0) + 45) * lstride];
        R[10].y = lds[offset_lds + ((thread + 0 + 0) + 50) * lstride];
    }

    __syncthreads();

    // pass 1, width 11
    // using 25 threads we need to do 5 radix-11 butterflies
    // therefore each thread will do 0.2 butterflies
    __syncthreads();
    W     = twiddles[4 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[1].x - W.y * R[1].y;
    t.y   = W.y * R[1].x + W.x * R[1].y;
    R[1]  = t;
    W     = twiddles[5 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[2].x - W.y * R[2].y;
    t.y   = W.y * R[2].x + W.x * R[2].y;
    R[2]  = t;
    W     = twiddles[6 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[3].x - W.y * R[3].y;
    t.y   = W.y * R[3].x + W.x * R[3].y;
    R[3]  = t;
    W     = twiddles[7 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[4].x - W.y * R[4].y;
    t.y   = W.y * R[4].x + W.x * R[4].y;
    R[4]  = t;
    W     = twiddles[8 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[5].x - W.y * R[5].y;
    t.y   = W.y * R[5].x + W.x * R[5].y;
    R[5]  = t;
    W     = twiddles[9 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[6].x - W.y * R[6].y;
    t.y   = W.y * R[6].x + W.x * R[6].y;
    R[6]  = t;
    W     = twiddles[10 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[7].x - W.y * R[7].y;
    t.y   = W.y * R[7].x + W.x * R[7].y;
    R[7]  = t;
    W     = twiddles[11 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[8].x - W.y * R[8].y;
    t.y   = W.y * R[8].x + W.x * R[8].y;
    R[8]  = t;
    W     = twiddles[12 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[9].x - W.y * R[9].y;
    t.y   = W.y * R[9].x + W.x * R[9].y;
    R[9]  = t;
    W     = twiddles[13 + 10 * ((thread + 0 + 0) % 5)];
    t.x   = W.x * R[10].x - W.y * R[10].y;
    t.y   = W.y * R[10].x + W.x * R[10].y;
    R[10] = t;

    FwdRad11B1(&R[0], &R[1], &R[2], &R[3], &R[4], &R[5], &R[6], &R[7], &R[8], &R[9], &R[10]);

    // more than enough threads, some do nothing
    if(write && thread < 5)
    {
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 0) * stride_out]
            = R[0];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 5) * stride_out]
            = R[1];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 10) * stride_out]
            = R[2];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 15) * stride_out]
            = R[3];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 20) * stride_out]
            = R[4];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 25) * stride_out]
            = R[5];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 30) * stride_out]
            = R[6];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 35) * stride_out]
            = R[7];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 40) * stride_out]
            = R[8];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 45) * stride_out]
            = R[9];
        buf_out[offset_out + (((thread + 0 + 0) / 5) * 55 + (thread + 0 + 0) % 5 + 50) * stride_out]
            = R[10];
    }
}

template <typename scalar_type>
__global__
    __launch_bounds__(250) void forward_length55_SBRR(const scalar_type* __restrict__ twiddles,
                                                      const int dim,
                                                      const int trans_dim,
                                                      const size_t* __restrict__ lengths,
                                                      const size_t* __restrict__ stride,
                                                      const size_t* __restrict__ permute,
                                                      const size_t nbatch,
                                                      scalar_type* __restrict__ buf_in,
                                                      scalar_type* __restrict__ buf_out)
{
    // this kernel:
    //   uses 25 threads per transform
    //   does 10 transforms per thread block
    // therefore it should be called with 250 threads per thread block
    extern __shared__ unsigned char __align__(sizeof(scalar_type)) lds_uchar[];
    real_type_t<scalar_type>* __restrict__ lds
        = reinterpret_cast<real_type_t<scalar_type>*>(lds_uchar);
    size_t       offset_in  = 0;
    size_t       offset_out = 0;
    size_t       stride_in;
    size_t       stride_out;
    unsigned int offset_lds = 0;
    size_t       stride_lds = 1;
    size_t       transform;
    bool         write;

    int indexes[8];

    // offsets
    size_t remaining;
    transform = blockIdx.x * 10 + threadIdx.x / 25;
    remaining = transform;
    for(int d = 0; d < dim; ++d)
    {
        if(d == trans_dim)
            continue;
        indexes[d] = remaining % lengths[d];
        remaining  = remaining / lengths[d];
    }
    indexes[trans_dim] = 0;
    indexes[dim]       = remaining;

    offset_in  = 0;
    offset_out = 0;
    for(int d = 0; d <= dim; ++d)
    {
        offset_in += indexes[d] * stride[d];
        offset_out += indexes[d] * stride[permute[d]];
    }
    stride_in  = stride[trans_dim];
    stride_out = stride[permute[trans_dim]];

    offset_lds = 55 * (transform % 10);

    write = true;
    if(indexes[dim] >= nbatch)
    {
        write = false;
    }

    // transform
    forward_length55_SBRR_device<scalar_type, SB_UNIT>(lds,
                                                       twiddles,
                                                       stride_lds,
                                                       offset_lds,
                                                       write,
                                                       buf_in,
                                                       buf_out,
                                                       stride_in,
                                                       stride_out,
                                                       offset_in,
                                                       offset_out);
}

template <typename scalar_type>
void forward_length55_launch(scalar_type*       buf_in,
                             scalar_type*       buf_out,
                             const int          dim,
                             const int          trans_dim,
                             size_t*            lengths,
                             size_t             nbatch,
                             const scalar_type* twiddles,
                             size_t*            kargs)
{
    size_t nblocks;
    size_t ntransforms = 1;
    for(int d = 0; d < dim; ++d)
    {
        if(d == trans_dim)
            continue;
        ntransforms *= lengths[d];
    }
    ntransforms *= nbatch;
    nblocks = (ntransforms + 9) / 10;
    forward_length55_SBRR<scalar_type><<<nblocks, 250, 8000>>>(
        twiddles, dim, trans_dim, kargs, kargs + dim, kargs + 2 * dim + 1, nbatch, buf_in, buf_out);
}

template <typename T>
tuple<vector<float>, vector<T>> fft_stockham_gpu(vector<T> const& x, size_t nx, size_t nbatch)
{
    vector<float> times;
    int           ntrials = 1;

    auto z = copy(x);

    T* X;
    HIP_CHECK(hipMalloc(&X, nx * nbatch * sizeof(T)));
    HIP_CHECK(hipMemcpy(X, z.data(), nx * nbatch * sizeof(T), hipMemcpyHostToDevice));

    T* twiddles;
    HIP_CHECK(hipMalloc(&twiddles, (nx - 1) * sizeof(T)));

    GPUTimer total;
    total.tic();
    vector<vector<int>> factors = {{5, 5, 3}};

    int* d_factors;
    HIP_CHECK(hipMalloc(&d_factors, factors[0].size() * sizeof(int)));
    HIP_CHECK(hipMemcpy(
        d_factors, factors[0].data(), factors[0].size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, nx - 1>>>(nx - 1, twiddles, factors[0].size(), d_factors);
    HIP_CHECK(hipFree(d_factors));

    if(false)
    {
        vector<T> t(nx - 1);
        HIP_CHECK(hipMemcpy(t.data(), twiddles, t.size() * sizeof(T), hipMemcpyDeviceToHost));
        for(int i = 0; i < nx - 1; ++i)
            cout << i << " " << t[i].x << " " << t[i].y << endl;
    }

    const int nargs = 5;
    size_t*   d_kargs;
    size_t    kargs[nargs];
    HIP_CHECK(hipMalloc(&d_kargs, nargs * sizeof(size_t)));

    kargs[0] = nx; // passed to global function as lengths[0]
    kargs[1] = 1; // passed to global function as stride[0]
    kargs[2] = nx; // passed to global function as stride[1]
    kargs[3] = 0; // passed to global function as permute[0]
    kargs[4] = 1; // passed to global function as permute[1]
    HIP_CHECK(hipMemcpy(d_kargs, kargs, nargs * sizeof(size_t), hipMemcpyHostToDevice));

    GPUTimer timer;
    for(int n = 0; n <= ntrials; ++n)
    {
        timer.tic();
        forward_length75_launch<T>(X, X, 1, 0, kargs, nbatch, twiddles, d_kargs);

        timer.toc();
        if(n > 0)
            times.push_back(timer.elapsed());
        if(n == 0)
            HIP_CHECK(hipMemcpy(z.data(), X, nx * nbatch * sizeof(T), hipMemcpyDeviceToHost));
    }
    total.toc();

    HIP_CHECK(hipFree(d_kargs));
    HIP_CHECK(hipFree(twiddles));
    HIP_CHECK(hipFree(X));

    return {move(times), move(z)};
}

template <typename T>
tuple<vector<float>, vector<T>>
    fft_stockham_gpu(vector<T> const& x, size_t nx, size_t ny, size_t nbatch)
{
    vector<float> times;
    int           ntrials = 10;

    auto z = copy(x);

    T* X0;
    T* X1;
    HIP_CHECK(hipMalloc(&X0, nx * ny * nbatch * sizeof(T)));
    HIP_CHECK(hipMalloc(&X1, nx * ny * nbatch * sizeof(T)));
    HIP_CHECK(hipMemcpy(X0, z.data(), nx * ny * nbatch * sizeof(T), hipMemcpyHostToDevice));

    // to be consistent with rocFFT; use nx and ny instead of nx-1 and ny-1 for twiddle tables
    T* twiddles;
    HIP_CHECK(hipMalloc(&twiddles, (nx + ny) * sizeof(T)));

    GPUTimer total;
    total.tic();
    vector<vector<int>> factors = {{5, 5, 3}, {5, 5, 3}};

    int* d_factors;
    HIP_CHECK(hipMalloc(&d_factors, max(factors[0].size(), factors[1].size()) * sizeof(int)));
    HIP_CHECK(hipMemcpy(
        d_factors, factors[0].data(), factors[0].size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, nx - 1>>>(nx - 1, twiddles, factors[0].size(), d_factors);
    HIP_CHECK(hipMemcpy(
        d_factors, factors[1].data(), factors[1].size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, ny - 1>>>(ny - 1, twiddles + nx, factors[1].size(), d_factors);
    HIP_CHECK(hipFree(d_factors));

    if(false)
    {
        vector<T> t(nx - 1);
        HIP_CHECK(hipMemcpy(t.data(), twiddles, t.size() * sizeof(T), hipMemcpyDeviceToHost));
        for(int i = 0; i < nx - 1; ++i)
            cout << i << " " << t[i].x << " " << t[i].y << endl;
    }

    const int nargs = 8;
    size_t*   d_kargs;
    size_t    kargs[nargs];

    const bool use_permute = true;

    kargs[0] = nx; // passed to global function as lengths[0]
    kargs[1] = ny; // passed to global function as lengths[1]
    kargs[2] = 1; // passed to global function as stride[0]
    kargs[3] = nx; // passed to global function as stride[1]
    kargs[4] = nx * ny; // passed to global function as stride[2]

    if(use_permute)
    {
        // permute as below means: x is read, transforms, and written into y
        kargs[5] = 1; // passed to global function as permute[0]
        kargs[6] = 0; // passed to global function as permute[1]
        kargs[7] = 2; // passed to global function as permute[2]
    }
    else
    {
        // permute as below means: identity
        kargs[5] = 0; // passed to global function as permute[0]
        kargs[6] = 1; // passed to global function as permute[1]
        kargs[7] = 2; // passed to global function as permute[2]
    }

    HIP_CHECK(hipMalloc(&d_kargs, nargs * sizeof(size_t)));
    HIP_CHECK(hipMemcpy(d_kargs, kargs, nargs * sizeof(size_t), hipMemcpyHostToDevice));

    GPUTimer timer;
    for(int n = 0; n <= ntrials; ++n)
    {
        timer.tic();

        if(use_permute)
        {
            // both calls transform x and write into y
            forward_length75_launch<T>(X0, X1, 2, 0, kargs, nbatch, twiddles, d_kargs);
            forward_length75_launch<T>(X1, X0, 2, 0, kargs, nbatch, twiddles, d_kargs);
        }
        else
        {
            // permute is the identify, second call transforms along y
            forward_length75_launch<T>(X0, X1, 2, 0, kargs, nbatch, twiddles, d_kargs);
            forward_length75_launch<T>(X1, X0, 2, 1, kargs, nbatch, twiddles, d_kargs);
        }

        timer.toc();
        if(n > 0)
            times.push_back(timer.elapsed());
        if(n == 0)
            HIP_CHECK(hipMemcpy(z.data(), X0, nx * ny * nbatch * sizeof(T), hipMemcpyDeviceToHost));
    }
    total.toc();

    HIP_CHECK(hipFree(d_kargs));
    HIP_CHECK(hipFree(twiddles));
    HIP_CHECK(hipFree(X0));
    HIP_CHECK(hipFree(X1));

    return {move(times), move(z)};
}

template <typename T>
tuple<vector<float>, vector<T>>
    fft_stockham_gpu(vector<T> const& x, size_t nx, size_t ny, size_t nz, size_t nbatch)
{
    vector<float> times;
    int           ntrials = 10;

    auto z = copy(x);

    T* X0;
    T* X1;
    HIP_CHECK(hipMalloc(&X0, nx * ny * nz * nbatch * sizeof(T)));
    HIP_CHECK(hipMalloc(&X1, nx * ny * nz * nbatch * sizeof(T)));
    HIP_CHECK(hipMemcpy(X0, z.data(), nx * ny * nz * nbatch * sizeof(T), hipMemcpyHostToDevice));

    T* twiddles;
    HIP_CHECK(hipMalloc(&twiddles, (nx + ny + nz) * sizeof(T)));

    GPUTimer total;
    total.tic();
    vector<vector<int>> factors = {{5, 5, 3}, {5, 11}, {5, 11}};

    int* d_factors;
    HIP_CHECK(hipMalloc(&d_factors, 32 * sizeof(int)));
    HIP_CHECK(hipMemcpy(
        d_factors, factors[0].data(), factors[0].size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, nx - 1>>>(nx - 1, twiddles, factors[0].size(), d_factors);
    HIP_CHECK(hipMemcpy(
        d_factors, factors[1].data(), factors[1].size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, ny - 1>>>(ny - 1, twiddles + nx, factors[1].size(), d_factors);
    HIP_CHECK(hipMemcpy(
        d_factors, factors[2].data(), factors[2].size() * sizeof(int), hipMemcpyHostToDevice));
    stockham_twiddles<<<1, nz - 1>>>(nz - 1, twiddles + nx + ny, factors[2].size(), d_factors);
    HIP_CHECK(hipFree(d_factors));

    const int nargs = 11;
    size_t*   d_kargs;
    size_t*   d_kargs1;
    size_t*   d_kargs2;
    size_t*   d_kargs3;
    size_t    kargs[nargs];

    const bool use_permute = false;

    kargs[0] = nx;
    kargs[1] = ny;
    kargs[2] = nz;
    kargs[3] = 1;
    kargs[4] = nx;
    kargs[5] = nx * ny;
    kargs[6] = nx * ny * nz;

    // d_kargs: permute: identity
    kargs[7]  = 0;
    kargs[8]  = 1;
    kargs[9]  = 2;
    kargs[10] = 3;

    HIP_CHECK(hipMalloc(&d_kargs, nargs * sizeof(size_t)));
    HIP_CHECK(hipMemcpy(d_kargs, kargs, nargs * sizeof(size_t), hipMemcpyHostToDevice));

    // d_kargs1: permute: write x into y, y into z, z into x
    kargs[7]  = 2;
    kargs[8]  = 0;
    kargs[9]  = 1;
    kargs[10] = 3;

    HIP_CHECK(hipMalloc(&d_kargs1, nargs * sizeof(size_t)));
    HIP_CHECK(hipMemcpy(d_kargs1, kargs, nargs * sizeof(size_t), hipMemcpyHostToDevice));

    // d_kargs2: permute: write y into x, x into y
    kargs[7]  = 1;
    kargs[8]  = 0;
    kargs[9]  = 2;
    kargs[10] = 3;

    HIP_CHECK(hipMalloc(&d_kargs2, nargs * sizeof(size_t)));
    HIP_CHECK(hipMemcpy(d_kargs2, kargs, nargs * sizeof(size_t), hipMemcpyHostToDevice));

    // d_kargs3: permute: write z into x, x into z
    kargs[7]  = 2;
    kargs[8]  = 1;
    kargs[9]  = 0;
    kargs[10] = 3;

    HIP_CHECK(hipMalloc(&d_kargs3, nargs * sizeof(size_t)));
    HIP_CHECK(hipMemcpy(d_kargs3, kargs, nargs * sizeof(size_t), hipMemcpyHostToDevice));

    GPUTimer timer;
    for(int n = 0; n <= ntrials; ++n)
    {
        timer.tic();

        if(use_permute)
        {
            forward_length75_launch<T>(X0, X1, 3, 0, kargs, nbatch, twiddles, d_kargs1);
            forward_length55_launch<T>(X1, X0, 3, 0, kargs, nbatch, twiddles+nx, d_kargs2);
            forward_length55_launch<T>(X0, X1, 3, 0, kargs, nbatch, twiddles+nx+ny, d_kargs3);
        }
        else
        {
            forward_length75_launch<T>(X0, X1, 3, 0, kargs, nbatch, twiddles, d_kargs);
            forward_length55_launch<T>(X1, X0, 3, 1, kargs, nbatch, twiddles+nx, d_kargs);
            forward_length55_launch<T>(X0, X1, 3, 2, kargs, nbatch, twiddles+nx+ny, d_kargs);
        }

        timer.toc();
        if(n > 0)
            times.push_back(timer.elapsed());
        if(n == 0)
            HIP_CHECK(
                hipMemcpy(z.data(), X1, nx * ny * nz * nbatch * sizeof(T), hipMemcpyDeviceToHost));
    }
    total.toc();

    HIP_CHECK(hipFree(d_kargs));
    HIP_CHECK(hipFree(d_kargs1));
    HIP_CHECK(hipFree(d_kargs2));
    HIP_CHECK(hipFree(d_kargs3));
    HIP_CHECK(hipFree(twiddles));
    HIP_CHECK(hipFree(X0));
    HIP_CHECK(hipFree(X1));

    return {move(times), move(z)};
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
void test1d(size_t n, size_t nbatch)
{
    double GiB = double(n * nbatch * sizeof(T)) / 1024 / 1024 / 1024;
    double GB  = double(n * nbatch * sizeof(T)) / 1000 / 1000 / 1000;

    cout << "# 1d test" << endl;
    cout << "1d input length:  " << n << " (" << nbatch << ")" << endl;
    cout << "1d input size:    " << GiB << " GiB" << endl;
    cout << "1d input size:    " << GB << " GB" << endl;

    auto x = random_vector<T>(n * nbatch);

    auto [t1, z1] = fft_fftw(x, n, nbatch);
    cout << "FFTW time:        " << t1 << " ms" << endl;

    auto [a2, z2] = fft_stockham_gpu(x, n, nbatch);
    auto t2       = average(a2);

    cout << "GPU rel diff:     " << compare(z1, z2) << endl;
    cout << "GPU kernel time:  " << t2 << " ms (average)" << endl;
    cout << "GPU kernel times: ";
    for(auto& t : a2)
    {
        cout << t << " ";
    }
    cout << "ms" << endl;
    cout << "GPU throughput:   " << GiB * 1000 / t2 << " GiB/s one-way" << endl;
    cout << "GPU throughput:   " << GB * 1000 / t2 << " GB/s one-way" << endl;
    cout << "GPU throughput:   " << 2 * GiB * 1000 / t2 << " GiB/s two-way" << endl;
    cout << "GPU throughput:   " << 2 * GB * 1000 / t2 << " GB/s two-way" << endl;
    cout << "GFLOPS:           " << nbatch * 5 * n * log(n) / log(2.0) / (1e6 * t2) << endl;
    cout << "TFLOPS:           " << nbatch * 5 * n * log(n) / log(2.0) / (1e9 * t2) << endl;
}

template <typename T>
void test2d(size_t nx, size_t ny, size_t nbatch)
{
    double GiB = double(nx * ny * nbatch * sizeof(T)) / 1024 / 1024 / 1024;
    double GB  = double(nx * ny * nbatch * sizeof(T)) / 1000 / 1000 / 1000;

    cout << "# 2d test" << endl;
    cout << "2d input length: " << nx << "x" << ny << " (" << nbatch << ")" << endl;
    cout << "2d input size:   " << GiB << " GiB" << endl;
    cout << "2d input size:   " << GB << " GB" << endl;

    auto x = random_vector<T>(nx * ny * nbatch);

    auto [t1, z1] = fft_fftw(x, nx, ny, nbatch);
    cout << "FFTW time:       " << t1 << " ms" << endl;

    auto [a2, z2] = fft_stockham_gpu(x, nx, ny, nbatch);
    auto t2       = average(a2);

    cout << "GPU rel diff:    " << compare(z1, z2) << endl;
    cout << "GPU kernel times: ";
    for(auto& t : a2)
    {
        cout << t << " ";
    }
    cout << "ms" << endl;
    cout << "GPU kernel time: " << t2 << " ms (average)" << endl;
    cout << "GPU throughput:  " << GiB * 1000 / t2 << " GiB/s one-way" << endl;
    cout << "GPU throughput:  " << GB * 1000 / t2 << " GB/s one-way" << endl;
    cout << "GPU throughput:  " << 2 * GiB * 1000 / t2 << " GiB/s two-way" << endl;
    cout << "GPU throughput:  " << 2 * GB * 1000 / t2 << " GB/s two-way" << endl;
}

template <typename T>
void test3d(size_t nx, size_t ny, size_t nz, size_t nbatch)
{
    double GiB = double(nx * ny * nz * nbatch * sizeof(T)) / 1024 / 1024 / 1024;
    double GB  = double(nx * ny * nz * nbatch * sizeof(T)) / 1000 / 1000 / 1000;

    cout << "# 3d test" << endl;
    cout << "3d input length: " << nx << "x" << ny << "x" << nz << " (" << nbatch << ")" << endl;
    cout << "3d input size:   " << GiB << " GiB" << endl;
    cout << "3d input size:   " << GB << " GB" << endl;

    auto x = random_vector<T>(nx * ny * nz * nbatch);

    auto [t1, z1] = fft_fftw(x, nx, ny, nz, nbatch);
    cout << "FFTW time:       " << t1 << " ms" << endl;

    auto [a2, z2] = fft_stockham_gpu(x, nx, ny, nz, nbatch);
    auto t2       = average(a2);

    cout << "GPU rel diff:    " << compare(z1, z2) << endl;
    cout << "GPU kernel times: ";
    for(auto& t : a2)
    {
        cout << t << " ";
    }
    cout << "ms" << endl;
    cout << "GPU kernel time: " << t2 << " ms (average)" << endl;
    cout << "GPU throughput:  " << GiB * 1000 / t2 << " GiB/s one-way" << endl;
    cout << "GPU throughput:  " << GB * 1000 / t2 << " GB/s one-way" << endl;
    cout << "GPU throughput:  " << 2 * GiB * 1000 / t2 << " GiB/s two-way" << endl;
    cout << "GPU throughput:  " << 2 * GB * 1000 / t2 << " GB/s two-way" << endl;
}

int main(int argc, char* argv[])
{
    int nbatch = 1;
    if(argc > 1)
        nbatch = stoi(argv[1]);

    // test1d<hipDoubleComplex>(75, nbatch);
    test2d<hipDoubleComplex>(75, 75, nbatch);
    test3d<hipDoubleComplex>(75, 55, 55, nbatch);
}

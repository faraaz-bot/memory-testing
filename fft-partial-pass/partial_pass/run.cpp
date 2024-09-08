#include <iostream>
#include <complex.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <random>
#include <iomanip>
#include <sstream>

#include "rocfft/printbuffer.h"
#include "rocfft/data_gen_host.h"

#include "hip/hip_runtime.h"
#include "compute_twiddles.h"
#include "launch_3D_FFT.h"

#define TIMING_NO_TRIALS 20
#define DATA_EMPTY_VALUE -123456789

template <typename T>
void ComputeRandomInputBuffer(const size_t N,
                              const std::vector<size_t> &whole_length,
                              const std::vector<size_t> &whole_stride,
                              const size_t idist,
                              const size_t nbatch,
                              gpubuf_t<T> &input)
{
    std::vector<T> input_host(N);

    generate_random_interleaved_data(input_host,
                                     std::make_tuple(whole_length[2], whole_length[1], whole_length[0]),
                                     std::make_tuple(whole_stride[2], whole_stride[1], whole_stride[0]),
                                     idist,
                                     nbatch);

    auto input_size_bytes = N * sizeof(T);

    if (input.alloc(input_size_bytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failure");

    if (hipMemcpy(input.data(),
                  input_host.data(),
                  input_size_bytes,
                  hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failure");
}

template <typename T>
void ComputeInputBuffer(size_t N, gpubuf_t<T> &input)
{
    std::vector<T> input_host(N);

    auto i = 0;
    for (auto &elem : input_host)
    {
        elem.x = i;
        elem.y = i;

        i++;
    }

    auto input_size_bytes = N * sizeof(T);

    if (input.alloc(input_size_bytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failure");

    if (hipMemcpy(input.data(),
                  input_host.data(),
                  input_size_bytes,
                  hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failure");
}

template <typename T>
void InitializeOutputBuffer(size_t N, gpubuf_t<T> &output)
{
    std::vector<T> output_host(N);

    for (auto &elem : output_host)
    {
        elem.x = DATA_EMPTY_VALUE;
        elem.y = DATA_EMPTY_VALUE;
    }

    auto output_size_bytes = N * sizeof(T);

    if (output.alloc(output_size_bytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failure");

    if (hipMemcpy(output.data(),
                  output_host.data(),
                  output_size_bytes,
                  hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failure");
}

template <typename T>
struct CS_3D_RC_64_64_64_Params
{
    CS_3D_RC_64_64_64_Params()
    {
        length = {64, 64, 64};

        N = 1;
        for (size_t i = 0; i < length.size(); ++i)
            N *= length[i];

        size_t M1 = length[0], M2 = M1 * M1;
        radices_RR = {8, 8};
        radices_CC = {8, 8};

        inStride_1 = {1, M1, M2};
        inStride_2 = {M1, 1, M2};
        inStride_3 = {M2, 1, M1};
        outStride_1 = inStride_1;
        outStride_2 = inStride_2;
        outStride_3 = inStride_3;
        iDist = M1 * M2;
        oDist = iDist;
        lds_padding = 0;

        batch = 1;

        ComputeTwiddles(M1, radices_RR, twiddles1_RR);
        ComputeTwiddles(M1, radices_CC, twiddles2_CC);
        ComputeTwiddles(M1, radices_CC, twiddles3_CC);

        Get_SBRR_64_Args<T>(length, inStride_1, outStride_1, iDist, oDist, batch, lds_padding, gridParam1, devKernArg1);
        Get_SBCC_64_Args<T>(length, inStride_2, outStride_2, iDist, oDist, batch, lds_padding, gridParam2, devKernArg2);
        Get_SBCC_64_Args<T>(length, inStride_3, outStride_3, iDist, oDist, batch, lds_padding, gridParam3, devKernArg3);

        if (ibuf.alloc(N * sizeof(rocfft_complex<T>)) != hipSuccess)
            throw std::runtime_error("hipMalloc failure");
        ComputeRandomInputBuffer(N, length, inStride_1, iDist, batch, ibuf);
        //ComputeInputBuffer(N, ibuf);

        if (obuf.alloc(N * sizeof(rocfft_complex<T>)) != hipSuccess)
            throw std::runtime_error("hipMalloc failure");
        InitializeOutputBuffer(N, obuf);
    }

    std::vector<size_t> length;

    size_t N;

    std::vector<size_t> radices_RR;
    std::vector<size_t> radices_CC;

    gpubuf_t<rocfft_complex<T>> twiddles1_RR;
    gpubuf_t<rocfft_complex<T>> twiddles2_CC;
    gpubuf_t<rocfft_complex<T>> twiddles3_CC;

    std::vector<size_t> inStride_1, inStride_2, inStride_3;
    std::vector<size_t> outStride_1, outStride_2, outStride_3;
    size_t iDist, oDist;

    size_t lds_padding;

    size_t batch;

    UserCallbacks callbacks;

    GridParam gridParam1;
    GridParam gridParam2;
    GridParam gridParam3;

    gpubuf_t<size_t> devKernArg1;
    gpubuf_t<size_t> devKernArg2;
    gpubuf_t<size_t> devKernArg3;

    gpubuf_t<rocfft_complex<T>> ibuf;
    gpubuf_t<rocfft_complex<T>> obuf;
};

// Computes the DFT matrix in row major format
template <typename T>
void ComputeDFTMatrix(size_t N, gpubuf_t<T> &input)
{
    std::vector<T> input_host(N * N);

    for (size_t i_row = 0; i_row < N; ++i_row)
    {
        for (size_t i_col = 0; i_col < N; ++i_col)
        {
            size_t n = i_row * i_col;

            auto arg = -TWO_PI * n / N;

            input_host[N * i_col + i_row].x = cos(arg);
            // TODO: Why minus sign here?
            input_host[N * i_col + i_row].y = -sin(arg);
        }
    }

    auto input_size_bytes = input_host.size() * sizeof(T);

    if (input.alloc(input_size_bytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failure");

    if (hipMemcpy(input.data(),
                  input_host.data(),
                  input_size_bytes,
                  hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failure");
}

template <typename T>
void ComputeDFTMatrixLarge(size_t N, gpubuf_t<T> &input)
{
    std::vector<T> input_host(N * N);

    for (size_t i_row = 0; i_row < N; ++i_row)
    {
        for (size_t i_col = 0; i_col < N; ++i_col)
        {
            size_t n = i_row * i_col;

            auto arg = -TWO_PI * n / N;

            input_host[N * i_col + i_row].x = cos(arg);
            input_host[N * i_col + i_row].y = sin(arg);
        }
    }

    auto input_size_bytes = input_host.size() * sizeof(T);

    if (input.alloc(input_size_bytes) != hipSuccess)
        throw std::runtime_error("hipMalloc failure");

    if (hipMemcpy(input.data(),
                  input_host.data(),
                  input_size_bytes,
                  hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("hipMemcpy failure");
}

template <typename T>
void DisplayBuffer(const size_t N,
                   const std::vector<size_t> &length,
                   const std::vector<size_t> &stride,
                   const size_t nbatch,
                   const size_t dist,
                   const gpubuf_t<T> &buffer)
{
    std::vector<T> buffer_host(N);

    auto buffer_size_bytes = N * sizeof(T);

    if (hipMemcpy(buffer_host.data(),
                  buffer.data(),
                  buffer_size_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("hipMemcpy failure");

    auto length_cm = length;
    std::reverse(length_cm.begin(), length_cm.end());
    auto stride_cm = stride;
    std::reverse(stride_cm.begin(), stride_cm.end());

    print_buffer(buffer_host,
                 length_cm,
                 stride_cm,
                 nbatch,
                 dist);
}

template <typename T>
void Run_CS_3D_RC_64_64_64(bool measure_exec, bool print_input)
{
    CS_3D_RC_64_64_64_Params<T> csParams;

    size_t ntrial = measure_exec ? TIMING_NO_TRIALS : 1;
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    if (hipEventCreate(&start) != hipSuccess)
        throw std::runtime_error("hipEventCreate failure");
    if (hipEventCreate(&stop) != hipSuccess)
        throw std::runtime_error("hipEventCreate failure");

    if (print_input && !measure_exec)
    {
        DisplayBuffer(csParams.N,
                      csParams.length,
                      csParams.inStride_1,
                      csParams.batch,
                      csParams.iDist,
                      csParams.ibuf);
        return;
    }

    for (unsigned int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
        if (hipEventRecord(start) != hipSuccess)
            throw std::runtime_error("hipEventRecord failure");

        Launch_3D_FFT(csParams.N,
                      csParams.length,
                      csParams.twiddles1_RR,
                      csParams.twiddles2_CC,
                      csParams.twiddles3_CC,
                      csParams.lds_padding,
                      csParams.gridParam1,
                      csParams.gridParam2,
                      csParams.gridParam3,
                      csParams.devKernArg1,
                      csParams.devKernArg2,
                      csParams.devKernArg3,
                      csParams.batch,
                      csParams.callbacks,
                      csParams.ibuf);

        if (hipEventRecord(stop) != hipSuccess)
            throw std::runtime_error("hipEventRecord failure");
        if (hipEventSynchronize(stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failure");

        float time;
        if (hipEventElapsedTime(&time, start, stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failure");
        gpu_time[itrial] = time;
    }

    if (measure_exec)
    {
        std::cout << "Execution gpu time:";
        for (const auto &time : gpu_time)
            std::cout << " " << time;
        std::cout << " ms" << std::endl;
    }
    else
    {
        DisplayBuffer(csParams.N,
                      csParams.length,
                      csParams.inStride_1,
                      csParams.batch,
                      csParams.iDist,
                      csParams.ibuf);
    }
}

template <typename T>
void Run_CS_3D_RC_64_64_64_PP(bool measure_exec, bool print_input)
{
    CS_3D_RC_64_64_64_Params<T> csParams;

    size_t pp_dim = 1;

    gpubuf_t<rocfft_complex<T>> DFT_matrix_large;
    if (DFT_matrix_large.alloc(csParams.length[pp_dim] * csParams.length[pp_dim] * sizeof(rocfft_complex<T>)) != hipSuccess)
        throw std::runtime_error("hipMalloc failure");
    ComputeDFTMatrix(csParams.length[pp_dim], DFT_matrix_large);

    size_t ntrial = measure_exec ? TIMING_NO_TRIALS : 1;
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    if (hipEventCreate(&start) != hipSuccess)
        throw std::runtime_error("hipEventCreate failure");
    if (hipEventCreate(&stop) != hipSuccess)
        throw std::runtime_error("hipEventCreate failure");

    if (print_input && !measure_exec)
    {
        DisplayBuffer(csParams.N,
                      csParams.length,
                      csParams.inStride_1,
                      csParams.batch,
                      csParams.iDist,
                      csParams.ibuf);
        return;
    }

    for (unsigned int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
        if (hipEventRecord(start) != hipSuccess)
            throw std::runtime_error("hipEventRecord failure");

        Launch_3D_FFT_PP(csParams.N,
                         csParams.length,
                         DFT_matrix_large,
                         csParams.twiddles1_RR,
                         csParams.twiddles2_CC,
                         csParams.twiddles3_CC,
                         csParams.lds_padding,
                         csParams.gridParam1,
                         csParams.gridParam2,
                         csParams.gridParam3,
                         csParams.devKernArg1,
                         csParams.devKernArg2,
                         csParams.devKernArg3,
                         csParams.batch,
                         csParams.callbacks,
                         csParams.ibuf,
                         csParams.obuf);

        if (hipEventRecord(stop) != hipSuccess)
            throw std::runtime_error("hipEventRecord failure");
        if (hipEventSynchronize(stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failure");

        float time;
        if (hipEventElapsedTime(&time, start, stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failure");
        gpu_time[itrial] = time;
    }

    if (measure_exec)
    {
        std::cout << "Execution gpu time:";
        for (const auto &time : gpu_time)
            std::cout << " " << time;
        std::cout << " ms" << std::endl;
    }
    else
    {
        DisplayBuffer(csParams.N,
                      csParams.length,
                      csParams.outStride_1,
                      csParams.batch,
                      csParams.oDist,
                      csParams.obuf);
    }
}

int main(int argc, char **argv)
{
    bool measure_exec, print_input, enable_pp;

    measure_exec = strtol(argv[1], nullptr, 0);
    print_input = strtol(argv[2], nullptr, 0);
    enable_pp = strtol(argv[3], nullptr, 0);

    if (enable_pp)
        Run_CS_3D_RC_64_64_64_PP<double>(measure_exec, print_input);
    else
        Run_CS_3D_RC_64_64_64<double>(measure_exec, print_input);

    return EXIT_SUCCESS;
}
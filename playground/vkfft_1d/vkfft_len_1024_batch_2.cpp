/// build:
///    hipcc vkfft_len_1024_batch_2.cpp -o v_test -lfftw3f

#include <chrono>
#include <fftw3.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#define VKFFT_BACKEND 2
#include "common.h"
#include "vkFFT.h"

#define HIP_CHECK(condition)                                                           \
    {                                                                                  \
        hipError_t error = condition;                                                  \
        if(error != hipSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }

#define VKFFT_CHECK(condition)                                                         \
    {                                                                                  \
        VkFFTResult res;                                                               \
        res = condition;                                                               \
        if(res != VKFFT_SUCCESS)                                                       \
        {                                                                              \
            std::cout << "VKFFT error: " << res << " line: " << __LINE__ << std::endl; \
            exit(-1);                                                                  \
        }                                                                              \
    }

bool check_accuracy(const size_t         N,
                    const size_t         nbatch,
                    const fftwf_complex* ref_out,
                    float2*              new_out,
                    bool                 verbose)
{
    double max_linf_eps_single = 0.0;
    double max_l2_eps_single   = 0.0;

    VectorNorms cpu_output_norm = norm_complex<std::complex<float>, size_t, size_t>(
        reinterpret_cast<const std::complex<float>*>(ref_out), N, nbatch, 1, N, {0});

    VectorNorms gpu_output_norm = norm_complex<std::complex<float>, size_t, size_t>(
        reinterpret_cast<const std::complex<float>*>(new_out), N, nbatch, 1, N, {0});

    std::vector<std::pair<size_t, size_t>> linf_failures;
    const auto                             total_length = N;
    const double linf_cutoff = single_epsilon * cpu_output_norm.l_inf * log(total_length);

    VectorNorms diff = distance_1to1_complex<std::complex<float>, size_t, size_t, size_t>(
        reinterpret_cast<const std::complex<float>*>(ref_out),
        reinterpret_cast<const std::complex<float>*>(new_out),
        N,
        nbatch,
        1,
        N,
        1,
        N,
        linf_failures,
        linf_cutoff,
        {0},
        {0});

    if(verbose)
    {
        std::cout << "CPU Output Linf norm: " << std::scientific << cpu_output_norm.l_inf << "\n";
        std::cout << "CPU Output L2 norm:   " << std::scientific << cpu_output_norm.l_2 << "\n";
        std::cout << "GPU output Linf norm: " << std::scientific << gpu_output_norm.l_inf << "\n";
        std::cout << "GPU output L2 norm:   " << std::scientific << gpu_output_norm.l_2 << "\n";
        std::cout << "GPU linf norm failures:";
        std::sort(linf_failures.begin(), linf_failures.end());
        for(const auto& i : linf_failures)
        {
            std::cout << " (" << i.first << "," << i.second << ")";
        }
        std::cout << std::endl;
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    if(diff.l_inf > linf_cutoff)
    {
        std::cout << "Linf test failed.  Linf:" << diff.l_inf
                  << "\tnormalized Linf: " << diff.l_inf / cpu_output_norm.l_inf
                  << "\tcutoff: " << linf_cutoff;
        return false;
    }

    if(diff.l_2 / cpu_output_norm.l_2 >= sqrt(log2(total_length)) * single_epsilon)
    {
        std::cout << "L2 test failed. L2: " << diff.l_2
                  << "\tnormalized L2: " << diff.l_2 / cpu_output_norm.l_2
                  << "\tepsilon: " << sqrt(log2(total_length)) * single_epsilon;
        return false;
    }

    max_linf_eps_single
        = std::max(max_linf_eps_single, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
    max_l2_eps_single
        = std::max(max_l2_eps_single, diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));

    std::cout << "\tsingle precision max l-inf epsilon: " << std::scientific << max_linf_eps_single
              << std::endl;
    std::cout << "\tsingle precision max l2 epsilon: " << std::scientific << max_l2_eps_single
              << std::endl;
    return true;
}

void run_test(const int nx, const int batch)
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    uint64_t total_element = nx * batch;
    uint64_t total_bytes   = total_element * sizeof(float2);

    std::vector<float2> h_in(total_element);
    std::vector<float2> h_out(total_element);

    srandom(2);
    for(size_t i = 0; i < h_in.size(); i++)
    {
        h_in[i].x = rand() / 2147483647.0f; // float(i % 33) - 126;
        h_in[i].y = rand() / 2147483647.0f; // h_in[i].x + 1;
    }

    /// FFTW reference
    fftwf_complex* fftw_in  = new fftwf_complex[total_element];
    fftwf_complex* fftw_out = new fftwf_complex[total_element];

    for(size_t i = 0; i < h_in.size(); i++)
    {
        fftw_in[i][0] = h_in[i].x;
        fftw_in[i][1] = h_in[i].y;
    }

    fftwf_plan fftw_p = fftwf_plan_many_dft(
        1, &nx, batch, fftw_in, NULL, 1, nx, fftw_out, NULL, 1, nx, FFTW_FORWARD, FFTW_ESTIMATE);

    fftwf_execute(fftw_p);
    fftwf_destroy_plan(fftw_p);

    float2* d_in;
    float2* d_out;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_in), total_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out), total_bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), total_bytes, hipMemcpyHostToDevice));

    VkFFTConfiguration configuration = {};
    VkFFTApplication   app           = {};

    configuration.FFTdim        = 1;
    configuration.size[0]       = nx;
    configuration.size[1]       = 1;
    configuration.size[2]       = 1;
    configuration.numberBatches = batch;

    configuration.performR2C = 0;
    int device;
    HIP_CHECK(hipGetDevice(&device));
    configuration.device      = &device;
    configuration.stream      = &stream;
    configuration.num_streams = 1;

    configuration.isInputFormatted           = 1;
    configuration.isOutputFormatted          = 1;
    configuration.inverseReturnToInputBuffer = 0;
    configuration.buffer                     = (void**)&d_out;
    configuration.inputBuffer                = (void**)&d_in;
    //configuration.bufferStride[0] = nx;
    configuration.outputBuffer     = (void**)&d_out;
    configuration.bufferSize       = &total_bytes;
    configuration.inputBufferSize  = &total_bytes;
    configuration.bufferSize       = &total_bytes;
    configuration.outputBufferSize = &total_bytes;
    // configuration.disableMergeSequencesR2C = 1;

    VKFFT_CHECK(initializeVkFFT(&app, configuration));
    VKFFT_CHECK(VkFFTAppend(&app, -1, NULL));

    HIP_CHECK(hipMemcpy(h_out.data(), d_out, total_bytes, hipMemcpyDeviceToHost));

    deleteVkFFT(&app);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    bool passed = check_accuracy(nx, batch, fftw_out, (float2*)h_out.data(), true);

    std::cout << ((passed == true) ? "Passed." : "Failed.") << std::endl;

    delete[] fftw_in;
    delete[] fftw_out;

    HIP_CHECK(hipStreamDestroy(stream));
}

int main(int argc, char* argv[])
{
    hipDeviceProp_t dev_prop;
    int             device = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&dev_prop, device));
    std::cout << "Device: " << dev_prop.name << std::endl;

    //run_test(1024, 1);

    run_test(1024, 2);
}

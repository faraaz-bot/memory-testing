/*
hipcc -std=c++14 bench_1d_FFT.cpp -DENABLE_ROCFFT -lfftw3 -lfftw3f
-I/home/ayala/tmp/rocFFT-internal/build/rocfft/include
-L/home/ayala/tmp/rocFFT-internal/build/library/src/ -lrocfft -o bench_1d_FFT

./bench_1d_FFT > test_data_half.csv
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <fftw3.h>

#if defined(ENABLE_CUFFT)
#include <cuda_runtime_api.h>
#include <cufft.h>
#define data_type cufftDoubleComplex
#define plan_type cufftHandle
#define device_malloc(g_, size_) cudaMalloc((void **)&(g_), (size_));
#define device_mfree(g_) cudaFree((g_));
#define gpu_copy_h2d(h_, g_, size_)                                            \
  cudaMemcpy((g_), (h_), (size_), cudaMemcpyHostToDevice)
#define gpu_copy_d2h(h_, g_, size_)                                            \
  cudaMemcpy((h_), (g_), (size_), cudaMemcpyDeviceToHost)
#elif defined(ENABLE_ROCFFT)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include "/home/ayala/tmp/rocFFT-internal/shared/rocfft_complex.h"
#include <hip/hip_runtime.h>
#include <rocfft.h>
#define data_type rocfft_complex<_Float16>
// #define data_type   rocfft_complex<float>
#define plan_type rocfft_plan
#define device_malloc(g_, size_) hipMalloc((void **)&(g_), (size_));
#define device_mfree(g_) hipFree((g_));
#define gpu_copy_h2d(h_, g_, size_)                                            \
  hipMemcpy((g_), (h_), (size_), hipMemcpyHostToDevice)
#define gpu_copy_d2h(h_, g_, size_)                                            \
  hipMemcpy((h_), (g_), (size_), hipMemcpyDeviceToHost)
#elif defined(ENABLE_VKFFT)

#else
#define data_type double
#define plan_type double
#define host_malloc(h_, size_)
#define device_malloc(g_, size_)
#define host_mfree(h_)
#define device_mfree(g_)
#define gpu_copy_h2d(h_, g_, size_)
#define gpu_copy_d2h(h_, g_, size_)

#endif

void test_half(size_t fft_size) {
  // size_t fft_size = 8;
  std::vector<std::complex<float>> input(fft_size);
  std::vector<std::complex<float>> output_fftw(fft_size);
  std::string prec = "half";

  std::vector<data_type> input_gpu(fft_size);
  std::vector<data_type> output_gpu(fft_size);
  data_type *d_data_in = NULL, *d_data_out = NULL;
  plan_type plan;

  size_t mem_size = sizeof(data_type) * fft_size;
  device_malloc(d_data_in, mem_size);
  device_malloc(d_data_out, mem_size);

  std::minstd_rand park_miller(1234);
  std::uniform_real_distribution<float> unif(0.0, 1.0);

  // std::cout<< "CPU input, prec=SINGLE \n";
  for (auto &e : input) {
    e = static_cast<float>(unif(park_miller));
    // std::cout << e << ' ';
  }

  // std::cout<< "\nGPU input, prec=HALF \n";
  for (int i = 0; i < input.size(); i++) {
    input_gpu.at(i).x = static_cast<_Float16>(input.at(i).real());
    input_gpu.at(i).y = static_cast<_Float16>(input.at(i).imag());

    // std::cout << input_gpu.at(i) << ' ';
  }
  // std::cout<< "\n\n";

  gpu_copy_h2d(input_gpu.data(), d_data_in, mem_size);

#if defined(ENABLE_CUFFT)
  cufftPlan1d(&plan, fft_size, CUFFT_Z2Z, 1);

  for (size_t i = 0; i < niter; i++) {
    cudaDeviceSynchronize();
    cufftExecZ2Z(plan, d_data_in, d_data_out, CUFFT_FORWARD);
    cudaDeviceSynchronize();
  }

#elif defined(ENABLE_ROCFFT)
  rocfft_setup();

  // Creatre plan and buffer
  rocfft_plan plan_rocfft = nullptr;

  rocfft_status st = rocfft_plan_create(
      &plan_rocfft, rocfft_placement_notinplace,
      rocfft_transform_type_complex_forward, rocfft_precision_half,
      1,         // Dimension
      &fft_size, // length
      1,         // Number of transforms
      nullptr);

  if (st != rocfft_status_success)
    throw std::runtime_error("failed to create plan");

  size_t work_buf_size = 0;
  rocfft_plan_get_work_buffer_size(plan_rocfft, &work_buf_size);
  void *work_buf = nullptr;
  rocfft_execution_info info = nullptr;

  if (work_buf_size) {
    rocfft_execution_info_create(&info);
    hipMalloc(&work_buf, work_buf_size);
    rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
  }

  // Execute plan
  rocfft_execute(plan_rocfft, (void **)&d_data_in, (void **)&d_data_out, info);

  // Wait for execution to finish
  hipDeviceSynchronize();

  // Clean up work buffer
  if (work_buf_size) {
    hipFree(work_buf);
    rocfft_execution_info_destroy(info);
  }

  std::cout << "rocFFT result in " << prec << " precision: \n";
#endif

  gpu_copy_d2h(output_gpu.data(), d_data_out, mem_size);

  for (int i = 0; i < output_gpu.size(); i++)
    std::cout << output_gpu.at(i) << ' ';
  std::cout << "\n";

  device_mfree(d_data_in);
  device_mfree(d_data_out);

  // Result from FFTW
  fftwf_plan plan_validate;
  plan_validate = fftwf_plan_dft_1d(fft_size, (fftwf_complex *)input.data(),
                                    (fftwf_complex *)output_fftw.data(),
                                    FFTW_FORWARD, FFTW_ESTIMATE);

  fftwf_execute(plan_validate);

  std::cout << "\nFFTW result in single precision: \n";
  for (int i = 0; i < output_fftw.size(); i++)
    std::cout << output_fftw.at(i) << ' ';
  std::cout << "\n";

  double aux1 = 0.0;
  double aux2 = 0.0;
  double aux3 = 0.0;
  double aux4 = 0.0;

  double diff_norm_l2 = 0.0;
  double cpu_norm_l2 = 0.0;

  for (int i = 0; i < output_gpu.size(); i++) {
    // aux1 += std::pow(output_gpu.at(i).x -
    // static_cast<_Float16>(output_fftw.at(i).real()), 2) +
    // std::pow(output_gpu.at(i).y -
    // static_cast<_Float16>(output_fftw.at(i).imag()), 2);

    aux1 = std::pow(static_cast<_Float16>(output_fftw.at(i).real()) -
                        output_gpu.at(i).x,
                    2) +
           std::pow(static_cast<_Float16>(output_fftw.at(i).imag()) -
                        output_gpu.at(i).y,
                    2);
    aux2 = std::pow(static_cast<_Float16>(output_fftw.at(i).real()), 2) +
           std::pow(static_cast<_Float16>(output_fftw.at(i).imag()), 2);

    diff_norm_l2 += aux1;
    cpu_norm_l2 += aux2;

    aux3 = std::max(aux3, std::sqrt(aux1));
    aux4 = std::max(aux4, std::sqrt(aux2));
  }

  diff_norm_l2 = std::sqrt(diff_norm_l2);
  cpu_norm_l2 = std::sqrt(cpu_norm_l2);

  double diff_norm_l_inf = aux3;
  double cpu_norm_l_inf = aux4;

  std::cout << fft_size << "\t" << diff_norm_l2 << "\t" << cpu_norm_l2 << "\t"
            << diff_norm_l_inf << "\t" << cpu_norm_l_inf << std::endl;

  // Test in infinity norm: diff.l_inf <= linf_cutoff
  // linf_cutoff = type_epsilon(params.precision) * cpu_output_norm.l_inf *
  // log(total_length);

  // Test in L2 norm: diff.l_2 / cpu_output_norm.l_2 < sqrt(log2(total_length))
  // * type_epsilon(params.precision))

  // std::cout<< "\n|FFT_GPU(X)- FFTW(X)|_{2}: " << e << std::endl;
}

int main() {

  std::cout << "Size\tdiff_l2\tcpu_l2\tdiff_linf\tcpu_linf" << std::endl;
  for (size_t size = 1; size < 1025; size++)
    test_half(size);

  return 0;
}
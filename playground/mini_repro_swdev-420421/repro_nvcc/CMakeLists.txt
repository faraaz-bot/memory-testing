cmake_minimum_required(VERSION 3.10)

project( rocfft_kernel_harness_0 LANGUAGES CXX )

add_executable( rocfft_kernel_harness_0
  rocfft_kernel_harness_0.cpp
  fft_rtc_fwd_len128_factors_16_8_wgs_256_tpt_16_halfLds_sp_ip_CI_unitstride_sbrr_dirReg.cpp
  fft_rtc_fwd_len128_factors_16_8_wgs_256_tpt_16_sp_ip_CI_sbcc_dirReg.cpp
  fft_rtc_fwd_len50_factors_10_5_wgs_250_tpt_10_sp_ip_CI_sbcc_dirReg.cpp
  )

target_compile_options( rocfft_kernel_harness_0 PRIVATE -fopenmp )
target_link_libraries( rocfft_kernel_harness_0 PRIVATE -fopenmp )

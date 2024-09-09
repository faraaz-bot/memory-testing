#ifndef LAUNCH_CS_3D_RC_64_64_64_H
#define LAUNCH_CS_3D_RC_64_64_64_H

#include "launch_common.h"

#include "CS_3D_RC_64_64_64/sbcc_64.h"
#include "CS_3D_RC_64_64_64/sbrr_64.h"

#include "partial_pass_RR_CC_64_64_64/sbcc_pp_64.h"
#include "partial_pass_RR_CC_64_64_64/sbrr_pp_64.h"

template <typename T>
void Get_SBRR_64_Args(const std::vector<size_t> &length,
                      const std::vector<size_t> &inStride,
                      const std::vector<size_t> &outStride,
                      const size_t &iDist,
                      const size_t &oDist,
                      const size_t &batch,
                      const size_t lds_padding,
                      GridParam &gridParam,
                      gpubuf_t<size_t> &devKernArg)
{
    size_t batch_accum = batch;
    for (size_t j = 1; j < length.size(); j++)
        batch_accum *= length[j];

    auto workgroup_size = 64;
    auto threads_per_transform = 8;
    auto transforms_per_block = workgroup_size / threads_per_transform;

    auto bwd = transforms_per_block;
    auto lds = (length[0] + lds_padding) * bwd;

    gridParam.b_x = (batch_accum + bwd - 1) / bwd;
    gridParam.wgs_x = workgroup_size;

    gridParam.lds_bytes = (lds + lds_padding * bwd) * 2 * sizeof(T);

    devKernArg = kargs_create(length, inStride, outStride, iDist, oDist);
}

template <typename T>
void Get_SBCC_64_Args(const std::vector<size_t> &length,
                      const std::vector<size_t> &inStride,
                      const std::vector<size_t> &outStride,
                      const size_t &iDist,
                      const size_t &oDist,
                      const size_t &batch,
                      const size_t lds_padding,
                      GridParam &gridParam,
                      gpubuf_t<size_t> &devKernArg)
{
    auto workgroup_size = 64;
    auto threads_per_transform = 8;
    auto transforms_per_block = workgroup_size / threads_per_transform;

    auto bwd = transforms_per_block;
    auto lds = length[0] * bwd;

    gridParam.b_x = ((length[1]) - 1) / bwd + 1;
    gridParam.b_x *= std::accumulate(length.begin() + 2, length.end(), batch, std::multiplies<size_t>());
    gridParam.wgs_x = workgroup_size;

    gridParam.lds_bytes = (lds + lds_padding * bwd) * 2 * sizeof(T);

    devKernArg = kargs_create(length, inStride, outStride, iDist, oDist);
}

template <typename T>
inline void Launch_FFT_64_SBRR(const gpubuf_t<T> &twiddles,
                               const size_t &lds_padding,
                               const size_t &batch,
                               const GridParam &gridParam,
                               const gpubuf_t<size_t> &devKernArg,
                               const UserCallbacks &callbacks,
                               gpubuf_t<T> &buf)
{
    hipLaunchKernelGGL(fft_rtc_fwd_len64_factors_8_8_wgs_64_tpt_8_dim3_dp_ip_CI_unitstride_sbrr,
                       dim3(gridParam.b_x),
                       dim3(gridParam.wgs_x),
                       gridParam.lds_bytes,
                       0,
                       twiddles.data(),
                       kargs_lengths(devKernArg),
                       kargs_stride_in(devKernArg),
                       batch,
                       lds_padding,
                       callbacks.load_cb_fn,
                       callbacks.load_cb_data,
                       callbacks.load_cb_lds_bytes,
                       callbacks.store_cb_fn,
                       callbacks.store_cb_data,
                       buf.data());
}

template <typename T>
inline void Launch_FFT_64_SBRR_PP(const gpubuf_t<T> &DFT_matrix_large,
                                  const gpubuf_t<T> &twiddles,
                                  const size_t &lds_padding,
                                  const size_t &batch,
                                  const GridParam &gridParam,
                                  const gpubuf_t<size_t> &devKernArg,
                                  const UserCallbacks &callbacks,
                                  gpubuf_t<T> &buf)
{
    size_t N = 64 * 64 * 64;

    hipLaunchKernelGGL(fft_rtc_fwd_len64_factors_8_8_wgs_64_tpt_8_dim3_dp_ip_CI_unitstride_sbrr_pp,
                       dim3(gridParam.b_x),
                       dim3(gridParam.wgs_x),
                       gridParam.lds_bytes,
                       0,
                       DFT_matrix_large.data(),
                       twiddles.data(),
                       kargs_lengths(devKernArg),
                       kargs_stride_in(devKernArg),
                       batch,
                       N,
                       lds_padding,
                       callbacks.load_cb_fn,
                       callbacks.load_cb_data,
                       callbacks.load_cb_lds_bytes,
                       callbacks.store_cb_fn,
                       callbacks.store_cb_data,
                       buf.data());
}

template <typename T>
inline void Launch_FFT_64_SBCC(const gpubuf_t<T> &twiddles,
                               const size_t &lds_padding,
                               const size_t &batch,
                               const GridParam &gridParam,
                               const gpubuf_t<size_t> &devKernArg,
                               const UserCallbacks &callbacks,
                               gpubuf_t<T> &buf)
{
    hipLaunchKernelGGL(fft_rtc_fwd_len64_factors_8_8_wgs_64_tpt_8_dim3_dp_ip_CI_sbcc_dirReg_intrinsicRead,
                       dim3(gridParam.b_x),
                       dim3(gridParam.wgs_x),
                       gridParam.lds_bytes,
                       0,
                       twiddles.data(),
                       nullptr,
                       kargs_lengths(devKernArg),
                       kargs_stride_in(devKernArg),
                       batch,
                       lds_padding,
                       callbacks.load_cb_fn,
                       callbacks.load_cb_data,
                       callbacks.load_cb_lds_bytes,
                       callbacks.store_cb_fn,
                       callbacks.store_cb_data,
                       buf.data());
}

template <typename T>
inline void Launch_FFT_64_SBCC_PP(const gpubuf_t<T> &twiddles,
                                  const size_t &lds_padding,
                                  const size_t &batch,
                                  const GridParam &gridParam,
                                  const gpubuf_t<size_t> &devKernArg,
                                  const UserCallbacks &callbacks,
                                  gpubuf_t<T> &ibuf,
                                  gpubuf_t<T> &obuf)
{
    hipLaunchKernelGGL(fft_rtc_fwd_len64_factors_8_8_wgs_64_tpt_8_dim3_dp_ip_CI_sbcc_dirReg_intrinsicRead_pp,
                       dim3(gridParam.b_x),
                       dim3(gridParam.wgs_x),
                       gridParam.lds_bytes,
                       0,
                       twiddles.data(),
                       nullptr,
                       kargs_lengths(devKernArg),
                       kargs_stride_in(devKernArg),
                       batch,
                       lds_padding,
                       callbacks.load_cb_fn,
                       callbacks.load_cb_data,
                       callbacks.load_cb_lds_bytes,
                       callbacks.store_cb_fn,
                       callbacks.store_cb_data,
                       ibuf.data(),
                       obuf.data());
}

template <typename T>
inline void Launch_3D_FFT(const size_t &N,
                          const std::vector<size_t> &length,
                          const gpubuf_t<T> &twiddles1_RR,
                          const gpubuf_t<T> &twiddles2_CC,
                          const gpubuf_t<T> &twiddles3_CC,
                          const size_t &lds_padding,
                          const GridParam &gridParam1,
                          const GridParam &gridParam2,
                          const GridParam &gridParam3,
                          gpubuf_t<size_t> &devKernArg1,
                          gpubuf_t<size_t> &devKernArg2,
                          gpubuf_t<size_t> &devKernArg3,
                          const size_t &batch,
                          const UserCallbacks &callbacks,
                          gpubuf_t<T> &buf)
{
    Launch_FFT_64_SBRR(twiddles1_RR,
                       lds_padding,
                       batch,
                       gridParam1,
                       devKernArg1,
                       callbacks,
                       buf);

    Launch_FFT_64_SBCC(twiddles2_CC,
                       lds_padding,
                       batch,
                       gridParam2,
                       devKernArg2,
                       callbacks,
                       buf);

    Launch_FFT_64_SBCC(twiddles3_CC,
                       lds_padding,
                       batch,
                       gridParam3,
                       devKernArg3,
                       callbacks,
                       buf);
}

template <typename T>
inline void Launch_3D_FFT_PP(const size_t &N,
                             const std::vector<size_t> &length,
                             const gpubuf_t<T> &DFT_matrix_large,
                             const gpubuf_t<T> &twiddles1_RR,
                             const gpubuf_t<T> &twiddles2_CC,
                             const gpubuf_t<T> &twiddles3_CC,
                             const size_t &lds_padding,
                             const GridParam &gridParam1,
                             const GridParam &gridParam2,
                             const GridParam &gridParam3,
                             gpubuf_t<size_t> &devKernArg1,
                             gpubuf_t<size_t> &devKernArg2,
                             gpubuf_t<size_t> &devKernArg3,
                             const size_t &batch,
                             const UserCallbacks &callbacks,
                             gpubuf_t<T> &ibuf,
                             gpubuf_t<T> &obuf)
{
    Launch_FFT_64_SBRR_PP(DFT_matrix_large,
                          twiddles1_RR,
                          lds_padding,
                          batch,
                          gridParam1,
                          devKernArg1,
                          callbacks,
                          ibuf);

    Launch_FFT_64_SBCC_PP(twiddles3_CC,
                          lds_padding,
                          batch,
                          gridParam3,
                          devKernArg3,
                          callbacks,
                          ibuf,
                          obuf);
}

#endif
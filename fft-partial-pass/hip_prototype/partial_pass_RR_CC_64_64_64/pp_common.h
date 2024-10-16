#ifndef PP_COMMON_H
#define PP_COMMON_H

#include "../rocfft/callback.h"
#include "../rocfft/butterfly_constant.h"
#include "../rocfft/rocfft_complex.h"
#include "../rocfft/common.h"
#include "../rocfft/real2complex_device.h"
#include "../rocfft/rocfft_butterfly_template.h"
#include <hip/hip_runtime.h>

typedef rocfft_complex<double> scalar_type;

__device__ size_t apply_local_transpose(size_t index)
{
       // wrap around index around first batch
       auto index_transpose = index % (64 * 64 * 64);

       // apply local transpose transformation on first batch
       index_transpose = ((index_transpose % 64) + ((index_transpose % 1024) / 64) * 1024) % (4096 - 64) +
                         ((index_transpose / 1024) * 256) + (index_transpose / 4096) * (4096 - 1024);

       // move transformed index to correct batch
       index_transpose = index_transpose + (index / (64 * 64 * 64) * (64 * 64 * 64));

       return index_transpose;
}

template <typename scalar_type, StrideBin sb>
__device__ void lds_to_reg_input_length64_device_sbcc(scalar_type *R,
                                                      scalar_type *__restrict__ lds_complex,
                                                      unsigned int stride_lds,
                                                      unsigned int offset_lds,
                                                      unsigned int thread,
                                                      bool write)
{
       const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
       unsigned int l_offset;
       __syncthreads();
       l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
       R[0] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 8 * 4) * lstride;
       R[1] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 16 * 4) * lstride;
       R[2] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 24 * 4) * lstride;
       R[3] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 32 * 4) * lstride;
       R[4] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 40 * 4) * lstride;
       R[5] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 48 * 4) * lstride;
       R[6] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 56 * 4) * lstride;
       R[7] = lds_complex[l_offset];
}

template <typename scalar_type, StrideBin sb>
__device__ void lds_from_reg_output_length64_device_sbcc(scalar_type *R,
                                                         scalar_type *__restrict__ lds_complex,
                                                         unsigned int stride_lds,
                                                         unsigned int offset_lds,
                                                         unsigned int thread,
                                                         bool write)
{
       const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
       unsigned int l_offset;
       __syncthreads();
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 0 * 4) * lstride;
       lds_complex[l_offset] = R[0];
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 8 * 4) * lstride;
       lds_complex[l_offset] = R[1];
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 16 * 4) * lstride;
       lds_complex[l_offset] = R[2];
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 24 * 4) * lstride;
       lds_complex[l_offset] = R[3];
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 32 * 4) * lstride;
       lds_complex[l_offset] = R[4];
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 40 * 4) * lstride;
       lds_complex[l_offset] = R[5];
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 48 * 4) * lstride;
       lds_complex[l_offset] = R[6];
       l_offset = offset_lds + (((thread + 0 + 0) / (8 * 4)) * 64 + (thread + 0 + 0) % (8 * 4) + 56 * 4) * lstride;
       lds_complex[l_offset] = R[7];
}

template <typename scalar_type, StrideBin sb>
__device__ void lds_to_reg_input_length64_device_sbrr(scalar_type *R,
                                                      scalar_type *__restrict__ lds_complex,
                                                      unsigned int stride_lds,
                                                      unsigned int offset_lds,
                                                      unsigned int thread,
                                                      bool write)
{
       const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
       unsigned int l_offset;
       __syncthreads();
       l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
       R[0] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
       R[1] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
       R[2] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
       R[3] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
       R[4] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
       R[5] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
       R[6] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
       R[7] = lds_complex[l_offset];
}

template <typename scalar_type, StrideBin sb>
__device__ void lds_from_reg_output_length64_device_sbrr(scalar_type *R,
                                                         scalar_type *__restrict__ lds_complex,
                                                         unsigned int stride_lds,
                                                         unsigned int offset_lds,
                                                         unsigned int thread,
                                                         bool write)
{
       const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
       unsigned int l_offset;
       __syncthreads();
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 0) * lstride;
       lds_complex[l_offset] = R[0];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 8) * lstride;
       lds_complex[l_offset] = R[1];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 16) * lstride;
       lds_complex[l_offset] = R[2];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 24) * lstride;
       lds_complex[l_offset] = R[3];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 32) * lstride;
       lds_complex[l_offset] = R[4];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 40) * lstride;
       lds_complex[l_offset] = R[5];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 48) * lstride;
       lds_complex[l_offset] = R[6];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 56) * lstride;
       lds_complex[l_offset] = R[7];
}

template <typename scalar_type>
__device__ void lds_to_reg_4_input_length64_device_pp(scalar_type *R,
                                                      scalar_type *__restrict__ lds_complex,
                                                      unsigned int stride,
                                                      unsigned int offset)
{
       unsigned int idx, thread;
       __syncthreads();

       thread = 0;
       idx = offset + thread * stride;
       R[0] = lds_complex[idx];

       thread = 1;
       idx = offset + thread * stride;
       R[1] = lds_complex[idx];

       thread = 2;
       idx = offset + thread * stride;
       R[2] = lds_complex[idx];

       thread = 3;
       idx = offset + thread * stride;
       R[3] = lds_complex[idx];

       thread = 4;
       idx = offset + thread * stride;
       R[4] = lds_complex[idx];

       thread = 5;
       idx = offset + thread * stride;
       R[5] = lds_complex[idx];

       thread = 6;
       idx = offset + thread * stride;
       R[6] = lds_complex[idx];

       thread = 7;
       idx = offset + thread * stride;
       R[7] = lds_complex[idx];
}

template <typename scalar_type>
__device__ void lds_to_reg_16_input_length64_device_pp(scalar_type *R,
                                                       scalar_type *__restrict__ lds_complex,
                                                       unsigned int stride,
                                                       unsigned int offset)
{
       unsigned int idx, thread;
       __syncthreads();

       thread = 0;
       idx = offset + thread * stride;
       R[0] = lds_complex[idx];

       thread = 1;
       idx = offset + thread * stride;
       R[1] = lds_complex[idx];

       thread = 2;
       idx = offset + thread * stride;
       R[2] = lds_complex[idx];

       thread = 3;
       idx = offset + thread * stride;
       R[3] = lds_complex[idx];

       thread = 4;
       idx = offset + thread * stride;
       R[4] = lds_complex[idx];

       thread = 5;
       idx = offset + thread * stride;
       R[5] = lds_complex[idx];

       thread = 6;
       idx = offset + thread * stride;
       R[6] = lds_complex[idx];

       thread = 7;
       idx = offset + thread * stride;
       R[7] = lds_complex[idx];

       thread = 8;
       idx = offset + thread * stride;
       R[8] = lds_complex[idx];

       thread = 9;
       idx = offset + thread * stride;
       R[9] = lds_complex[idx];

       thread = 10;
       idx = offset + thread * stride;
       R[10] = lds_complex[idx];

       thread = 11;
       idx = offset + thread * stride;
       R[11] = lds_complex[idx];

       thread = 12;
       idx = offset + thread * stride;
       R[12] = lds_complex[idx];

       thread = 13;
       idx = offset + thread * stride;
       R[13] = lds_complex[idx];

       thread = 14;
       idx = offset + thread * stride;
       R[14] = lds_complex[idx];

       thread = 15;
       idx = offset + thread * stride;
       R[15] = lds_complex[idx];
}

template <typename scalar_type>
__device__ void lds_from_reg_4_output_length64_device_pp(scalar_type *R,
                                                         scalar_type *__restrict__ lds_complex,
                                                         unsigned int stride,
                                                         unsigned int offset)
{
       unsigned int idx, thread;
       __syncthreads();

       thread = 0;
       idx = offset + thread * stride;
       lds_complex[idx] = R[0];

       thread = 1;
       idx = offset + thread * stride;
       lds_complex[idx] = R[1];

       thread = 2;
       idx = offset + thread * stride;
       lds_complex[idx] = R[2];

       thread = 3;
       idx = offset + thread * stride;
       lds_complex[idx] = R[3];

       thread = 4;
       idx = offset + thread * stride;
       lds_complex[idx] = R[4];

       thread = 5;
       idx = offset + thread * stride;
       lds_complex[idx] = R[5];

       thread = 6;
       idx = offset + thread * stride;
       lds_complex[idx] = R[6];

       thread = 7;
       idx = offset + thread * stride;
       lds_complex[idx] = R[7];
}

template <typename scalar_type>
__device__ void lds_from_reg_16_output_length64_device_pp(scalar_type *R,
                                                          scalar_type *__restrict__ lds_complex,
                                                          unsigned int stride,
                                                          unsigned int offset)
{
       unsigned int idx, thread;
       __syncthreads();

       thread = 0;
       idx = offset + thread * stride;
       lds_complex[idx] = R[0];

       thread = 1;
       idx = offset + thread * stride;
       lds_complex[idx] = R[1];

       thread = 2;
       idx = offset + thread * stride;
       lds_complex[idx] = R[2];

       thread = 3;
       idx = offset + thread * stride;
       lds_complex[idx] = R[3];

       thread = 4;
       idx = offset + thread * stride;
       lds_complex[idx] = R[4];

       thread = 5;
       idx = offset + thread * stride;
       lds_complex[idx] = R[5];

       thread = 6;
       idx = offset + thread * stride;
       lds_complex[idx] = R[6];

       thread = 7;
       idx = offset + thread * stride;
       lds_complex[idx] = R[7];

       thread = 8;
       idx = offset + thread * stride;
       lds_complex[idx] = R[8];

       thread = 9;
       idx = offset + thread * stride;
       lds_complex[idx] = R[9];

       thread = 10;
       idx = offset + thread * stride;
       lds_complex[idx] = R[10];

       thread = 11;
       idx = offset + thread * stride;
       lds_complex[idx] = R[11];

       thread = 12;
       idx = offset + thread * stride;
       lds_complex[idx] = R[12];

       thread = 13;
       idx = offset + thread * stride;
       lds_complex[idx] = R[13];

       thread = 14;
       idx = offset + thread * stride;
       lds_complex[idx] = R[14];

       thread = 15;
       idx = offset + thread * stride;
       lds_complex[idx] = R[15];
}

template <typename scalar_type>
__device__ void twiddle_multiple_pp(scalar_type *R,
                                    unsigned int thread,
                                    const scalar_type *__restrict__ DFTmatrix_large)
{
       R[0] = DFTmatrix_large[thread * 64 + 0] * R[0];
       R[1] = DFTmatrix_large[thread * 64 + 1] * R[1];
       R[2] = DFTmatrix_large[thread * 64 + 2] * R[2];
       R[3] = DFTmatrix_large[thread * 64 + 3] * R[3];
       R[4] = DFTmatrix_large[thread * 64 + 4] * R[4];
       R[5] = DFTmatrix_large[thread * 64 + 5] * R[5];
       R[6] = DFTmatrix_large[thread * 64 + 6] * R[6];
       R[7] = DFTmatrix_large[thread * 64 + 7] * R[7];
       R[8] = DFTmatrix_large[thread * 64 + 8] * R[8];
       R[9] = DFTmatrix_large[thread * 64 + 9] * R[9];
       R[10] = DFTmatrix_large[thread * 64 + 10] * R[10];
       R[11] = DFTmatrix_large[thread * 64 + 11] * R[11];
       R[12] = DFTmatrix_large[thread * 64 + 12] * R[12];
       R[13] = DFTmatrix_large[thread * 64 + 13] * R[13];
       R[14] = DFTmatrix_large[thread * 64 + 14] * R[14];
       R[15] = DFTmatrix_large[thread * 64 + 15] * R[15];
}

#endif

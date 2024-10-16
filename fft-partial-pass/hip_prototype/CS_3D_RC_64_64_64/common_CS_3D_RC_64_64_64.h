#ifndef COMMON_CS_3D_RC_64_64_64_H
#define COMMON_CS_3D_RC_64_64_64_H

#include "../rocfft/callback.h"
#include "../rocfft/butterfly_constant.h"
#include "../rocfft/rocfft_complex.h"
#include "../rocfft/common.h"
#include "../rocfft/real2complex_device.h"
#include "../rocfft/rocfft_butterfly_template.h"
#include <hip/hip_runtime.h>

typedef rocfft_complex<double> scalar_type;

template <typename scalar_type, StrideBin sb>
__device__ void lds_to_reg_input_length64_device(scalar_type *R,
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
__device__ void lds_from_reg_output_length64_device(scalar_type *R,
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

#endif
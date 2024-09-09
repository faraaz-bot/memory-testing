#ifndef SBRR_64_H
#define SBRR_64_H

#include "common_CS_3D_RC_64_64_64.h"

template <typename scalar_type,
          const bool lds_is_real,
          StrideBin sb,
          const bool lds_linear,
          const bool direct_load_to_reg>
__device__ void forward_length64_SBRR_device(scalar_type *R,
                                             real_type_t<scalar_type> *__restrict__ lds_real,
                                             scalar_type *__restrict__ lds_complex,
                                             const scalar_type *__restrict__ twiddles,
                                             unsigned int stride_lds,
                                             unsigned int offset_lds,
                                             unsigned int thread,
                                             bool write)
{
    scalar_type W;
    scalar_type t;
    const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    unsigned int l_offset;

    // pass 0, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    FwdRad8B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7);
    if (!lds_is_real)
    {
        if (!direct_load_to_reg)
        {
            __syncthreads();
        }

        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
        lds_complex[l_offset] = R[0];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
        lds_complex[l_offset] = R[1];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
        lds_complex[l_offset] = R[2];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
        lds_complex[l_offset] = R[3];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
        lds_complex[l_offset] = R[4];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
        lds_complex[l_offset] = R[5];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
        lds_complex[l_offset] = R[6];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
        lds_complex[l_offset] = R[7];
    }

    else
    {
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
        lds_real[l_offset] = R[0].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
        lds_real[l_offset] = R[1].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
        lds_real[l_offset] = R[2].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
        lds_real[l_offset] = R[3].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
        lds_real[l_offset] = R[4].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
        lds_real[l_offset] = R[5].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
        lds_real[l_offset] = R[6].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
        lds_real[l_offset] = R[7].x;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0].x = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
        R[1].x = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
        R[2].x = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
        R[3].x = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
        R[4].x = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
        R[5].x = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
        R[6].x = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
        R[7].x = lds_real[l_offset];
        __syncthreads();
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
        lds_real[l_offset] = R[0].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
        lds_real[l_offset] = R[1].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
        lds_real[l_offset] = R[2].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
        lds_real[l_offset] = R[3].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
        lds_real[l_offset] = R[4].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
        lds_real[l_offset] = R[5].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
        lds_real[l_offset] = R[6].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
        lds_real[l_offset] = R[7].y;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0].y = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
        R[1].y = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
        R[2].y = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
        R[3].y = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
        R[4].y = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
        R[5].y = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
        R[6].y = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
        R[7].y = lds_real[l_offset];
    }

    // pass 1, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    if (!lds_is_real)
    {
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

    W = twiddles[0 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
    R[1] = t;
    W = twiddles[1 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
    R[2] = t;
    W = twiddles[2 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
    R[3] = t;
    W = twiddles[3 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
    R[4] = t;
    W = twiddles[4 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[5].x * W.x - R[5].y * W.y, R[5].y * W.x + R[5].x * W.y};
    R[5] = t;
    W = twiddles[5 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[6].x * W.x - R[6].y * W.y, R[6].y * W.x + R[6].x * W.y};
    R[6] = t;
    W = twiddles[6 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[7].x * W.x - R[7].y * W.y, R[7].y * W.x + R[7].x * W.y};
    R[7] = t;
    FwdRad8B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7);
}

extern "C" __global__
    __launch_bounds__(64) void fft_rtc_fwd_len64_factors_8_8_wgs_64_tpt_8_dim3_dp_ip_CI_unitstride_sbrr(
        const scalar_type *__restrict__ twiddles,
        const size_t *__restrict__ lengths,
        const size_t *__restrict__ stride,
        const size_t nbatch,
        const unsigned int lds_padding,
        void *__restrict__ load_cb_fn,
        void *__restrict__ load_cb_data,
        unsigned int load_cb_lds_bytes,
        void *__restrict__ store_cb_fn,
        void *__restrict__ store_cb_data,
        scalar_type *__restrict__ buf)
{
    const StrideBin sb = SB_UNIT;
    const EmbeddedType ebtype = EmbeddedType::NONE;
    const SBRC_TYPE sbrc_type = SBRC_2D;
    const SBRC_TRANSPOSE_TYPE transpose_type = NONE;
    const CallbackType cbtype = CallbackType::NONE;
    const DirectRegType drtype = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;
    const bool apply_large_twiddle = false;
    const IntrinsicAccessType intrinsic_mode = IntrinsicAccessType::DISABLE_BOTH;
    const size_t large_twiddle_base = 8;
    const size_t large_twiddle_steps = 0;

    // this kernel:
    //   uses 8 threads per transform
    //   does 8 transforms per thread block
    // therefore it should be called with 64 threads per thread block
    scalar_type R[8];
    extern __shared__ unsigned char __attribute__((aligned(sizeof(scalar_type)))) lds_uchar[];
    real_type_t<scalar_type> *__restrict__ lds_real = reinterpret_cast<real_type_t<scalar_type> *>(lds_uchar);
    scalar_type *__restrict__ lds_complex = reinterpret_cast<scalar_type *>(lds_uchar);
    size_t offset = 0;
    unsigned int offset_lds;
    unsigned int stride_lds;
    size_t batch;
    size_t transform;
    const bool direct_load_to_reg = false;
    const bool direct_store_from_reg = false;
    const bool lds_linear = true;
    const bool lds_is_real = false;
    auto load_cb = get_load_cb<scalar_type, cbtype>(load_cb_fn);
    auto store_cb = get_store_cb<scalar_type, cbtype>(store_cb_fn);

    // large twiddles
    // - no large twiddles

    // offsets
    const size_t dim = 3;
    const size_t stride0 = (sb == SB_UNIT) ? (1) : (stride[0]);
    unsigned int thread;
    size_t remaining;
    size_t index_along_d;
    transform = blockIdx.x * 8 + threadIdx.x / 8;
    remaining = transform;
    for (int d = 1; d < dim; ++d)
    {
        index_along_d = remaining % lengths[d];
        remaining = remaining / lengths[d];
        offset = offset + index_along_d * stride[d];
    }
    batch = remaining;
    offset = offset + batch * stride[dim];
    stride_lds = 64 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding);
    offset_lds = stride_lds * (transform % 8);
    bool inbound = batch < nbatch;

    // load global into lds
    if (inbound)
    {
        thread = threadIdx.x % 8;
        lds_complex[offset_lds + (thread + 0)] = load_cb(buf, offset + (thread + 0) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + (thread + 8)] = load_cb(buf, offset + (thread + 8) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + (thread + 16)] = load_cb(buf, offset + (thread + 16) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + (thread + 24)] = load_cb(buf, offset + (thread + 24) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + (thread + 32)] = load_cb(buf, offset + (thread + 32) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + (thread + 40)] = load_cb(buf, offset + (thread + 40) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + (thread + 48)] = load_cb(buf, offset + (thread + 48) * stride0, load_cb_data, nullptr);
        lds_complex[offset_lds + (thread + 56)] = load_cb(buf, offset + (thread + 56) * stride0, load_cb_data, nullptr);

        // append extra global loading for C2Real pre-process only
        if (ebtype == EmbeddedType::C2Real_PRE)
        {
            // use the last thread of each transform to load one more element per row
            if (thread == 7)
            {
                lds_complex[offset_lds + thread + 56 + 1] = load_cb(buf, offset + (thread + 56 + 1) * stride0, load_cb_data, nullptr);
            }
        }
    }

    // handle even-length real to complex pre-process in lds before transform
    if (ebtype == EmbeddedType::C2Real_PRE)
    {
        __syncthreads();

        real_pre_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 0,
                                                           64 - threadIdx.x % 8 - 0,
                                                           32,
                                                           lds_complex + offset_lds,
                                                           0,
                                                           twiddles + 56);
        real_pre_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 8,
                                                           64 - threadIdx.x % 8 - 8,
                                                           32,
                                                           lds_complex + offset_lds,
                                                           0,
                                                           twiddles + 56);
        real_pre_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 16,
                                                           64 - threadIdx.x % 8 - 16,
                                                           32,
                                                           lds_complex + offset_lds,
                                                           0,
                                                           twiddles + 56);
        real_pre_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 24,
                                                           64 - threadIdx.x % 8 - 24,
                                                           32,
                                                           lds_complex + offset_lds,
                                                           0,
                                                           twiddles + 56);
        __syncthreads();
    }

    // calc the thread_in_device value once and for all device funcs
    unsigned int thread_in_device = lds_linear ? threadIdx.x % 8 : threadIdx.x / 8;

    // call a pre-load from lds to registers (if necessary)
    lds_to_reg_input_length64_device<scalar_type, lds_linear ? SB_UNIT : SB_NONUNIT>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true);

    // transform
    forward_length64_SBRR_device<scalar_type,
                                 lds_is_real,
                                 lds_linear ? SB_UNIT : SB_NONUNIT,
                                 lds_linear,
                                 direct_load_to_reg>(
        R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, thread_in_device, true);

    // call a post-store from registers to lds (if necessary)
    lds_from_reg_output_length64_device<scalar_type, lds_linear ? SB_UNIT : SB_NONUNIT>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true);

    // handle even-length real to complex pre-process in lds after transform
    if (ebtype == EmbeddedType::Real2C_POST)
    {
        __syncthreads();

        real_post_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 0,
                                                            64 - threadIdx.x % 8 - 0,
                                                            32,
                                                            lds_complex + offset_lds,
                                                            0,
                                                            twiddles + 56);
        real_post_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 8,
                                                            64 - threadIdx.x % 8 - 8,
                                                            32,
                                                            lds_complex + offset_lds,
                                                            0,
                                                            twiddles + 56);
        real_post_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 16,
                                                            64 - threadIdx.x % 8 - 16,
                                                            32,
                                                            lds_complex + offset_lds,
                                                            0,
                                                            twiddles + 56);
        real_post_process_kernel_inplace<scalar_type, true>(threadIdx.x % 8 + 24,
                                                            64 - threadIdx.x % 8 - 24,
                                                            32,
                                                            lds_complex + offset_lds,
                                                            0,
                                                            twiddles + 56);
    }

    // store global
    __syncthreads();
    if (inbound)
    {
        store_cb(buf,
                 offset + (thread + 0) * stride0,
                 lds_complex[offset_lds + (thread + 0)],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 8) * stride0,
                 lds_complex[offset_lds + (thread + 8)],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 16) * stride0,
                 lds_complex[offset_lds + (thread + 16)],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 24) * stride0,
                 lds_complex[offset_lds + (thread + 24)],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 32) * stride0,
                 lds_complex[offset_lds + (thread + 32)],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 40) * stride0,
                 lds_complex[offset_lds + (thread + 40)],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 48) * stride0,
                 lds_complex[offset_lds + (thread + 48)],
                 store_cb_data,
                 nullptr);
        store_cb(buf,
                 offset + (thread + 56) * stride0,
                 lds_complex[offset_lds + (thread + 56)],
                 store_cb_data,
                 nullptr);

        // append extra global write for Real2C post-process only
        if (ebtype == EmbeddedType::Real2C_POST)
        {
            // use the last thread of each transform to write one more element per row
            if (thread == 7)
            {
                store_cb(buf,
                         offset + (thread + 56 + 1) * stride0,
                         lds_complex[offset_lds + thread + 56 + 1],
                         store_cb_data,
                         nullptr);
            }
        }
    }
}
#endif

#ifndef SBRR_PP_64_H
#define SBRR_PP_64_H

#include "pp_common.h"

template <typename scalar_type,
          const bool lds_is_real,
          StrideBin sb,
          const bool lds_linear,
          const bool direct_load_to_reg>
__device__ void forward_length64_SBRR_device_pp(scalar_type *R,
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

    __syncthreads();
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

    // pass 1, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
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
    __launch_bounds__(128) void fft_rtc_fwd_len64_factors_8_8_wgs_128_tpt_8_dim3_dp_ip_CI_unitstride_sbrr_pp(
        const scalar_type *__restrict__ DFTmatrix_large,
        const scalar_type *__restrict__ twiddles,
        const size_t *__restrict__ lengths,
        const size_t *__restrict__ stride,
        const size_t nbatch,
        const size_t N,
        const unsigned int lds_padding,
        void *__restrict__ load_cb_fn,
        void *__restrict__ load_cb_data,
        unsigned int load_cb_lds_bytes,
        void *__restrict__ store_cb_fn,
        void *__restrict__ store_cb_data,
        scalar_type *__restrict__ ibuf,
        scalar_type *__restrict__ obuf)
{
    auto const sb = SB_UNIT;
    auto const ebtype = EmbeddedType::NONE;
    auto const sbrc_type = SBRC_2D;
    auto const transpose_type = NONE;
    auto const cbtype = CallbackType::NONE;
    auto const drtype = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;
    auto const apply_large_twiddle = false;
    auto const intrinsic_mode = IntrinsicAccessType::DISABLE_BOTH;
    const size_t large_twiddle_base = 8;
    const size_t large_twiddle_steps = 0;

    // this kernel:
    //   uses 8 threads per transform
    //   does 16 transforms per thread block
    // therefore it should be called with 128 threads per thread block
    scalar_type R[16];
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

    size_t global_mem_idx = 0, offset_pp = 0, remaining_pp = 0;

    const StrideBin SB_1ST = (ebtype == EmbeddedType::C2Real_PRE) ? SB_NONUNIT : SB_UNIT;
    const StrideBin SB_2ND = (ebtype == EmbeddedType::C2Real_PRE) ? SB_UNIT : SB_NONUNIT;

    // large twiddles
    // - no large twiddles

    // offsets
    const size_t dim = 3;
    // const size_t stride0 = (sb == SB_UNIT) ? (1) : (stride[0]);
    const size_t stride0 = (stride[0]);
    unsigned int thread;
    size_t remaining;
    size_t index_along_d;
    transform = blockIdx.x * 16 + threadIdx.x / 8;
    remaining = transform;
    remaining_pp = 64 * (transform / 64) + (transform % 64) / 16 + (transform * 4) % 64;
    for (int d = 1; d < dim; ++d)
    {
        // index_along_d = remaining % lengths[d];
        remaining = remaining / lengths[d];
        // offset = offset + index_along_d * stride[d];

        index_along_d = remaining_pp % lengths[d];
        remaining_pp = remaining_pp / lengths[d];
        offset_pp = offset_pp + index_along_d * stride[d];
    }
    batch = remaining;
    // offset = offset + batch * stride[dim];
    offset_pp = offset_pp + batch * stride[dim];
    stride_lds = 64 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding);
    offset_lds = stride_lds * (transform % 16);
    bool inbound = batch < nbatch;

    // load global into lds
    if (inbound)
    {
        thread = threadIdx.x % 8;

        global_mem_idx = offset_pp + (thread + 0) * stride0;
        lds_complex[offset_lds + (thread + 0)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 8) * stride0;
        lds_complex[offset_lds + (thread + 8)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 16) * stride0;
        lds_complex[offset_lds + (thread + 16)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 24) * stride0;
        lds_complex[offset_lds + (thread + 24)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 32) * stride0;
        lds_complex[offset_lds + (thread + 32)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 40) * stride0;
        lds_complex[offset_lds + (thread + 40)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 48) * stride0;
        lds_complex[offset_lds + (thread + 48)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 56) * stride0;
        lds_complex[offset_lds + (thread + 56)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);
    }

    // calc the thread_in_device value once and for all device funcs
    unsigned int thread_in_device = lds_linear ? threadIdx.x % 8 : threadIdx.x / 16;

    // call a pre-load from lds to registers (if necessary)
    lds_to_reg_input_length64_device_sbrr<scalar_type, SB_1ST>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true);

    // transform
    forward_length64_SBRR_device_pp<scalar_type,
                                    lds_is_real,
                                    SB_1ST,
                                    lds_linear,
                                    direct_load_to_reg>(
        R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, thread_in_device, true);

    // call a post-store from registers to lds (if necessary)
    lds_from_reg_output_length64_device_sbrr<scalar_type, SB_1ST>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true);

    auto stride_lds_pp = 64;
    auto offset_lds_pp = (blockIdx.x * 16 + threadIdx.x) % 64;

    // call a pre-load from lds to registers (if necessary)
    lds_to_reg_16_input_length64_device_pp<scalar_type>(R, lds_complex, stride_lds_pp, offset_lds_pp);

    // Partial pass step 1: length-16 DFT on off-dimension

    // Radix-16 pass
    FwdRad16B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7, R + 8, R + 9, R + 10, R + 11, R + 12, R + 13, R + 14, R + 15);

    // Partial pass step 2: Hadamard product with twiddle factors
    twiddle_multiple_pp<scalar_type>(R, blockIdx.x % 4, DFTmatrix_large);

    // call a post-store from registers to lds (if necessary)
    lds_from_reg_16_output_length64_device_pp<scalar_type>(R, lds_complex, stride_lds_pp, offset_lds_pp);

    // =======================================================================================================

    // store global
    __syncthreads();
    if (inbound)
    {
        global_mem_idx = offset_pp + (thread + 0) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 0)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 8) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 8)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 16) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 16)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 24) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 24)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 32) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 32)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 40) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 40)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 48) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 48)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 56) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 56)],
                 store_cb_data,
                 nullptr);
    }
}
#endif

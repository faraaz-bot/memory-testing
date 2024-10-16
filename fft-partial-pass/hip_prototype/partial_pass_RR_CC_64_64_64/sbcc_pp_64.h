#ifndef SBCC_PP_64_H
#define SBCC_PP_64_H

#include "pp_common.h"

template <typename scalar_type,
          const bool lds_is_real,
          StrideBin sb,
          const bool lds_linear,
          const bool direct_load_to_reg,
          bool apply_large_twiddle,
          size_t large_twiddle_steps = 3,
          size_t large_twiddle_base = 8>
__device__ void forward_length64_SBCC_device_pp(scalar_type *R,
                                                real_type_t<scalar_type> *__restrict__ lds_real,
                                                scalar_type *__restrict__ lds_complex,
                                                const scalar_type *__restrict__ twiddles,
                                                unsigned int stride_lds,
                                                unsigned int offset_lds,
                                                unsigned int thread_twd,
                                                unsigned int thread,
                                                bool write,
                                                const scalar_type *large_twiddles,
                                                size_t trans_local)
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

        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (0 * 4)) * lstride;
        lds_complex[l_offset] = R[0];
        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (1 * 4)) * lstride;
        lds_complex[l_offset] = R[1];
        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (2 * 4)) * lstride;
        lds_complex[l_offset] = R[2];
        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (3 * 4)) * lstride;
        lds_complex[l_offset] = R[3];
        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (4 * 4)) * lstride;
        lds_complex[l_offset] = R[4];
        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (5 * 4)) * lstride;
        lds_complex[l_offset] = R[5];
        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (6 * 4)) * lstride;
        lds_complex[l_offset] = R[6];
        l_offset = offset_lds + (((thread + 0 + 0) / (1 * 4)) * (8 * 4) + (thread + 0 + 0) % (1 * 4) + (7 * 4)) * lstride;
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
        l_offset = offset_lds + ((thread + 0 + 0) + (0 * 4)) * lstride;
        R[0] = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + (8 * 4)) * lstride;
        R[1] = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + (16 * 4)) * lstride;
        R[2] = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + (24 * 4)) * lstride;
        R[3] = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + (32 * 4)) * lstride;
        R[4] = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + (40 * 4)) * lstride;
        R[5] = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + (48 * 4)) * lstride;
        R[6] = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + (56 * 4)) * lstride;
        R[7] = lds_complex[l_offset];
    }

    W = twiddles[0 + 7 * ((thread_twd + 0 + 0) % 8)];
    t = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
    R[1] = t;
    W = twiddles[1 + 7 * ((thread_twd + 0 + 0) % 8)];
    t = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
    R[2] = t;
    W = twiddles[2 + 7 * ((thread_twd + 0 + 0) % 8)];
    t = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
    R[3] = t;
    W = twiddles[3 + 7 * ((thread_twd + 0 + 0) % 8)];
    t = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
    R[4] = t;
    W = twiddles[4 + 7 * ((thread_twd + 0 + 0) % 8)];
    t = {R[5].x * W.x - R[5].y * W.y, R[5].y * W.x + R[5].x * W.y};
    R[5] = t;
    W = twiddles[5 + 7 * ((thread_twd + 0 + 0) % 8)];
    t = {R[6].x * W.x - R[6].y * W.y, R[6].y * W.x + R[6].x * W.y};
    R[6] = t;
    W = twiddles[6 + 7 * ((thread_twd + 0 + 0) % 8)];
    t = {R[7].x * W.x - R[7].y * W.y, R[7].y * W.x + R[7].x * W.y};
    R[7] = t;
    FwdRad8B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7);
    if (apply_large_twiddle)
    {
        // large twiddle multiplication
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 0 * 8) * trans_local);
        t = {R[0].x * W.x - R[0].y * W.y, R[0].y * W.x + R[0].x * W.y};
        R[0] = t;
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 1 * 8) * trans_local);
        t = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
        R[1] = t;
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 2 * 8) * trans_local);
        t = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
        R[2] = t;
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 3 * 8) * trans_local);
        t = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
        R[3] = t;
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 4 * 8) * trans_local);
        t = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
        R[4] = t;
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 5 * 8) * trans_local);
        t = {R[5].x * W.x - R[5].y * W.y, R[5].y * W.x + R[5].x * W.y};
        R[5] = t;
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 6 * 8) * trans_local);
        t = {R[6].x * W.x - R[6].y * W.y, R[6].y * W.x + R[6].x * W.y};
        R[6] = t;
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 8) + 7 * 8) * trans_local);
        t = {R[7].x * W.x - R[7].y * W.y, R[7].y * W.x + R[7].x * W.y};
        R[7] = t;
    }
}

extern "C" __global__
    __launch_bounds__(256) void fft_rtc_fwd_len64_factors_8_8_wgs_64_tpt_8_dim3_dp_ip_CI_sbcc_dirReg_intrinsicRead_pp(
        const scalar_type *__restrict__ twiddles,
        const scalar_type *large_twiddles,
        const size_t *__restrict__ lengths,
        const size_t *__restrict__ stride,
        const size_t nbatch,
        const unsigned int lds_padding,
        void *__restrict__ load_cb_fn,
        void *__restrict__ load_cb_data,
        unsigned int load_cb_lds_bytes,
        void *__restrict__ store_cb_fn,
        void *__restrict__ store_cb_data,
        scalar_type *__restrict__ ibuf,
        scalar_type *__restrict__ obuf)
{
    auto const sb = SB_NONUNIT;
    auto const ebtype = EmbeddedType::NONE;
    auto const sbrc_type = SBRC_2D;
    auto const transpose_type = NONE;
    auto const cbtype = CallbackType::NONE;
    auto const drtype = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;
    auto const apply_large_twiddle = false;
    auto const intrinsic_mode = IntrinsicAccessType::DISABLE_BOTH;
    const size_t large_twiddle_base = 8;
    const size_t large_twiddle_steps = 0;

    scalar_type R[8];
    extern __shared__ unsigned char __attribute__((aligned(sizeof(scalar_type)))) lds_uchar[];
    real_type_t<scalar_type> *__restrict__ lds_real = reinterpret_cast<real_type_t<scalar_type> *>(lds_uchar);
    scalar_type *__restrict__ lds_complex = reinterpret_cast<scalar_type *>(lds_uchar);
    size_t offset = 0;
    unsigned int offset_lds;
    unsigned int stride_lds;
    size_t batch;
    size_t transform;
    const bool direct_load_to_reg = drtype == DirectRegType::TRY_ENABLE_IF_SUPPORT;
    const bool direct_store_from_reg = direct_load_to_reg;
    const bool lds_linear = !direct_load_to_reg;
    const bool lds_is_real = false;
    auto load_cb = get_load_cb<scalar_type, cbtype>(load_cb_fn);
    auto store_cb = get_store_cb<scalar_type, cbtype>(store_cb_fn);

    // large twiddles
    __shared__ scalar_type large_twd_lds[(apply_large_twiddle && large_twiddle_base < 8)
                                             ? ((1 << large_twiddle_base) * 3)
                                             : (0)];
    if (apply_large_twiddle && large_twiddle_base < 8)
    {
        size_t ltwd_id = threadIdx.x;
        while (ltwd_id < (1 << large_twiddle_base) * 3)
        {
            large_twd_lds[ltwd_id] = large_twiddles[ltwd_id];
            ltwd_id += 64;
        }
    }

    // offsets
    const size_t dim = 3;
    const size_t stride0 = (sb == SB_UNIT) ? (1) : (stride[0]);
    size_t tile_index;
    size_t num_of_tiles;

    // calculate offset for each tile:
    //   tile_index  now means index of the tile along dim1
    //   num_of_tiles now means number of tiles along dim1
    size_t plength = 1;
    size_t remaining;
    size_t index_along_d;
    num_of_tiles = (lengths[1] - 1) / 8 + 1;
    plength = num_of_tiles;
    tile_index = blockIdx.x % num_of_tiles;

    // mod 128 required to work with nbatch > 1
    remaining = (blockIdx.x % 128) / num_of_tiles;
    offset = tile_index * 8 * stride[1];
    for (int d = 2; d < dim; ++d)
    {
        plength = plength * lengths[d];

        index_along_d = remaining % lengths[d];
        remaining = remaining / lengths[d];
        offset = offset + index_along_d * stride[d];
    }

    batch = blockIdx.x / plength;
    // offset = offset + batch * stride[dim]; // don't add batch here
    transform = lds_linear ? tile_index * 8 + threadIdx.x / 8 : tile_index * 8 + threadIdx.x % 8;
    stride_lds = lds_linear ? 64 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding)
                            : 8 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding);
    stride_lds *= 4;

    offset_lds = lds_linear ? stride_lds * (transform % 8) : threadIdx.x % 8;
    bool in_bound = ((tile_index + 1) * 8 > lengths[1]) ? false : true;
    unsigned int thread = threadIdx.x / 8;
    unsigned int tid_hor = threadIdx.x % 8;

    unsigned int thread_lds = threadIdx.x / 8;
    unsigned int tid_hor_lds = threadIdx.x % 8;

    auto tid_hor_pp = threadIdx.x % 8 + 64 * (thread % 4);

    auto thread_new = threadIdx.x / (8 * 4);
    auto batch_new = blockIdx.x / (plength / 4);

    auto thread_idx = threadIdx.x;
    auto block_idx = blockIdx.x;

    auto offset_pp = offset + (offset / 64) * 192 + batch_new * stride[dim];

    auto offset_tid_hor = (offset_pp + tid_hor_pp * stride[1]);

    transform = lds_linear ? tile_index * 8 + threadIdx.x / (8 * 4) : tile_index * 8 + threadIdx.x % 8;
    offset_lds = lds_linear ? stride_lds * (transform % 8) : threadIdx.x % 8;

    size_t global_mem_idx = 0;

    // load global into lds
    // no intrinsic when load to lds. FIXME- check why use nested branch is better
    if (in_bound)
    {
        global_mem_idx = offset_tid_hor + (thread_new + 0) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 0) * 1] = load_cb(ibuf,
                                                                               global_mem_idx,
                                                                               load_cb_data,
                                                                               nullptr);
        global_mem_idx = offset_tid_hor + (thread_new + 8) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 8 * 4) * 1] = load_cb(ibuf,
                                                                                   global_mem_idx,
                                                                                   load_cb_data,
                                                                                   nullptr);
        global_mem_idx = offset_tid_hor + (thread_new + 16) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 16 * 4) * 1] = load_cb(ibuf,
                                                                                    global_mem_idx,
                                                                                    load_cb_data,
                                                                                    nullptr);
        global_mem_idx = offset_tid_hor + (thread_new + 24) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 24 * 4) * 1] = load_cb(ibuf,
                                                                                    global_mem_idx,
                                                                                    load_cb_data,

                                                                                    nullptr);
        global_mem_idx = offset_tid_hor + (thread_new + 32) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 32 * 4) * 1] = load_cb(ibuf,
                                                                                    global_mem_idx,
                                                                                    load_cb_data,
                                                                                    nullptr);
        global_mem_idx = offset_tid_hor + (thread_new + 40) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 40 * 4) * 1] = load_cb(ibuf,
                                                                                    global_mem_idx,
                                                                                    load_cb_data,
                                                                                    nullptr);
        global_mem_idx = offset_tid_hor + (thread_new + 48) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 48 * 4) * 1] = load_cb(ibuf,
                                                                                    global_mem_idx,
                                                                                    load_cb_data,
                                                                                    nullptr);
        global_mem_idx = offset_tid_hor + (thread_new + 56) * stride0;
        lds_complex[tid_hor_lds * stride_lds + (thread_lds + 56 * 4) * 1] = load_cb(ibuf,
                                                                                    global_mem_idx,
                                                                                    load_cb_data,
                                                                                    nullptr);
    }

    if (!in_bound)
    {
        if (tile_index * 8 + tid_hor < lengths[1])
        {
            global_mem_idx = offset_tid_hor + (thread_new + 0) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 0) * 1] = load_cb(ibuf,
                                                                                   global_mem_idx,
                                                                                   load_cb_data,
                                                                                   nullptr);
            global_mem_idx = offset_tid_hor + (thread_new + 8) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 8 * 4) * 1] = load_cb(ibuf,
                                                                                       global_mem_idx,
                                                                                       load_cb_data,
                                                                                       nullptr);
            global_mem_idx = offset_tid_hor + (thread_new + 16) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 16 * 4) * 1] = load_cb(ibuf,
                                                                                        global_mem_idx,
                                                                                        load_cb_data,
                                                                                        nullptr);
            global_mem_idx = offset_tid_hor + (thread_new + 24) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 24 * 4) * 1] = load_cb(ibuf,
                                                                                        global_mem_idx,
                                                                                        load_cb_data,
                                                                                        nullptr);
            global_mem_idx = offset_tid_hor + (thread_new + 32) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 32 * 4) * 1] = load_cb(ibuf,
                                                                                        global_mem_idx,
                                                                                        load_cb_data,
                                                                                        nullptr);
            global_mem_idx = offset_tid_hor + (thread_new + 40) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 40 * 4) * 1] = load_cb(ibuf,
                                                                                        global_mem_idx,
                                                                                        load_cb_data,
                                                                                        nullptr);
            global_mem_idx = offset_tid_hor + (thread_new + 48) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 48 * 4) * 1] = load_cb(ibuf,
                                                                                        global_mem_idx,
                                                                                        load_cb_data,
                                                                                        nullptr);
            global_mem_idx = offset_tid_hor + (thread_new + 56) * stride0;
            lds_complex[tid_hor_lds * stride_lds + (thread_lds + 56 * 4) * 1] = load_cb(ibuf,
                                                                                        global_mem_idx,
                                                                                        load_cb_data,
                                                                                        nullptr);
        }
    }

    auto stride_lds_pp = 1;
    auto offset_lds_pp = threadIdx.x * 8;

    // Partial pass step 3: local transposition performed on store to global memory below

    // call a pre-load from lds to registers (if necessary)
    lds_to_reg_4_input_length64_device_pp<scalar_type>(R, lds_complex, stride_lds_pp, offset_lds_pp);

    // Partial pass step 4: length-4 DFT on off-dimension

    // Radix-4 pass
    FwdRad4B1(R + 0, R + 1, R + 2, R + 3);

    // Radix-4 pass
    FwdRad4B1(R + 4, R + 5, R + 6, R + 7);

    // call a post-store from registers to lds (if necessary)
    lds_from_reg_4_output_length64_device_pp<scalar_type>(R, lds_complex, stride_lds_pp, offset_lds_pp);

    // calc the thread_in_device value once and for all device funcs
    auto thread_in_device = lds_linear ? threadIdx.x % (8 * 4) : threadIdx.x / 8;
    auto thread_in_device_twd = (threadIdx.x / 4) % (8);

    // call a pre-load from lds to registers (if necessary)
    lds_to_reg_input_length64_device_sbcc<scalar_type, lds_linear ? SB_UNIT : SB_NONUNIT>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true);

    // transform
    forward_length64_SBCC_device_pp<scalar_type,
                                    lds_is_real,
                                    lds_linear ? SB_UNIT : SB_NONUNIT,
                                    lds_linear,
                                    direct_load_to_reg,
                                    apply_large_twiddle,
                                    large_twiddle_steps,
                                    large_twiddle_base>(
        R,
        lds_real,
        lds_complex,
        twiddles,
        stride_lds,
        offset_lds,
        thread_in_device_twd,
        thread_in_device,
        true,
        (apply_large_twiddle && large_twiddle_base < 8) ? (large_twd_lds) : (large_twiddles),
        transform);

    // call a post-store from registers to lds (if necessary)
    lds_from_reg_output_length64_device_sbcc<scalar_type, lds_linear ? SB_UNIT : SB_NONUNIT>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true);

    // store global
    __syncthreads();
    // no intrinsic when store from lds. FIXME- check why use nested branch is better
    if (in_bound)
    {
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 0) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 0) * 1],
                 store_cb_data,
                 nullptr);
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 8) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 8 * 4) * 1],
                 store_cb_data,
                 nullptr);
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 16) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 16 * 4) * 1],
                 store_cb_data,
                 nullptr);
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 24) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 24 * 4) * 1],
                 store_cb_data,
                 nullptr);
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 32) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 32 * 4) * 1],
                 store_cb_data,
                 nullptr);
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 40) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 40 * 4) * 1],
                 store_cb_data,
                 nullptr);
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 48) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 48 * 4) * 1],
                 store_cb_data,
                 nullptr);
        global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 56) * stride0);
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[tid_hor_lds * stride_lds + (thread_lds + 56 * 4) * 1],
                 store_cb_data,
                 nullptr);
    }

    if (!in_bound)
    {
        if (tile_index * 8 + tid_hor < lengths[1])
        {
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 0) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 0 * 4) * 1],
                     store_cb_data,
                     nullptr);
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 8) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 8 * 4) * 1],
                     store_cb_data,
                     nullptr);
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 16) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 16 * 4) * 1],
                     store_cb_data,
                     nullptr);
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 24) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 24 * 4) * 1],
                     store_cb_data,
                     nullptr);
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 32) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 32 * 4) * 1],
                     store_cb_data,
                     nullptr);
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 40) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 40 * 4) * 1],
                     store_cb_data,
                     nullptr);
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 48) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 48 * 4) * 1],
                     store_cb_data,
                     nullptr);
            global_mem_idx = apply_local_transpose(offset_tid_hor + (thread_new + 56) * stride0);
            store_cb(obuf,
                     global_mem_idx,
                     lds_complex[tid_hor_lds * stride_lds + (thread_lds + 56 * 4) * 1],
                     store_cb_data,
                     nullptr);
        }
    }
}

#endif

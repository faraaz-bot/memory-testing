// ROCFFT_RTC_BEGIN fft_rtc_fwd_len125_sp_ip_CI_sbcc_dirReg
#define ROCFFT_CALLBACKS_ENABLED

// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef COMMON_H
#define COMMON_H

#ifdef WIN32
#define ROCFFT_DEVICE_EXPORT __declspec(dllexport)
#else
#define ROCFFT_DEVICE_EXPORT
#endif

// NB:
//   All kernels were compiled based on the assumption that the default max
//   work group size is 256. This default value in compiler might change in
//   future. Each kernel has to explicitly set proper sizes through
//   __launch_bounds__ or __attribute__.
//   Further performance tuning might be done later.
static const unsigned int LAUNCH_BOUNDS_R2C_C2R_KERNEL = 256;

#ifdef __HIP_PLATFORM_NVIDIA__

__device__ inline float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ inline float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ inline float2 operator*(const float& a, const float2& b)
{
    return make_float2(a * b.x, a * b.y);
}
__device__ inline float2 operator*=(float2& a, const float2& b)
{
    a = cuCmulf(a, b);
    return a;
}
__device__ inline float2 operator*=(float2& a, const float& b)
{
    a = cuCmulf(a, make_float2(b, b));
    return a;
}
__device__ inline float2 operator-(const float2& a)
{
    return cuCmulf(a, make_float2(-1.0, -1.0));
}

__device__ inline double2 operator-(const double2& a, const double2& b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
__device__ inline double2 operator+(const double2& a, const double2& b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ inline double2 operator*(const double& a, const double2& b)
{
    return make_double2(a * b.x, a * b.y);
}
__device__ inline double2 operator*=(double2& a, const double2& b)
{
    a = cuCmul(a, b);
    return a;
}
__device__ inline double2 operator*=(double2& a, const double& b)
{
    a = cuCmul(a, make_double2(b, b));
    return a;
}
__device__ inline double2 operator-(const double2& a)
{
    return cuCmul(a, make_double2(-1.0, -1.0));
}

#endif

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

enum class EmbeddedType
{
    NONE, // Works as the regular complex to complex FFT kernel
    Real2C_POST, // Works with even-length real2complex post-processing
    C2Real_PRE, // Works with even-length complex2real pre-processing
};

// TODO: rework this
//
//
// NB:
// SBRC kernels can be used in various scenarios. Instead of tmeplate all
// combinations, we define/enable the cases in using only. In this way,
// the logic in POWX_LARGE_SBRC_GENERATOR() would be simple. People could
// add more later or find a way to simply POWX_LARGE_SBRC_GENERATOR().
enum SBRC_TYPE
{
    SBRC_2D = 2, // for one step in 1D middle size decomposition

    SBRC_3D_FFT_TRANS_XY_Z = 3, // for 3D C2C middle size fused kernel
    SBRC_3D_FFT_TRANS_Z_XY = 4, // for 3D R2C middle size fused kernel
    SBRC_3D_TRANS_XY_Z_FFT = 5, // for 3D C2R middle size fused kernel

    // for 3D R2C middle size, to fuse FFT, Even-length real2complex, and Transpose_Z_XY
    SBRC_3D_FFT_ERC_TRANS_Z_XY = 6,

    // for 3D C2R middle size, to fuse Transpose_XY_Z, Even-length complex2real, and FFT
    SBRC_3D_TRANS_XY_Z_ECR_FFT = 7,
};

enum SBRC_TRANSPOSE_TYPE
{
    NONE,
    // best, but requires cube sizes
    DIAGONAL,
    // OK, doesn't require handling unaligned corner case
    TILE_ALIGNED,
    TILE_UNALIGNED,
};

enum DirectRegType
{
    // the direct-to-from-reg codes are not even generated from generator
    // or is generated but we don't want to use it in some arch
    FORCE_OFF_OR_NOT_SUPPORT,
    TRY_ENABLE_IF_SUPPORT, // Use the direct-to-from-reg function
};

enum IntrinsicAccessType
{
    DISABLE_BOTH, // turn-off intrinsic buffer load/store
    ENABLE_LOAD_ONLY, // turn-on intrinsic buffer load only
    ENABLE_BOTH, // turn-on both intrinsic buffer load/store
};

template <class T>
struct real_type;

template <>
struct real_type<float4>
{
    typedef float type;
};

template <>
struct real_type<double4>
{
    typedef double type;
};

template <>
struct real_type<float2>
{
    typedef float type;
};

template <>
struct real_type<double2>
{
    typedef double type;
};

template <class T>
using real_type_t = typename real_type<T>::type;

/* example of using real_type_t */
// real_type_t<float2> float_scalar;
// real_type_t<double2> double_scalar;

template <class T>
struct complex_type;

template <>
struct complex_type<float>
{
    typedef float2 type;
};

template <>
struct complex_type<double>
{
    typedef double2 type;
};

template <class T>
using complex_type_t = typename complex_type<T>::type;

/// example of using complex_type_t:
// complex_type_t<float> float_complex_val;
// complex_type_t<double> double_complex_val;

template <class T>
struct vector4_type;

template <>
struct vector4_type<float2>
{
    typedef float4 type;
};

template <>
struct vector4_type<double2>
{
    typedef double4 type;
};

template <class T>
using vector4_type_t = typename vector4_type<T>::type;

/* example of using vector4_type_t */
// vector4_type_t<float2> float4_scalar;
// vector4_type_t<double2> double4_scalar;

template <typename T>
__device__ inline T lib_make_vector2(real_type_t<T> v0, real_type_t<T> v1);

template <>
__device__ inline float2 lib_make_vector2(float v0, float v1)
{
    return make_float2(v0, v1);
}

template <>
__device__ inline double2 lib_make_vector2(double v0, double v1)
{
    return make_double2(v0, v1);
}

template <typename T>
__device__ inline T
    lib_make_vector4(real_type_t<T> v0, real_type_t<T> v1, real_type_t<T> v2, real_type_t<T> v3);

template <>
__device__ inline float4 lib_make_vector4(float v0, float v1, float v2, float v3)
{
    return make_float4(v0, v1, v2, v3);
}

template <>
__device__ inline double4 lib_make_vector4(double v0, double v1, double v2, double v3)
{
    return make_double4(v0, v1, v2, v3);
}

template <typename T>
__device__ T TWLstep1(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    return result;
}

template <typename T>
__device__ T TWLstep2(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    return result;
}

template <typename T>
__device__ T TWLstep3(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    return result;
}

template <typename T>
__device__ T TWLstep4(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[768 + j].x - result.y * twiddles[768 + j].y),
                                 (result.y * twiddles[768 + j].x + result.x * twiddles[768 + j].y));
    return result;
}

#define TWIDDLE_STEP_MUL_FWD(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T              W = TWFUNC(TWIDDLES, INDEX);        \
        real_type_t<T> TR, TI;                             \
        TR    = (W.x * REG.x) - (W.y * REG.y);             \
        TI    = (W.y * REG.x) + (W.x * REG.y);             \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#define TWIDDLE_STEP_MUL_INV(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T              W = TWFUNC(TWIDDLES, INDEX);        \
        real_type_t<T> TR, TI;                             \
        TR    = (W.x * REG.x) + (W.y * REG.y);             \
        TI    = -(W.y * REG.x) + (W.x * REG.y);            \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#endif // COMMON_H

/******************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *****************************************************************************/
/*! \file
    \brief Architecture-specific operators on memory added for GFX9
*/
// reference:
//   https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/llvm.amdgcn.raw.buffer.load.ll

#ifndef INTRINSIC_MEM_ACCESS_H
#define INTRINSIC_MEM_ACCESS_H

#if defined(__clang__) && defined(__HIP__)

#if(defined(__NVCC__) || defined(__HIPCC__)) \
    || (defined(__clang__) && (defined(__CUDA__)) || defined(__HIP__))
#define ROCFFT_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define ROCFFT_DEVICE __forceinline__ __device__
#else
#define ROCFFT_DEVICE inline
#endif

#if defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) \
    || defined(__gfx90a__) // test device
#define USE_GFX_BUFFER_INTRINSIC
#define BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx1030__) // special device
#define USE_GFX_BUFFER_INTRINSIC
#define BUFFER_RESOURCE_3RD_DWORD 0x31014000
#else // not support
#define BUFFER_RESOURCE_3RD_DWORD -1
#endif

/// Controls AMD gfx arch cache operations
struct CacheOperation
{
    enum Kind
    {
        /// Cache at all levels - accessed again
        Always,
        /// Cache at global level; glc = 1
        Global,
        /// Streaming - likely to be accessed once; slc = 1
        Streaming,
        /// Indicates the line will not be used again, glc = 1; slc = 1
        LastUse
    };
};

using float16_t = _Float16;
using float32_t = float;

template <typename T, int N>
struct NativeVector
{
    using type = T __attribute__((ext_vector_type(N)));
};

// template <int N>
// struct NativeVector<cutlass::half_t, N>
// {
//   using type = typename NativeVector<float16_t, N>::type;
// };

// template <int N>
// struct NativeVector<cutlass::bfloat16_t, N>
// {
//   using type = typename NativeVector<float16_t, N>::type;
// };

using float32x2_t = NativeVector<float, 2>::type;
using float32x4_t = NativeVector<float, 4>::type;

using int32x4_t = NativeVector<int, 4>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct alignas(16) BufferResource
{
    union Desc
    {
        int32x4_t d128;
        void*     d64[2];
        uint32_t  d32[4];
    };

    ROCFFT_DEVICE
    BufferResource(void const* base_addr, uint32_t num_records = (0xFFFFFFFF - 1))
    {
        // Reference:
        //   For CDNA: see section 9.1.8 in the AMD resources
        //   https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf
        //   For RDNA: see section 8.1.8 in the AMD resources
        //   https://developer.amd.com/wp-content/resources/RDNA2_Shader_ISA_November2020.pdf
        //   The d32[3] field represents the 0x[127] ~ [96]

        // 64-bit base address
        desc_.d64[0] = const_cast<void*>(base_addr);
        // 32-bit number of records in bytes which is used to guard against out-of-range access
        desc_.d32[2] = num_records;
        // 32-bit buffer resource descriptor
        desc_.d32[3] = BUFFER_RESOURCE_3RD_DWORD;
    }

    ROCFFT_DEVICE
    operator int32x4_t()
    {
        // return desc_.d128; // NOTE HIP: Crashes compiler; see below

        /// This hack is to enforce scalarization of the variable "base_addr", where in some
        /// circumstances it becomes vectorized and then in turn causes illegal lowering to GCN ISA
        /// since compiler effectively tries to stuff VGPRs in slots where it only accepts SGPRs
        Desc ret;
        ret.d32[0] = __builtin_amdgcn_readfirstlane(desc_.d32[0]);
        ret.d32[1] = __builtin_amdgcn_readfirstlane(desc_.d32[1]);
        ret.d64[1] = desc_.d64[1];
        return ret.d128;
        ///
    }

    Desc desc_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// Load
///

// 1 byte
__device__ char
    llvm_amdgcn_raw_buffer_load_i8(int32x4_t buffer_resource,
                                   uint32_t  voffset,
                                   uint32_t  soffset,
                                   int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.load.i8");

// 2 bytes
__device__ float16_t
    llvm_amdgcn_raw_buffer_load_f16(int32x4_t buffer_resource,
                                    uint32_t  voffset,
                                    uint32_t  soffset,
                                    int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.load.f16");

// 4 bytes
__device__ float32_t
    llvm_amdgcn_raw_buffer_load_f32(int32x4_t buffer_resource,
                                    uint32_t  voffset,
                                    uint32_t  soffset,
                                    int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.load.f32");

// 8 bytes
__device__ float32x2_t
    llvm_amdgcn_raw_buffer_load_f32x2(int32x4_t buffer_resource,
                                      uint32_t  voffset,
                                      uint32_t  soffset,
                                      int32_t cache_op) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

// 16 bytes
__device__ float32x4_t
    llvm_amdgcn_raw_buffer_load_f32x4(int32x4_t buffer_resource,
                                      uint32_t  voffset,
                                      uint32_t  soffset,
                                      int32_t cache_op) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

///
/// Store
///

// 1 byte
__device__ void
    llvm_amdgcn_raw_buffer_store_i8(char      data,
                                    int32x4_t buffer_resource,
                                    uint32_t  voffset,
                                    uint32_t  soffset,
                                    int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.store.i8");

// 2 bytes
__device__ void
    llvm_amdgcn_raw_buffer_store_f16(float16_t data,
                                     int32x4_t buffer_resource,
                                     uint32_t  voffset,
                                     uint32_t  soffset,
                                     int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.store.f16");

// 4 bytes
__device__ void
    llvm_amdgcn_raw_buffer_store_f32(float32_t data,
                                     int32x4_t buffer_resource,
                                     uint32_t  voffset,
                                     uint32_t  soffset,
                                     int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.store.f32");

// 8 bytes
__device__ void llvm_amdgcn_raw_buffer_store_f32x2(
    float32x2_t data,
    int32x4_t   buffer_resource,
    uint32_t    voffset,
    uint32_t    soffset,
    int32_t     cache_op) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

// 16 bytes
__device__ void llvm_amdgcn_raw_buffer_store_f32x4(
    float32x4_t data,
    int32x4_t   buffer_resource,
    uint32_t    voffset,
    uint32_t    soffset,
    int32_t     cache_op) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The bytes of loading
    int LoadBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct buffer_load;

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 1, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset  = pred_guard ? voffset : -1;
        char ret = llvm_amdgcn_raw_buffer_load_i8(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset  = pred_guard ? voffset : -1;
        char ret = llvm_amdgcn_raw_buffer_load_i8(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 2, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float16_t ret = llvm_amdgcn_raw_buffer_load_f16(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float16_t ret = llvm_amdgcn_raw_buffer_load_f16(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 4, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float32_t ret = llvm_amdgcn_raw_buffer_load_f32(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float32_t ret = llvm_amdgcn_raw_buffer_load_f32(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 8, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x2_t ret = llvm_amdgcn_raw_buffer_load_f32x2(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x2_t ret = llvm_amdgcn_raw_buffer_load_f32x2(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 16, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x4_t ret = llvm_amdgcn_raw_buffer_load_f32x4(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x4_t ret = llvm_amdgcn_raw_buffer_load_f32x4(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The width of loading
    int NumElements,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct buffer_store;

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 1, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset   = pred_guard ? voffset : -1;
        char data = *reinterpret_cast<char const*>(&D);
        llvm_amdgcn_raw_buffer_store_i8(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 2, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset        = pred_guard ? voffset : -1;
        float16_t data = *reinterpret_cast<float16_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f16(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 4, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset        = pred_guard ? voffset : -1;
        float32_t data = *reinterpret_cast<float32_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f32(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 8, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset          = pred_guard ? voffset : -1;
        float32x2_t data = *reinterpret_cast<float32x2_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f32x2(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 16, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset          = pred_guard ? voffset : -1;
        float32x4_t data = *reinterpret_cast<float32x4_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f32x4(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(__clang__) && defined(__HIP__)

#endif // INTRINSIC_MEM_ACCESS_H

// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_DEVICE_CALLBACK_H
#define ROCFFT_DEVICE_CALLBACK_H

// user-provided data saying what callbacks to run
struct UserCallbacks
{
    void*  load_cb_fn        = nullptr;
    void*  load_cb_data      = nullptr;
    size_t load_cb_lds_bytes = 0;

    void*  store_cb_fn        = nullptr;
    void*  store_cb_data      = nullptr;
    size_t store_cb_lds_bytes = 0;
};

// default callback implementations that just do simple load/store
template <typename T>
__device__ T load_cb_default(T* data, size_t offset, void* cbdata, void* sharedMem)
{
    return data[offset];
}

template <typename T>
__device__ void store_cb_default(T* data, size_t offset, T element, void* cbdata, void* sharedMem)
{
    data[offset] = element;
}

// callback function types
template <typename T>
struct callback_type;

template <>
struct callback_type<float>
{
    typedef float (*load)(float* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(float* data, size_t offset, float element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_float  = load_cb_default<float>;
static __device__ auto store_cb_default_float = store_cb_default<float>;

template <>
struct callback_type<float2>
{
    typedef float2 (*load)(float2* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(
        float2* data, size_t offset, float2 element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_float2  = load_cb_default<float2>;
static __device__ auto store_cb_default_float2 = store_cb_default<float2>;

template <>
struct callback_type<double>
{
    typedef double (*load)(double* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(
        double* data, size_t offset, double element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_double  = load_cb_default<double>;
static __device__ auto store_cb_default_double = store_cb_default<double>;

template <>
struct callback_type<double2>
{
    typedef double2 (*load)(double2* data, size_t offset, void* cbdata, void* sharedMem);
    typedef void (*store)(
        double2* data, size_t offset, double2 element, void* cbdata, void* sharedMem);
};

static __device__ auto load_cb_default_double2  = load_cb_default<double2>;
static __device__ auto store_cb_default_double2 = store_cb_default<double2>;

// intrinsic
template <typename T>
__device__ void intrinsic_load_to_dest(
    T& target, const T* data, unsigned int voffset, unsigned int soffset, bool rw)
{
#ifdef USE_GFX_BUFFER_INTRINSIC
    buffer_load<T, sizeof(T)>(target,
                              reinterpret_cast<void*>(const_cast<T*>(data)),
                              (uint32_t)(voffset * sizeof(T)),
                              (uint32_t)(soffset * sizeof(T)),
                              rw);
#else
    target = data[soffset + voffset];
#endif
}

template <typename T>
__device__ T intrinsic_load(const T* data, unsigned int voffset, unsigned int soffset, bool rw)
{
#ifdef USE_GFX_BUFFER_INTRINSIC
    return buffer_load<T, sizeof(T)>().load(reinterpret_cast<void*>(const_cast<T*>(data)),
                                            (uint32_t)(voffset * sizeof(T)),
                                            (uint32_t)(soffset * sizeof(T)),
                                            rw);
#else
    return data[soffset + voffset];
#endif
}

template <typename T>
__device__ void
    store_intrinsic(T* data, unsigned int voffset, unsigned int soffset, T element, bool rw)
{
#ifdef USE_GFX_BUFFER_INTRINSIC
    buffer_store<T, sizeof(T)>(element,
                               reinterpret_cast<void*>(const_cast<T*>(data)),
                               (uint32_t)(voffset * sizeof(T)),
                               (uint32_t)(soffset * sizeof(T)),
                               rw);
#else
    if(rw)
        data[soffset + voffset] = element;
#endif
}

enum struct CallbackType
{
    // don't run user callbacks
    NONE,
    // run user load/store callbacks
    USER_LOAD_STORE,
};

// helpers to cast void* to the correct function pointer type
template <typename T, CallbackType cbtype>
static __device__ typename callback_type<T>::load get_load_cb(void* ptr)
{
#ifdef ROCFFT_CALLBACKS_ENABLED
    if(cbtype == CallbackType::USER_LOAD_STORE)
        return reinterpret_cast<typename callback_type<T>::load>(ptr);
#endif
    return load_cb_default<T>;
}

template <typename T, CallbackType cbtype>
static __device__ typename callback_type<T>::store get_store_cb(void* ptr)
{
#ifdef ROCFFT_CALLBACKS_ENABLED
    if(cbtype == CallbackType::USER_LOAD_STORE)
        return reinterpret_cast<typename callback_type<T>::store>(ptr);
#endif
    return store_cb_default<T>;
}

#endif

/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef BUTTERFLY_CONSTANT_H
#define BUTTERFLY_CONSTANT_H

// butterfly radix-3 constants
#define C3QA static_cast<real_type_t<T>>(0.50000000000000000000000000000000)
#define C3QB static_cast<real_type_t<T>>(0.86602540378443864676372317075294)

// butterfly radix-5 constants
#define C5QA static_cast<real_type_t<T>>(0.30901699437494742410229341718282)
#define C5QB static_cast<real_type_t<T>>(0.95105651629515357211643933337938)
#define C5QC static_cast<real_type_t<T>>(0.50000000000000000000000000000000)
#define C5QD static_cast<real_type_t<T>>(0.58778525229247312916870595463907)
#define C5QE static_cast<real_type_t<T>>(0.80901699437494742410229341718282)

// butterfly radix-7 constants
#define C7Q1 static_cast<real_type_t<T>>(-1.16666666666666651863693004997913)
#define C7Q2 static_cast<real_type_t<T>>(0.79015646852540022404554065360571)
#define C7Q3 static_cast<real_type_t<T>>(0.05585426728964774240049351305970)
#define C7Q4 static_cast<real_type_t<T>>(0.73430220123575240531721419756650)
#define C7Q5 static_cast<real_type_t<T>>(0.44095855184409837868031445395900)
#define C7Q6 static_cast<real_type_t<T>>(0.34087293062393136944265847887436)
#define C7Q7 static_cast<real_type_t<T>>(-0.53396936033772524066165487965918)
#define C7Q8 static_cast<real_type_t<T>>(0.87484229096165666561546458979137)

// butterfly radix-8 constants
#define C8Q static_cast<real_type_t<T>>(0.70710678118654752440084436210485)

// butterfly radix-9 constants
#define C9QA static_cast<real_type_t<T>>(0.766044443118978)
#define C9QB static_cast<real_type_t<T>>(0.6427876096865393)
#define C9QC static_cast<real_type_t<T>>(0.1736481776669304)
#define C9QD static_cast<real_type_t<T>>(0.984807753012208)
#define C9QE static_cast<real_type_t<T>>(0.5000000000000000)
#define C9QF static_cast<real_type_t<T>>(0.8660254037844387)
#define C9QG static_cast<real_type_t<T>>(0.9396926207859083)
#define C9QH static_cast<real_type_t<T>>(0.3420201433256689)

//
// For radix-11 and radix-13 the butterfly constants correspond to
// the roots of unity for the radix; and are named according to:
//
//   "Q" + radix + "i" + i + "j" + j + "R"/"I"
//
// where i and j are the row/col indicies of the DFT matrix A
// corresponding to the radix and R/I is the real/imaginary part.
// More specifically:
//
//  A[i,j] = exp(-2 pi I i j / radix)
//
// and hence, for example
//
//  Q11i2j5R = Re( exp(-2 pi I 2 * 5 / 11) )
//

// butterfly radix-11 constants
#define Q11i1j1R static_cast<real_type_t<T>>((0.8412535328311811688618))
#define Q11i1j1I static_cast<real_type_t<T>>((-0.5406408174555975821076))
#define Q11i1j2R static_cast<real_type_t<T>>((0.4154150130018864255293))
#define Q11i1j2I static_cast<real_type_t<T>>((-0.9096319953545183714117))
#define Q11i1j3R static_cast<real_type_t<T>>((-0.1423148382732851404438))
#define Q11i1j3I static_cast<real_type_t<T>>((-0.9898214418809327323761))
#define Q11i1j4R static_cast<real_type_t<T>>((-0.6548607339452850640569))
#define Q11i1j4I static_cast<real_type_t<T>>((-0.7557495743542582837740))
#define Q11i1j5R static_cast<real_type_t<T>>((-0.9594929736144973898904))
#define Q11i1j5I static_cast<real_type_t<T>>((-0.2817325568414296977114))
#define Q11i2j1R static_cast<real_type_t<T>>((0.4154150130018864255293))
#define Q11i2j1I static_cast<real_type_t<T>>((-0.9096319953545183714117))
#define Q11i2j2R static_cast<real_type_t<T>>((-0.6548607339452850640569))
#define Q11i2j2I static_cast<real_type_t<T>>((-0.7557495743542582837740))
#define Q11i2j3R static_cast<real_type_t<T>>((-0.9594929736144973898904))
#define Q11i2j3I static_cast<real_type_t<T>>((0.2817325568414296977114))
#define Q11i2j4R static_cast<real_type_t<T>>((-0.1423148382732851404438))
#define Q11i2j4I static_cast<real_type_t<T>>((0.9898214418809327323761))
#define Q11i2j5R static_cast<real_type_t<T>>((0.8412535328311811688618))
#define Q11i2j5I static_cast<real_type_t<T>>((0.5406408174555975821076))
#define Q11i3j1R static_cast<real_type_t<T>>((-0.1423148382732851404438))
#define Q11i3j1I static_cast<real_type_t<T>>((-0.9898214418809327323761))
#define Q11i3j2R static_cast<real_type_t<T>>((-0.9594929736144973898904))
#define Q11i3j2I static_cast<real_type_t<T>>((0.2817325568414296977114))
#define Q11i3j3R static_cast<real_type_t<T>>((0.4154150130018864255293))
#define Q11i3j3I static_cast<real_type_t<T>>((0.9096319953545183714117))
#define Q11i3j4R static_cast<real_type_t<T>>((0.8412535328311811688618))
#define Q11i3j4I static_cast<real_type_t<T>>((-0.5406408174555975821076))
#define Q11i3j5R static_cast<real_type_t<T>>((-0.6548607339452850640569))
#define Q11i3j5I static_cast<real_type_t<T>>((-0.7557495743542582837740))
#define Q11i4j1R static_cast<real_type_t<T>>((-0.6548607339452850640569))
#define Q11i4j1I static_cast<real_type_t<T>>((-0.7557495743542582837740))
#define Q11i4j2R static_cast<real_type_t<T>>((-0.1423148382732851404438))
#define Q11i4j2I static_cast<real_type_t<T>>((0.9898214418809327323761))
#define Q11i4j3R static_cast<real_type_t<T>>((0.8412535328311811688618))
#define Q11i4j3I static_cast<real_type_t<T>>((-0.5406408174555975821076))
#define Q11i4j4R static_cast<real_type_t<T>>((-0.9594929736144973898904))
#define Q11i4j4I static_cast<real_type_t<T>>((-0.2817325568414296977114))
#define Q11i4j5R static_cast<real_type_t<T>>((0.4154150130018864255293))
#define Q11i4j5I static_cast<real_type_t<T>>((0.9096319953545183714117))
#define Q11i5j1R static_cast<real_type_t<T>>((-0.9594929736144973898904))
#define Q11i5j1I static_cast<real_type_t<T>>((-0.2817325568414296977114))
#define Q11i5j2R static_cast<real_type_t<T>>((0.8412535328311811688618))
#define Q11i5j2I static_cast<real_type_t<T>>((0.5406408174555975821076))
#define Q11i5j3R static_cast<real_type_t<T>>((-0.6548607339452850640569))
#define Q11i5j3I static_cast<real_type_t<T>>((-0.7557495743542582837740))
#define Q11i5j4R static_cast<real_type_t<T>>((0.4154150130018864255293))
#define Q11i5j4I static_cast<real_type_t<T>>((0.9096319953545183714117))
#define Q11i5j5R static_cast<real_type_t<T>>((-0.1423148382732851404438))
#define Q11i5j5I static_cast<real_type_t<T>>((-0.9898214418809327323761))

// butterfly radix-13 constants
#define Q13i1j1R static_cast<real_type_t<T>>((0.8854560256532098959004))
#define Q13i1j1I static_cast<real_type_t<T>>((-0.4647231720437685456560))
#define Q13i1j2R static_cast<real_type_t<T>>((0.5680647467311558025118))
#define Q13i1j2I static_cast<real_type_t<T>>((-0.8229838658936563945796))
#define Q13i1j3R static_cast<real_type_t<T>>((0.1205366802553230533491))
#define Q13i1j3I static_cast<real_type_t<T>>((-0.9927088740980539928007))
#define Q13i1j4R static_cast<real_type_t<T>>((-0.3546048870425356259696))
#define Q13i1j4I static_cast<real_type_t<T>>((-0.9350162426854148234398))
#define Q13i1j5R static_cast<real_type_t<T>>((-0.7485107481711010986346))
#define Q13i1j5I static_cast<real_type_t<T>>((-0.6631226582407952023768))
#define Q13i1j6R static_cast<real_type_t<T>>((-0.9709418174260520271570))
#define Q13i1j6I static_cast<real_type_t<T>>((-0.2393156642875577671488))
#define Q13i2j1R static_cast<real_type_t<T>>((0.5680647467311558025118))
#define Q13i2j1I static_cast<real_type_t<T>>((-0.8229838658936563945796))
#define Q13i2j2R static_cast<real_type_t<T>>((-0.3546048870425356259696))
#define Q13i2j2I static_cast<real_type_t<T>>((-0.9350162426854148234398))
#define Q13i2j3R static_cast<real_type_t<T>>((-0.9709418174260520271570))
#define Q13i2j3I static_cast<real_type_t<T>>((-0.2393156642875577671488))
#define Q13i2j4R static_cast<real_type_t<T>>((-0.7485107481711010986346))
#define Q13i2j4I static_cast<real_type_t<T>>((0.6631226582407952023768))
#define Q13i2j5R static_cast<real_type_t<T>>((0.1205366802553230533491))
#define Q13i2j5I static_cast<real_type_t<T>>((0.9927088740980539928007))
#define Q13i2j6R static_cast<real_type_t<T>>((0.8854560256532098959004))
#define Q13i2j6I static_cast<real_type_t<T>>((0.4647231720437685456560))
#define Q13i3j1R static_cast<real_type_t<T>>((0.1205366802553230533491))
#define Q13i3j1I static_cast<real_type_t<T>>((-0.9927088740980539928007))
#define Q13i3j2R static_cast<real_type_t<T>>((-0.9709418174260520271570))
#define Q13i3j2I static_cast<real_type_t<T>>((-0.2393156642875577671488))
#define Q13i3j3R static_cast<real_type_t<T>>((-0.3546048870425356259696))
#define Q13i3j3I static_cast<real_type_t<T>>((0.9350162426854148234398))
#define Q13i3j4R static_cast<real_type_t<T>>((0.8854560256532098959004))
#define Q13i3j4I static_cast<real_type_t<T>>((0.4647231720437685456560))
#define Q13i3j5R static_cast<real_type_t<T>>((0.5680647467311558025118))
#define Q13i3j5I static_cast<real_type_t<T>>((-0.8229838658936563945796))
#define Q13i3j6R static_cast<real_type_t<T>>((-0.7485107481711010986346))
#define Q13i3j6I static_cast<real_type_t<T>>((-0.6631226582407952023768))
#define Q13i4j1R static_cast<real_type_t<T>>((-0.3546048870425356259696))
#define Q13i4j1I static_cast<real_type_t<T>>((-0.9350162426854148234398))
#define Q13i4j2R static_cast<real_type_t<T>>((-0.7485107481711010986346))
#define Q13i4j2I static_cast<real_type_t<T>>((0.6631226582407952023768))
#define Q13i4j3R static_cast<real_type_t<T>>((0.8854560256532098959004))
#define Q13i4j3I static_cast<real_type_t<T>>((0.4647231720437685456560))
#define Q13i4j4R static_cast<real_type_t<T>>((0.1205366802553230533491))
#define Q13i4j4I static_cast<real_type_t<T>>((-0.9927088740980539928007))
#define Q13i4j5R static_cast<real_type_t<T>>((-0.9709418174260520271570))
#define Q13i4j5I static_cast<real_type_t<T>>((0.2393156642875577671488))
#define Q13i4j6R static_cast<real_type_t<T>>((0.5680647467311558025118))
#define Q13i4j6I static_cast<real_type_t<T>>((0.8229838658936563945796))
#define Q13i5j1R static_cast<real_type_t<T>>((-0.7485107481711010986346))
#define Q13i5j1I static_cast<real_type_t<T>>((-0.6631226582407952023768))
#define Q13i5j2R static_cast<real_type_t<T>>((0.1205366802553230533491))
#define Q13i5j2I static_cast<real_type_t<T>>((0.9927088740980539928007))
#define Q13i5j3R static_cast<real_type_t<T>>((0.5680647467311558025118))
#define Q13i5j3I static_cast<real_type_t<T>>((-0.8229838658936563945796))
#define Q13i5j4R static_cast<real_type_t<T>>((-0.9709418174260520271570))
#define Q13i5j4I static_cast<real_type_t<T>>((0.2393156642875577671488))
#define Q13i5j5R static_cast<real_type_t<T>>((0.8854560256532098959004))
#define Q13i5j5I static_cast<real_type_t<T>>((0.4647231720437685456560))
#define Q13i5j6R static_cast<real_type_t<T>>((-0.3546048870425356259696))
#define Q13i5j6I static_cast<real_type_t<T>>((-0.9350162426854148234398))
#define Q13i6j1R static_cast<real_type_t<T>>((-0.9709418174260520271570))
#define Q13i6j1I static_cast<real_type_t<T>>((-0.2393156642875577671488))
#define Q13i6j2R static_cast<real_type_t<T>>((0.8854560256532098959004))
#define Q13i6j2I static_cast<real_type_t<T>>((0.4647231720437685456560))
#define Q13i6j3R static_cast<real_type_t<T>>((-0.7485107481711010986346))
#define Q13i6j3I static_cast<real_type_t<T>>((-0.6631226582407952023768))
#define Q13i6j4R static_cast<real_type_t<T>>((0.5680647467311558025118))
#define Q13i6j4I static_cast<real_type_t<T>>((0.8229838658936563945796))
#define Q13i6j5R static_cast<real_type_t<T>>((-0.3546048870425356259696))
#define Q13i6j5I static_cast<real_type_t<T>>((-0.9350162426854148234398))
#define Q13i6j6R static_cast<real_type_t<T>>((0.1205366802553230533491))
#define Q13i6j6I static_cast<real_type_t<T>>((0.9927088740980539928007))

#define Q17i1j1R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i1j1I static_cast<real_type_t<T>>((-0.3612416661871529487447))
#define Q17i1j2R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i1j2I static_cast<real_type_t<T>>((-0.6736956436465572117127))
#define Q17i1j3R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i1j3I static_cast<real_type_t<T>>((-0.8951632913550623220670))
#define Q17i1j4R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i1j4I static_cast<real_type_t<T>>((-0.9957341762950345218712))
#define Q17i1j5R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i1j5I static_cast<real_type_t<T>>((-0.9618256431728190704088))
#define Q17i1j6R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i1j6I static_cast<real_type_t<T>>((-0.7980172272802395033328))
#define Q17i1j7R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i1j7I static_cast<real_type_t<T>>((-0.5264321628773558002446))
#define Q17i1j8R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i1j8I static_cast<real_type_t<T>>((-0.1837495178165703315744))
#define Q17i2j1R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i2j1I static_cast<real_type_t<T>>((-0.6736956436465572117127))
#define Q17i2j2R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i2j2I static_cast<real_type_t<T>>((-0.9957341762950345218712))
#define Q17i2j3R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i2j3I static_cast<real_type_t<T>>((-0.7980172272802395033328))
#define Q17i2j4R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i2j4I static_cast<real_type_t<T>>((-0.1837495178165703315744))
#define Q17i2j5R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i2j5I static_cast<real_type_t<T>>((0.5264321628773558002446))
#define Q17i2j6R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i2j6I static_cast<real_type_t<T>>((0.9618256431728190704088))
#define Q17i2j7R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i2j7I static_cast<real_type_t<T>>((0.8951632913550623220670))
#define Q17i2j8R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i2j8I static_cast<real_type_t<T>>((0.3612416661871529487447))
#define Q17i3j1R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i3j1I static_cast<real_type_t<T>>((-0.8951632913550623220670))
#define Q17i3j2R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i3j2I static_cast<real_type_t<T>>((-0.7980172272802395033328))
#define Q17i3j3R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i3j3I static_cast<real_type_t<T>>((0.1837495178165703315744))
#define Q17i3j4R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i3j4I static_cast<real_type_t<T>>((0.9618256431728190704088))
#define Q17i3j5R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i3j5I static_cast<real_type_t<T>>((0.6736956436465572117127))
#define Q17i3j6R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i3j6I static_cast<real_type_t<T>>((-0.3612416661871529487447))
#define Q17i3j7R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i3j7I static_cast<real_type_t<T>>((-0.9957341762950345218712))
#define Q17i3j8R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i3j8I static_cast<real_type_t<T>>((-0.5264321628773558002446))
#define Q17i4j1R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i4j1I static_cast<real_type_t<T>>((-0.9957341762950345218712))
#define Q17i4j2R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i4j2I static_cast<real_type_t<T>>((-0.1837495178165703315744))
#define Q17i4j3R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i4j3I static_cast<real_type_t<T>>((0.9618256431728190704088))
#define Q17i4j4R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i4j4I static_cast<real_type_t<T>>((0.3612416661871529487447))
#define Q17i4j5R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i4j5I static_cast<real_type_t<T>>((-0.8951632913550623220670))
#define Q17i4j6R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i4j6I static_cast<real_type_t<T>>((-0.5264321628773558002446))
#define Q17i4j7R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i4j7I static_cast<real_type_t<T>>((0.7980172272802395033328))
#define Q17i4j8R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i4j8I static_cast<real_type_t<T>>((0.6736956436465572117127))
#define Q17i5j1R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i5j1I static_cast<real_type_t<T>>((-0.9618256431728190704088))
#define Q17i5j2R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i5j2I static_cast<real_type_t<T>>((0.5264321628773558002446))
#define Q17i5j3R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i5j3I static_cast<real_type_t<T>>((0.6736956436465572117127))
#define Q17i5j4R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i5j4I static_cast<real_type_t<T>>((-0.8951632913550623220670))
#define Q17i5j5R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i5j5I static_cast<real_type_t<T>>((-0.1837495178165703315744))
#define Q17i5j6R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i5j6I static_cast<real_type_t<T>>((0.9957341762950345218712))
#define Q17i5j7R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i5j7I static_cast<real_type_t<T>>((-0.3612416661871529487447))
#define Q17i5j8R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i5j8I static_cast<real_type_t<T>>((-0.7980172272802395033328))
#define Q17i6j1R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i6j1I static_cast<real_type_t<T>>((-0.7980172272802395033328))
#define Q17i6j2R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i6j2I static_cast<real_type_t<T>>((0.9618256431728190704088))
#define Q17i6j3R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i6j3I static_cast<real_type_t<T>>((-0.3612416661871529487447))
#define Q17i6j4R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i6j4I static_cast<real_type_t<T>>((-0.5264321628773558002446))
#define Q17i6j5R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i6j5I static_cast<real_type_t<T>>((0.9957341762950345218712))
#define Q17i6j6R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i6j6I static_cast<real_type_t<T>>((-0.6736956436465572117127))
#define Q17i6j7R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i6j7I static_cast<real_type_t<T>>((-0.1837495178165703315744))
#define Q17i6j8R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i6j8I static_cast<real_type_t<T>>((0.8951632913550623220670))
#define Q17i7j1R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i7j1I static_cast<real_type_t<T>>((-0.5264321628773558002446))
#define Q17i7j2R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i7j2I static_cast<real_type_t<T>>((0.8951632913550623220670))
#define Q17i7j3R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i7j3I static_cast<real_type_t<T>>((-0.9957341762950345218712))
#define Q17i7j4R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i7j4I static_cast<real_type_t<T>>((0.7980172272802395033328))
#define Q17i7j5R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i7j5I static_cast<real_type_t<T>>((-0.3612416661871529487447))
#define Q17i7j6R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i7j6I static_cast<real_type_t<T>>((-0.1837495178165703315744))
#define Q17i7j7R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i7j7I static_cast<real_type_t<T>>((0.6736956436465572117127))
#define Q17i7j8R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i7j8I static_cast<real_type_t<T>>((-0.9618256431728190704088))
#define Q17i8j1R static_cast<real_type_t<T>>((-0.9829730996839017782819))
#define Q17i8j1I static_cast<real_type_t<T>>((-0.1837495178165703315744))
#define Q17i8j2R static_cast<real_type_t<T>>((0.9324722294043558045731))
#define Q17i8j2I static_cast<real_type_t<T>>((0.3612416661871529487447))
#define Q17i8j3R static_cast<real_type_t<T>>((-0.8502171357296141521341))
#define Q17i8j3I static_cast<real_type_t<T>>((-0.5264321628773558002446))
#define Q17i8j4R static_cast<real_type_t<T>>((0.7390089172206591159245))
#define Q17i8j4I static_cast<real_type_t<T>>((0.6736956436465572117127))
#define Q17i8j5R static_cast<real_type_t<T>>((-0.6026346363792563891786))
#define Q17i8j5I static_cast<real_type_t<T>>((-0.7980172272802395033328))
#define Q17i8j6R static_cast<real_type_t<T>>((0.4457383557765382673965))
#define Q17i8j6I static_cast<real_type_t<T>>((0.8951632913550623220670))
#define Q17i8j7R static_cast<real_type_t<T>>((-0.2736629900720828635391))
#define Q17i8j7I static_cast<real_type_t<T>>((-0.9618256431728190704088))
#define Q17i8j8R static_cast<real_type_t<T>>((0.09226835946330199523965))
#define Q17i8j8I static_cast<real_type_t<T>>((0.9957341762950345218712))

// butterfly radix-11 constants
#define b11_0 static_cast<real_type_t<T>>(0.9898214418809327)
#define b11_1 static_cast<real_type_t<T>>(0.9594929736144973)
#define b11_2 static_cast<real_type_t<T>>(0.9189859472289947)
#define b11_3 static_cast<real_type_t<T>>(0.8767688310025893)
#define b11_4 static_cast<real_type_t<T>>(0.8308300260037728)
#define b11_5 static_cast<real_type_t<T>>(0.7784344533346518)
#define b11_6 static_cast<real_type_t<T>>(0.7153703234534297)
#define b11_7 static_cast<real_type_t<T>>(0.6343562706824244)
#define b11_8 static_cast<real_type_t<T>>(0.3425847256816375)
#define b11_9 static_cast<real_type_t<T>>(0.5211085581132027)

// butterfly radix-13 constants
#define b13_0 static_cast<real_type_t<T>>(0.9682872443619840)
#define b13_1 static_cast<real_type_t<T>>(0.9578059925946651)
#define b13_2 static_cast<real_type_t<T>>(0.8755023024091479)
#define b13_3 static_cast<real_type_t<T>>(0.8660254037844386)
#define b13_4 static_cast<real_type_t<T>>(0.8595425350987748)
#define b13_5 static_cast<real_type_t<T>>(0.8534800018598239)
#define b13_6 static_cast<real_type_t<T>>(0.7693388175729806)
#define b13_7 static_cast<real_type_t<T>>(0.6865583707817543)
#define b13_8 static_cast<real_type_t<T>>(0.6122646503767565)
#define b13_9 static_cast<real_type_t<T>>(0.6004772719326652)
#define b13_10 static_cast<real_type_t<T>>(0.5817047785105157)
#define b13_11 static_cast<real_type_t<T>>(0.5751407294740031)
#define b13_12 static_cast<real_type_t<T>>(0.5220263851612750)
#define b13_13 static_cast<real_type_t<T>>(0.5200285718888646)
#define b13_14 static_cast<real_type_t<T>>(0.5165207806234897)
#define b13_15 static_cast<real_type_t<T>>(0.5149187780863157)
#define b13_16 static_cast<real_type_t<T>>(0.5035370328637666)
#define b13_17 static_cast<real_type_t<T>>(0.5000000000000000)
#define b13_18 static_cast<real_type_t<T>>(0.3027756377319946)
#define b13_19 static_cast<real_type_t<T>>(0.3014792600477098)
#define b13_20 static_cast<real_type_t<T>>(0.3004626062886657)
#define b13_21 static_cast<real_type_t<T>>(0.2517685164318833)
#define b13_22 static_cast<real_type_t<T>>(0.2261094450357824)
#define b13_23 static_cast<real_type_t<T>>(0.0833333333333333)
#define b13_24 static_cast<real_type_t<T>>(0.0386329546443481)

// butterfly radix-16 constants
#define C16A static_cast<real_type_t<T>>(0.923879532511286738)
#define C16B static_cast<real_type_t<T>>(0.382683432365089837)

#endif //  BUTTERFLY_CONSTANT_H

/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef ROCFFT_BUTTERFLY_TEMPLATE_H
#define ROCFFT_BUTTERFLY_TEMPLATE_H

template <typename T, size_t Base, size_t Steps>
__device__ T TW_NSteps(const T* const twiddles, size_t u)
{
    size_t j      = u & ((1 << Base) - 1); // get the lowest Base bits
    T      result = twiddles[j];
    u >>= Base; // discard the lowest Base bits
    int i = 0;
    // static compiled, currently, steps can only be 2 or 3
    if(Steps >= 2)
    {
        i += 1;
        j      = u & ((1 << Base) - 1);
        result = lib_make_vector2<T>((result.x * twiddles[(1 << Base) * i + j].x
                                      - result.y * twiddles[(1 << Base) * i + j].y),
                                     (result.y * twiddles[(1 << Base) * i + j].x
                                      + result.x * twiddles[(1 << Base) * i + j].y));
    }
    // static compiled
    if(Steps >= 3)
    {
        u >>= Base; // discard the lowest Base bits

        i += 1;
        j      = u & ((1 << Base) - 1);
        result = lib_make_vector2<T>((result.x * twiddles[(1 << Base) * i + j].x
                                      - result.y * twiddles[(1 << Base) * i + j].y),
                                     (result.y * twiddles[(1 << Base) * i + j].x
                                      + result.x * twiddles[(1 << Base) * i + j].y));
    }
    // we probably don't have 4-steps for large-twiddle
    // if(Steps >= 4){...}

    return result;
}

template <typename T>
__device__ T TW3step(const T* const twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];

    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
                                 (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));

    u >>= 8;
    j      = u & 255;
    result = lib_make_vector2<T>((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
                                 (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    return result;
}

template <typename T>
__device__ void FwdRad2B1(T* R0, T* R1)
{

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

template <typename T>
__device__ void InvRad2B1(T* R0, T* R1)
{

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

template <typename T>
__device__ void FwdRad3B1(T* R0, T* R1, T* R2)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2;

    TR0 = (*R0).x + (*R1).x + (*R2).x;
    TR1 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) + C3QB * ((*R1).y - (*R2).y);
    TR2 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) - C3QB * ((*R1).y - (*R2).y);

    TI0 = (*R0).y + (*R1).y + (*R2).y;
    TI1 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) - C3QB * ((*R1).x - (*R2).x);
    TI2 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) + C3QB * ((*R1).x - (*R2).x);

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
}

template <typename T>
__device__ void InvRad3B1(T* R0, T* R1, T* R2)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2;

    TR0 = (*R0).x + (*R1).x + (*R2).x;
    TR1 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) - C3QB * ((*R1).y - (*R2).y);
    TR2 = ((*R0).x - C3QA * ((*R1).x + (*R2).x)) + C3QB * ((*R1).y - (*R2).y);

    TI0 = (*R0).y + (*R1).y + (*R2).y;
    TI1 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) + C3QB * ((*R1).x - (*R2).x);
    TI2 = ((*R0).y - C3QA * ((*R1).y + (*R2).y)) - C3QB * ((*R1).x - (*R2).x);

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
}

template <typename T>
__device__ void FwdRad4B1(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);

    (*R3) = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

template <typename T>
__device__ void InvRad4B1(T* R0, T* R2, T* R1, T* R3)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

template <typename T>
__device__ void FwdRad5B1(T* R0, T* R1, T* R2, T* R3, T* R4)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4;

    TR0 = (*R0).x + (*R1).x + (*R2).x + (*R3).x + (*R4).x;
    TR1 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) + C5QB * ((*R1).y - (*R4).y)
          + C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) - C5QB * ((*R1).y - (*R4).y)
          - C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR2 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) - C5QB * ((*R2).y - (*R3).y)
          + C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));
    TR3 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) + C5QB * ((*R2).y - (*R3).y)
          - C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));

    TI0 = (*R0).y + (*R1).y + (*R2).y + (*R3).y + (*R4).y;
    TI1 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) - C5QB * ((*R1).x - (*R4).x)
          - C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) + C5QB * ((*R1).x - (*R4).x)
          + C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI2 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) + C5QB * ((*R2).x - (*R3).x)
          - C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));
    TI3 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) - C5QB * ((*R2).x - (*R3).x)
          + C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
    ((*R3).x) = TR3;
    ((*R3).y) = TI3;
    ((*R4).x) = TR4;
    ((*R4).y) = TI4;
}

template <typename T>
__device__ void InvRad5B1(T* R0, T* R1, T* R2, T* R3, T* R4)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4;

    TR0 = (*R0).x + (*R1).x + (*R2).x + (*R3).x + (*R4).x;
    TR1 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) - C5QB * ((*R1).y - (*R4).y)
          - C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R3).x)) + C5QB * ((*R1).y - (*R4).y)
          + C5QD * ((*R2).y - (*R3).y) + C5QA * (((*R1).x - (*R2).x) + ((*R4).x - (*R3).x));
    TR2 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) + C5QB * ((*R2).y - (*R3).y)
          - C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));
    TR3 = ((*R0).x - C5QC * ((*R1).x + (*R4).x)) - C5QB * ((*R2).y - (*R3).y)
          + C5QD * ((*R1).y - (*R4).y) + C5QA * (((*R2).x - (*R1).x) + ((*R3).x - (*R4).x));

    TI0 = (*R0).y + (*R1).y + (*R2).y + (*R3).y + (*R4).y;
    TI1 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) + C5QB * ((*R1).x - (*R4).x)
          + C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R3).y)) - C5QB * ((*R1).x - (*R4).x)
          - C5QD * ((*R2).x - (*R3).x) + C5QA * (((*R1).y - (*R2).y) + ((*R4).y - (*R3).y));
    TI2 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) - C5QB * ((*R2).x - (*R3).x)
          + C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));
    TI3 = ((*R0).y - C5QC * ((*R1).y + (*R4).y)) + C5QB * ((*R2).x - (*R3).x)
          - C5QD * ((*R1).x - (*R4).x) + C5QA * (((*R2).y - (*R1).y) + ((*R3).y - (*R4).y));

    ((*R0).x) = TR0;
    ((*R0).y) = TI0;
    ((*R1).x) = TR1;
    ((*R1).y) = TI1;
    ((*R2).x) = TR2;
    ((*R2).y) = TI2;
    ((*R3).x) = TR3;
    ((*R3).y) = TI3;
    ((*R4).x) = TR4;
    ((*R4).y) = TI4;
}

template <typename T>
__device__ void FwdRad6B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5;

    TR0 = (*R0).x + (*R2).x + (*R4).x;
    TR2 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) + C3QB * ((*R2).y - (*R4).y);
    TR4 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) - C3QB * ((*R2).y - (*R4).y);

    TI0 = (*R0).y + (*R2).y + (*R4).y;
    TI2 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) - C3QB * ((*R2).x - (*R4).x);
    TI4 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) + C3QB * ((*R2).x - (*R4).x);

    TR1 = (*R1).x + (*R3).x + (*R5).x;
    TR3 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) + C3QB * ((*R3).y - (*R5).y);
    TR5 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) - C3QB * ((*R3).y - (*R5).y);

    TI1 = (*R1).y + (*R3).y + (*R5).y;
    TI3 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) - C3QB * ((*R3).x - (*R5).x);
    TI5 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) + C3QB * ((*R3).x - (*R5).x);

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C3QA * TR3 + C3QB * TI3);
    (*R2).x = TR4 + (-C3QA * TR5 + C3QB * TI5);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (-C3QB * TR3 + C3QA * TI3);
    (*R2).y = TI4 + (-C3QB * TR5 - C3QA * TI5);

    (*R3).x = TR0 - TR1;
    (*R4).x = TR2 - (C3QA * TR3 + C3QB * TI3);
    (*R5).x = TR4 - (-C3QA * TR5 + C3QB * TI5);

    (*R3).y = TI0 - TI1;
    (*R4).y = TI2 - (-C3QB * TR3 + C3QA * TI3);
    (*R5).y = TI4 - (-C3QB * TR5 - C3QA * TI5);
}

template <typename T>
__device__ void InvRad6B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5;

    TR0 = (*R0).x + (*R2).x + (*R4).x;
    TR2 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) - C3QB * ((*R2).y - (*R4).y);
    TR4 = ((*R0).x - C3QA * ((*R2).x + (*R4).x)) + C3QB * ((*R2).y - (*R4).y);

    TI0 = (*R0).y + (*R2).y + (*R4).y;
    TI2 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) + C3QB * ((*R2).x - (*R4).x);
    TI4 = ((*R0).y - C3QA * ((*R2).y + (*R4).y)) - C3QB * ((*R2).x - (*R4).x);

    TR1 = (*R1).x + (*R3).x + (*R5).x;
    TR3 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) - C3QB * ((*R3).y - (*R5).y);
    TR5 = ((*R1).x - C3QA * ((*R3).x + (*R5).x)) + C3QB * ((*R3).y - (*R5).y);

    TI1 = (*R1).y + (*R3).y + (*R5).y;
    TI3 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) + C3QB * ((*R3).x - (*R5).x);
    TI5 = ((*R1).y - C3QA * ((*R3).y + (*R5).y)) - C3QB * ((*R3).x - (*R5).x);

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C3QA * TR3 - C3QB * TI3);
    (*R2).x = TR4 + (-C3QA * TR5 - C3QB * TI5);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (C3QB * TR3 + C3QA * TI3);
    (*R2).y = TI4 + (C3QB * TR5 - C3QA * TI5);

    (*R3).x = TR0 - TR1;
    (*R4).x = TR2 - (C3QA * TR3 - C3QB * TI3);
    (*R5).x = TR4 - (-C3QA * TR5 - C3QB * TI5);

    (*R3).y = TI0 - TI1;
    (*R4).y = TI2 - (C3QB * TR3 + C3QA * TI3);
    (*R5).y = TI4 - (C3QB * TR5 - C3QA * TI5);
}

template <typename T>
__device__ void FwdRad7B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6)
{

    T p0;
    T p1;
    T p2;
    T p3;
    T p4;
    T p5;
    T p6;
    T p7;
    T p8;
    T p9;
    T q0;
    T q1;
    T q2;
    T q3;
    T q4;
    T q5;
    T q6;
    T q7;
    T q8;
    /*FFT7 Forward Complex */

    p0 = *R1 + *R6;
    p1 = *R1 - *R6;
    p2 = *R2 + *R5;
    p3 = *R2 - *R5;
    p4 = *R4 + *R3;
    p5 = *R4 - *R3;

    p6 = p2 + p0;
    q4 = p2 - p0;
    q2 = p0 - p4;
    q3 = p4 - p2;
    p7 = p5 + p3;
    q7 = p5 - p3;
    q6 = p1 - p5;
    q8 = p3 - p1;
    q1 = p6 + p4;
    q5 = p7 + p1;
    q0 = *R0 + q1;

    q1 *= C7Q1;
    q2 *= C7Q2;
    q3 *= C7Q3;
    q4 *= C7Q4;

    q5 *= (C7Q5);
    q6 *= (C7Q6);
    q7 *= (C7Q7);
    q8 *= (C7Q8);

    p0 = q0 + q1;
    p1 = q2 + q3;
    p2 = q4 - q3;
    p3 = -q2 - q4;
    p4 = q6 + q7;
    p5 = q8 - q7;
    p6 = -q8 - q6;
    p7 = p0 + p1;
    p8 = p0 + p2;
    p9 = p0 + p3;
    q6 = p4 + q5;
    q7 = p5 + q5;
    q8 = p6 + q5;

    *R0     = q0;
    (*R1).x = p7.x + q6.y;
    (*R1).y = p7.y - q6.x;
    (*R2).x = p9.x + q8.y;
    (*R2).y = p9.y - q8.x;
    (*R3).x = p8.x - q7.y;
    (*R3).y = p8.y + q7.x;
    (*R4).x = p8.x + q7.y;
    (*R4).y = p8.y - q7.x;
    (*R5).x = p9.x - q8.y;
    (*R5).y = p9.y + q8.x;
    (*R6).x = p7.x - q6.y;
    (*R6).y = p7.y + q6.x;
}

template <typename T>
__device__ void InvRad7B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6)
{

    T p0;
    T p1;
    T p2;
    T p3;
    T p4;
    T p5;
    T p6;
    T p7;
    T p8;
    T p9;
    T q0;
    T q1;
    T q2;
    T q3;
    T q4;
    T q5;
    T q6;
    T q7;
    T q8;
    /*FFT7 Backward Complex */

    p0 = *R1 + *R6;
    p1 = *R1 - *R6;
    p2 = *R2 + *R5;
    p3 = *R2 - *R5;
    p4 = *R4 + *R3;
    p5 = *R4 - *R3;

    p6 = p2 + p0;
    q4 = p2 - p0;
    q2 = p0 - p4;
    q3 = p4 - p2;
    p7 = p5 + p3;
    q7 = p5 - p3;
    q6 = p1 - p5;
    q8 = p3 - p1;
    q1 = p6 + p4;
    q5 = p7 + p1;
    q0 = *R0 + q1;

    q1 *= C7Q1;
    q2 *= C7Q2;
    q3 *= C7Q3;
    q4 *= C7Q4;

    q5 *= -(C7Q5);
    q6 *= -(C7Q6);
    q7 *= -(C7Q7);
    q8 *= -(C7Q8);

    p0 = q0 + q1;
    p1 = q2 + q3;
    p2 = q4 - q3;
    p3 = -q2 - q4;
    p4 = q6 + q7;
    p5 = q8 - q7;
    p6 = -q8 - q6;
    p7 = p0 + p1;
    p8 = p0 + p2;
    p9 = p0 + p3;
    q6 = p4 + q5;
    q7 = p5 + q5;
    q8 = p6 + q5;

    *R0     = q0;
    (*R1).x = p7.x + q6.y;
    (*R1).y = p7.y - q6.x;
    (*R2).x = p9.x + q8.y;
    (*R2).y = p9.y - q8.x;
    (*R3).x = p8.x - q7.y;
    (*R3).y = p8.y + q7.x;
    (*R4).x = p8.x + q7.y;
    (*R4).y = p8.y - q7.x;
    (*R5).x = p9.x - q8.y;
    (*R5).y = p9.y + q8.x;
    (*R6).x = p7.x - q6.y;
    (*R6).y = p7.y + q6.x;
}

template <typename T>
__device__ void FwdRad8B1(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + lib_make_vector2<T>(-(*R7).y, (*R7).x);

    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) - C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + lib_make_vector2<T>(-(*R6).y, (*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) - C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename T>
__device__ void InvRad8B1(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{

    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + lib_make_vector2<T>((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) + C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + lib_make_vector2<T>((*R6).y, -(*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) + C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename T>
__device__ void FwdRad9B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8)
{
    // p2 is always multiplied by C9QF, so do it once in p2
    // update R0 and the end since the original R0 is used by others
    // we can also use v3 = R4 + R5 and p3 = R4 - R5
    // but it's ok to do without them and save regs
    T v0 = (*R1) + (*R8);
    T v1 = (*R2) + (*R7);
    T v2 = (*R3) + (*R6);

    T p0 = (*R1) - (*R8);
    T p1 = (*R2) - (*R7);
    T p2 = ((*R3) - (*R6)) * C9QF;

    // borrow R8 as temp
    (*R8) = (C9QB * p0) + (C9QD * p1) + (p2) + (C9QH * ((*R4) - (*R5)));
    (*R1) = ((*R0) + (C9QA * v0) + (C9QC * v1) - (C9QE * v2) - (C9QG * ((*R4) + (*R5))))
            + lib_make_vector2<T>((*R8).y, -(*R8).x);
    (*R8) = (*R1) + 2.0 * lib_make_vector2<T>(-(*R8).y, (*R8).x);
    // borrow R7 as temp
    (*R7) = -(C9QB * ((*R4) - (*R5))) + (C9QD * p0) - (p2) + (C9QH * p1);
    (*R2) = ((*R0) + (C9QA * ((*R4) + (*R5))) + (C9QC * v0) - (C9QE * v2) - (C9QG * v1))
            + lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R7) = (*R2) + 2.0 * lib_make_vector2<T>(-(*R7).y, (*R7).x);
    // borrow R6 temp
    (*R6) = C9QF * (p0 + ((*R4) - (*R5)) - p1);
    (*R3) = ((*R0) + v2 - C9QE * (v0 + v1 + ((*R4) + (*R5))))
            + lib_make_vector2<T>((*R6).y, -(*R6).x);
    (*R6) = (*R3) + 2.0 * lib_make_vector2<T>(-(*R6).y, (*R6).x);
    // borrow p0 as temp
    p0 = -(C9QB * p1) - (C9QD * ((*R4) - (*R5))) + (p2) + (C9QH * p0);
    p1 = (*R0);
    (*R0) += (v0 + v1 + v2 + (*R4) + (*R5));
    (*R4) = (p1 + (C9QA * v1) + (C9QC * ((*R4) + (*R5))) - (C9QE * v2) - (C9QG * v0))
            + lib_make_vector2<T>(p0.y, -p0.x);
    (*R5) = (*R4) + 2.0 * lib_make_vector2<T>(-p0.y, p0.x);
}

template <typename T>
__device__ void InvRad9B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8)
{
    // p2 is always multiplied by C9QF, so do it once in p2
    // update R0 and the end since the original R0 is used by others
    T v0 = (*R1) + (*R8);
    T v1 = (*R2) + (*R7);
    T v2 = (*R3) + (*R6);

    T p0 = (*R1) - (*R8);
    T p1 = (*R2) - (*R7);
    T p2 = ((*R3) - (*R6)) * C9QF;

    // borrow R8 as temp
    (*R8) = (C9QB * p0) + (C9QD * p1) + (p2) + (C9QH * ((*R4) - (*R5)));
    (*R1) = ((*R0) + (C9QA * v0) + (C9QC * v1) - (C9QE * v2) - (C9QG * ((*R4) + (*R5))))
            + lib_make_vector2<T>(-(*R8).y, (*R8).x);
    (*R8) = (*R1) + 2.0 * lib_make_vector2<T>((*R8).y, -(*R8).x);
    // borrow R7 as temp
    (*R7) = -(C9QB * ((*R4) - (*R5))) + (C9QD * p0) - (p2) + (C9QH * p1);
    (*R2) = ((*R0) + (C9QA * ((*R4) + (*R5))) + (C9QC * v0) - (C9QE * v2) - (C9QG * v1))
            + lib_make_vector2<T>(-(*R7).y, (*R7).x);
    (*R7) = (*R2) + 2.0 * lib_make_vector2<T>((*R7).y, -(*R7).x);
    // borrow R6 temp
    (*R6) = C9QF * (p0 + ((*R4) - (*R5)) - p1);
    (*R3) = ((*R0) + v2 - C9QE * (v0 + v1 + ((*R4) + (*R5))))
            + lib_make_vector2<T>(-(*R6).y, (*R6).x);
    (*R6) = (*R3) + 2.0 * lib_make_vector2<T>((*R6).y, -(*R6).x);
    // borrow p0 as temp
    p0 = -(C9QB * p1) - (C9QD * ((*R4) - (*R5))) + (p2) + (C9QH * p0);
    p1 = (*R0);
    (*R0) += (v0 + v1 + v2 + (*R4) + (*R5));
    (*R4) = (p1 + (C9QA * v1) + (C9QC * ((*R4) + (*R5))) - (C9QE * v2) - (C9QG * v0))
            + lib_make_vector2<T>(-p0.y, p0.x);
    (*R5) = (*R4) + 2.0 * lib_make_vector2<T>(p0.y, -p0.x);
}

template <typename T>
__device__ void FwdRad10B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5, TR6, TI6, TR7, TI7,
        TR8, TI8, TR9, TI9;

    TR0 = (*R0).x + (*R2).x + (*R4).x + (*R6).x + (*R8).x;
    TR2 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) + C5QB * ((*R2).y - (*R8).y)
          + C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR8 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) - C5QB * ((*R2).y - (*R8).y)
          - C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) - C5QB * ((*R4).y - (*R6).y)
          + C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));
    TR6 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) + C5QB * ((*R4).y - (*R6).y)
          - C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));

    TI0 = (*R0).y + (*R2).y + (*R4).y + (*R6).y + (*R8).y;
    TI2 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) - C5QB * ((*R2).x - (*R8).x)
          - C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI8 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) + C5QB * ((*R2).x - (*R8).x)
          + C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) + C5QB * ((*R4).x - (*R6).x)
          - C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));
    TI6 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) - C5QB * ((*R4).x - (*R6).x)
          + C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));

    TR1 = (*R1).x + (*R3).x + (*R5).x + (*R7).x + (*R9).x;
    TR3 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) + C5QB * ((*R3).y - (*R9).y)
          + C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR9 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) - C5QB * ((*R3).y - (*R9).y)
          - C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR5 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) - C5QB * ((*R5).y - (*R7).y)
          + C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));
    TR7 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) + C5QB * ((*R5).y - (*R7).y)
          - C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));

    TI1 = (*R1).y + (*R3).y + (*R5).y + (*R7).y + (*R9).y;
    TI3 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) - C5QB * ((*R3).x - (*R9).x)
          - C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI9 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) + C5QB * ((*R3).x - (*R9).x)
          + C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI5 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) + C5QB * ((*R5).x - (*R7).x)
          - C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));
    TI7 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) - C5QB * ((*R5).x - (*R7).x)
          + C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C5QE * TR3 + C5QD * TI3);
    (*R2).x = TR4 + (C5QA * TR5 + C5QB * TI5);
    (*R3).x = TR6 + (-C5QA * TR7 + C5QB * TI7);
    (*R4).x = TR8 + (-C5QE * TR9 + C5QD * TI9);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (-C5QD * TR3 + C5QE * TI3);
    (*R2).y = TI4 + (-C5QB * TR5 + C5QA * TI5);
    (*R3).y = TI6 + (-C5QB * TR7 - C5QA * TI7);
    (*R4).y = TI8 + (-C5QD * TR9 - C5QE * TI9);

    (*R5).x = TR0 - TR1;
    (*R6).x = TR2 - (C5QE * TR3 + C5QD * TI3);
    (*R7).x = TR4 - (C5QA * TR5 + C5QB * TI5);
    (*R8).x = TR6 - (-C5QA * TR7 + C5QB * TI7);
    (*R9).x = TR8 - (-C5QE * TR9 + C5QD * TI9);

    (*R5).y = TI0 - TI1;
    (*R6).y = TI2 - (-C5QD * TR3 + C5QE * TI3);
    (*R7).y = TI4 - (-C5QB * TR5 + C5QA * TI5);
    (*R8).y = TI6 - (-C5QB * TR7 - C5QA * TI7);
    (*R9).y = TI8 - (-C5QD * TR9 - C5QE * TI9);
}

template <typename T>
__device__ void InvRad10B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9)
{

    real_type_t<T> TR0, TI0, TR1, TI1, TR2, TI2, TR3, TI3, TR4, TI4, TR5, TI5, TR6, TI6, TR7, TI7,
        TR8, TI8, TR9, TI9;

    TR0 = (*R0).x + (*R2).x + (*R4).x + (*R6).x + (*R8).x;
    TR2 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) - C5QB * ((*R2).y - (*R8).y)
          - C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR8 = ((*R0).x - C5QC * ((*R4).x + (*R6).x)) + C5QB * ((*R2).y - (*R8).y)
          + C5QD * ((*R4).y - (*R6).y) + C5QA * (((*R2).x - (*R4).x) + ((*R8).x - (*R6).x));
    TR4 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) + C5QB * ((*R4).y - (*R6).y)
          - C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));
    TR6 = ((*R0).x - C5QC * ((*R2).x + (*R8).x)) - C5QB * ((*R4).y - (*R6).y)
          + C5QD * ((*R2).y - (*R8).y) + C5QA * (((*R4).x - (*R2).x) + ((*R6).x - (*R8).x));

    TI0 = (*R0).y + (*R2).y + (*R4).y + (*R6).y + (*R8).y;
    TI2 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) + C5QB * ((*R2).x - (*R8).x)
          + C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI8 = ((*R0).y - C5QC * ((*R4).y + (*R6).y)) - C5QB * ((*R2).x - (*R8).x)
          - C5QD * ((*R4).x - (*R6).x) + C5QA * (((*R2).y - (*R4).y) + ((*R8).y - (*R6).y));
    TI4 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) - C5QB * ((*R4).x - (*R6).x)
          + C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));
    TI6 = ((*R0).y - C5QC * ((*R2).y + (*R8).y)) + C5QB * ((*R4).x - (*R6).x)
          - C5QD * ((*R2).x - (*R8).x) + C5QA * (((*R4).y - (*R2).y) + ((*R6).y - (*R8).y));

    TR1 = (*R1).x + (*R3).x + (*R5).x + (*R7).x + (*R9).x;
    TR3 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) - C5QB * ((*R3).y - (*R9).y)
          - C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR9 = ((*R1).x - C5QC * ((*R5).x + (*R7).x)) + C5QB * ((*R3).y - (*R9).y)
          + C5QD * ((*R5).y - (*R7).y) + C5QA * (((*R3).x - (*R5).x) + ((*R9).x - (*R7).x));
    TR5 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) + C5QB * ((*R5).y - (*R7).y)
          - C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));
    TR7 = ((*R1).x - C5QC * ((*R3).x + (*R9).x)) - C5QB * ((*R5).y - (*R7).y)
          + C5QD * ((*R3).y - (*R9).y) + C5QA * (((*R5).x - (*R3).x) + ((*R7).x - (*R9).x));

    TI1 = (*R1).y + (*R3).y + (*R5).y + (*R7).y + (*R9).y;
    TI3 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) + C5QB * ((*R3).x - (*R9).x)
          + C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI9 = ((*R1).y - C5QC * ((*R5).y + (*R7).y)) - C5QB * ((*R3).x - (*R9).x)
          - C5QD * ((*R5).x - (*R7).x) + C5QA * (((*R3).y - (*R5).y) + ((*R9).y - (*R7).y));
    TI5 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) - C5QB * ((*R5).x - (*R7).x)
          + C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));
    TI7 = ((*R1).y - C5QC * ((*R3).y + (*R9).y)) + C5QB * ((*R5).x - (*R7).x)
          - C5QD * ((*R3).x - (*R9).x) + C5QA * (((*R5).y - (*R3).y) + ((*R7).y - (*R9).y));

    (*R0).x = TR0 + TR1;
    (*R1).x = TR2 + (C5QE * TR3 - C5QD * TI3);
    (*R2).x = TR4 + (C5QA * TR5 - C5QB * TI5);
    (*R3).x = TR6 + (-C5QA * TR7 - C5QB * TI7);
    (*R4).x = TR8 + (-C5QE * TR9 - C5QD * TI9);

    (*R0).y = TI0 + TI1;
    (*R1).y = TI2 + (C5QD * TR3 + C5QE * TI3);
    (*R2).y = TI4 + (C5QB * TR5 + C5QA * TI5);
    (*R3).y = TI6 + (C5QB * TR7 - C5QA * TI7);
    (*R4).y = TI8 + (C5QD * TR9 - C5QE * TI9);

    (*R5).x = TR0 - TR1;
    (*R6).x = TR2 - (C5QE * TR3 - C5QD * TI3);
    (*R7).x = TR4 - (C5QA * TR5 - C5QB * TI5);
    (*R8).x = TR6 - (-C5QA * TR7 - C5QB * TI7);
    (*R9).x = TR8 - (-C5QE * TR9 - C5QD * TI9);

    (*R5).y = TI0 - TI1;
    (*R6).y = TI2 - (C5QD * TR3 + C5QE * TI3);
    (*R7).y = TI4 - (C5QB * TR5 + C5QA * TI5);
    (*R8).y = TI6 - (C5QB * TR7 - C5QA * TI7);
    (*R9).y = TI8 - (C5QD * TR9 - C5QE * TI9);
}

template <typename T>
__device__ void FwdRad16B1(T* R0,
                           T* R8,
                           T* R4,
                           T* R12,
                           T* R2,
                           T* R10,
                           T* R6,
                           T* R14,
                           T* R1,
                           T* R9,
                           T* R5,
                           T* R13,
                           T* R3,
                           T* R11,
                           T* R7,
                           T* R15)
{

    T res;

    (*R1)  = (*R0) - (*R1);
    (*R0)  = 2.0 * (*R0) - (*R1);
    (*R3)  = (*R2) - (*R3);
    (*R2)  = 2.0 * (*R2) - (*R3);
    (*R5)  = (*R4) - (*R5);
    (*R4)  = 2.0 * (*R4) - (*R5);
    (*R7)  = (*R6) - (*R7);
    (*R6)  = 2.0 * (*R6) - (*R7);
    (*R9)  = (*R8) - (*R9);
    (*R8)  = 2.0 * (*R8) - (*R9);
    (*R11) = (*R10) - (*R11);
    (*R10) = 2.0 * (*R10) - (*R11);
    (*R13) = (*R12) - (*R13);
    (*R12) = 2.0 * (*R12) - (*R13);
    (*R15) = (*R14) - (*R15);
    (*R14) = 2.0 * (*R14) - (*R15);

    (*R2)  = (*R0) - (*R2);
    (*R0)  = 2.0 * (*R0) - (*R2);
    (*R3)  = (*R1) + lib_make_vector2<T>(-(*R3).y, (*R3).x);
    (*R1)  = 2.0 * (*R1) - (*R3);
    (*R6)  = (*R4) - (*R6);
    (*R4)  = 2.0 * (*R4) - (*R6);
    (*R7)  = (*R5) + lib_make_vector2<T>(-(*R7).y, (*R7).x);
    (*R5)  = 2.0 * (*R5) - (*R7);
    (*R10) = (*R8) - (*R10);
    (*R8)  = 2.0 * (*R8) - (*R10);
    (*R11) = (*R9) + lib_make_vector2<T>(-(*R11).y, (*R11).x);
    (*R9)  = 2.0 * (*R9) - (*R11);
    (*R14) = (*R12) - (*R14);
    (*R12) = 2.0 * (*R12) - (*R14);
    (*R15) = (*R13) + lib_make_vector2<T>(-(*R15).y, (*R15).x);
    (*R13) = 2.0 * (*R13) - (*R15);

    (*R4)  = (*R0) - (*R4);
    (*R0)  = 2.0 * (*R0) - (*R4);
    (*R5)  = ((*R1) - C8Q * (*R5)) - C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1)  = 2.0 * (*R1) - (*R5);
    (*R6)  = (*R2) + lib_make_vector2<T>(-(*R6).y, (*R6).x);
    (*R2)  = 2.0 * (*R2) - (*R6);
    (*R7)  = ((*R3) + C8Q * (*R7)) - C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3)  = 2.0 * (*R3) - (*R7);
    (*R12) = (*R8) - (*R12);
    (*R8)  = 2.0 * (*R8) - (*R12);
    (*R13) = ((*R9) - C8Q * (*R13)) - C8Q * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R9)  = 2.0 * (*R9) - (*R13);
    (*R14) = (*R10) + lib_make_vector2<T>(-(*R14).y, (*R14).x);
    (*R10) = 2.0 * (*R10) - (*R14);
    (*R15) = ((*R11) + C8Q * (*R15)) - C8Q * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R11) = 2.0 * (*R11) - (*R15);

    (*R8) = (*R0) - (*R8);
    (*R0) = 2.0 * (*R0) - (*R8);
    (*R9) = ((*R1) - C16A * (*R9)) - C16B * lib_make_vector2<T>((*R9).y, -(*R9).x);
    res   = (*R8);
    (*R1) = 2.0 * (*R1) - (*R9);

    (*R10) = ((*R2) - C8Q * (*R10)) - C8Q * lib_make_vector2<T>((*R10).y, -(*R10).x);
    (*R2)  = 2.0 * (*R2) - (*R10);
    (*R11) = ((*R3) - C16B * (*R11)) - C16A * lib_make_vector2<T>((*R11).y, -(*R11).x);
    (*R3)  = 2.0 * (*R3) - (*R11);

    (*R12) = (*R4) + lib_make_vector2<T>(-(*R12).y, (*R12).x);
    (*R4)  = 2.0 * (*R4) - (*R12);
    (*R13) = ((*R5) + C16B * (*R13)) - C16A * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R5)  = 2.0 * (*R5) - (*R13);

    (*R14) = ((*R6) + C8Q * (*R14)) - C8Q * lib_make_vector2<T>((*R14).y, -(*R14).x);
    (*R6)  = 2.0 * (*R6) - (*R14);
    (*R15) = ((*R7) + C16A * (*R15)) - C16B * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R7)  = 2.0 * (*R7) - (*R15);

    res    = (*R1);
    (*R1)  = (*R8);
    (*R8)  = res;
    res    = (*R2);
    (*R2)  = (*R4);
    (*R4)  = res;
    res    = (*R3);
    (*R3)  = (*R12);
    (*R12) = res;
    res    = (*R5);
    (*R5)  = (*R10);
    (*R10) = res;
    res    = (*R7);
    (*R7)  = (*R14);
    (*R14) = res;
    res    = (*R11);
    (*R11) = (*R13);
    (*R13) = res;
}

template <typename T>
__device__ void InvRad16B1(T* R0,
                           T* R8,
                           T* R4,
                           T* R12,
                           T* R2,
                           T* R10,
                           T* R6,
                           T* R14,
                           T* R1,
                           T* R9,
                           T* R5,
                           T* R13,
                           T* R3,
                           T* R11,
                           T* R7,
                           T* R15)
{

    T res;

    (*R1)  = (*R0) - (*R1);
    (*R0)  = 2.0 * (*R0) - (*R1);
    (*R3)  = (*R2) - (*R3);
    (*R2)  = 2.0 * (*R2) - (*R3);
    (*R5)  = (*R4) - (*R5);
    (*R4)  = 2.0 * (*R4) - (*R5);
    (*R7)  = (*R6) - (*R7);
    (*R6)  = 2.0 * (*R6) - (*R7);
    (*R9)  = (*R8) - (*R9);
    (*R8)  = 2.0 * (*R8) - (*R9);
    (*R11) = (*R10) - (*R11);
    (*R10) = 2.0 * (*R10) - (*R11);
    (*R13) = (*R12) - (*R13);
    (*R12) = 2.0 * (*R12) - (*R13);
    (*R15) = (*R14) - (*R15);
    (*R14) = 2.0 * (*R14) - (*R15);

    (*R2)  = (*R0) - (*R2);
    (*R0)  = 2.0 * (*R0) - (*R2);
    (*R3)  = (*R1) + lib_make_vector2<T>((*R3).y, -(*R3).x);
    (*R1)  = 2.0 * (*R1) - (*R3);
    (*R6)  = (*R4) - (*R6);
    (*R4)  = 2.0 * (*R4) - (*R6);
    (*R7)  = (*R5) + lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R5)  = 2.0 * (*R5) - (*R7);
    (*R10) = (*R8) - (*R10);
    (*R8)  = 2.0 * (*R8) - (*R10);
    (*R11) = (*R9) + lib_make_vector2<T>((*R11).y, -(*R11).x);
    (*R9)  = 2.0 * (*R9) - (*R11);
    (*R14) = (*R12) - (*R14);
    (*R12) = 2.0 * (*R12) - (*R14);
    (*R15) = (*R13) + lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R13) = 2.0 * (*R13) - (*R15);

    (*R4)  = (*R0) - (*R4);
    (*R0)  = 2.0 * (*R0) - (*R4);
    (*R5)  = ((*R1) - C8Q * (*R5)) + C8Q * lib_make_vector2<T>((*R5).y, -(*R5).x);
    (*R1)  = 2.0 * (*R1) - (*R5);
    (*R6)  = (*R2) + lib_make_vector2<T>((*R6).y, -(*R6).x);
    (*R2)  = 2.0 * (*R2) - (*R6);
    (*R7)  = ((*R3) + C8Q * (*R7)) + C8Q * lib_make_vector2<T>((*R7).y, -(*R7).x);
    (*R3)  = 2.0 * (*R3) - (*R7);
    (*R12) = (*R8) - (*R12);
    (*R8)  = 2.0 * (*R8) - (*R12);
    (*R13) = ((*R9) - C8Q * (*R13)) + C8Q * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R9)  = 2.0 * (*R9) - (*R13);
    (*R14) = (*R10) + lib_make_vector2<T>((*R14).y, -(*R14).x);
    (*R10) = 2.0 * (*R10) - (*R14);
    (*R15) = ((*R11) + C8Q * (*R15)) + C8Q * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R11) = 2.0 * (*R11) - (*R15);

    (*R8)  = (*R0) - (*R8);
    (*R0)  = 2.0 * (*R0) - (*R8);
    (*R9)  = ((*R1) - C16A * (*R9)) + C16B * lib_make_vector2<T>((*R9).y, -(*R9).x);
    (*R1)  = 2.0 * (*R1) - (*R9);
    (*R10) = ((*R2) - C8Q * (*R10)) + C8Q * lib_make_vector2<T>((*R10).y, -(*R10).x);
    (*R2)  = 2.0 * (*R2) - (*R10);
    (*R11) = ((*R3) - C16B * (*R11)) + C16A * lib_make_vector2<T>((*R11).y, -(*R11).x);
    (*R3)  = 2.0 * (*R3) - (*R11);
    (*R12) = (*R4) + lib_make_vector2<T>((*R12).y, -(*R12).x);
    (*R4)  = 2.0 * (*R4) - (*R12);
    (*R13) = ((*R5) + C16B * (*R13)) + C16A * lib_make_vector2<T>((*R13).y, -(*R13).x);
    (*R5)  = 2.0 * (*R5) - (*R13);
    (*R14) = ((*R6) + C8Q * (*R14)) + C8Q * lib_make_vector2<T>((*R14).y, -(*R14).x);
    (*R6)  = 2.0 * (*R6) - (*R14);
    (*R15) = ((*R7) + C16A * (*R15)) + C16B * lib_make_vector2<T>((*R15).y, -(*R15).x);
    (*R7)  = 2.0 * (*R7) - (*R15);

    res    = (*R1);
    (*R1)  = (*R8);
    (*R8)  = res;
    res    = (*R2);
    (*R2)  = (*R4);
    (*R4)  = res;
    res    = (*R3);
    (*R3)  = (*R12);
    (*R12) = res;
    res    = (*R5);
    (*R5)  = (*R10);
    (*R10) = res;
    res    = (*R7);
    (*R7)  = (*R14);
    (*R14) = res;
    res    = (*R11);
    (*R11) = (*R13);
    (*R13) = res;
}

template <typename T>
__device__ void
    FwdRad11B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, dp, dm;

    x0  = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    dp  = (*R1) + (*R10);
    dm  = (*R1) - (*R10);
    x1.x += Q11i1j1R * dp.x - Q11i1j1I * dm.y;
    x1.y += Q11i1j1R * dp.y + Q11i1j1I * dm.x;
    x10.x += Q11i1j1R * dp.x + Q11i1j1I * dm.y;
    x10.y += Q11i1j1R * dp.y - Q11i1j1I * dm.x;
    x2.x += Q11i2j1R * dp.x - Q11i2j1I * dm.y;
    x2.y += Q11i2j1R * dp.y + Q11i2j1I * dm.x;
    x9.x += Q11i2j1R * dp.x + Q11i2j1I * dm.y;
    x9.y += Q11i2j1R * dp.y - Q11i2j1I * dm.x;
    x3.x += Q11i3j1R * dp.x - Q11i3j1I * dm.y;
    x3.y += Q11i3j1R * dp.y + Q11i3j1I * dm.x;
    x8.x += Q11i3j1R * dp.x + Q11i3j1I * dm.y;
    x8.y += Q11i3j1R * dp.y - Q11i3j1I * dm.x;
    x4.x += Q11i4j1R * dp.x - Q11i4j1I * dm.y;
    x4.y += Q11i4j1R * dp.y + Q11i4j1I * dm.x;
    x7.x += Q11i4j1R * dp.x + Q11i4j1I * dm.y;
    x7.y += Q11i4j1R * dp.y - Q11i4j1I * dm.x;
    x5.x += Q11i5j1R * dp.x - Q11i5j1I * dm.y;
    x5.y += Q11i5j1R * dp.y + Q11i5j1I * dm.x;
    x6.x += Q11i5j1R * dp.x + Q11i5j1I * dm.y;
    x6.y += Q11i5j1R * dp.y - Q11i5j1I * dm.x;
    dp = (*R2) + (*R9);
    dm = (*R2) - (*R9);
    x1.x += Q11i1j2R * dp.x - Q11i1j2I * dm.y;
    x1.y += Q11i1j2R * dp.y + Q11i1j2I * dm.x;
    x10.x += Q11i1j2R * dp.x + Q11i1j2I * dm.y;
    x10.y += Q11i1j2R * dp.y - Q11i1j2I * dm.x;
    x2.x += Q11i2j2R * dp.x - Q11i2j2I * dm.y;
    x2.y += Q11i2j2R * dp.y + Q11i2j2I * dm.x;
    x9.x += Q11i2j2R * dp.x + Q11i2j2I * dm.y;
    x9.y += Q11i2j2R * dp.y - Q11i2j2I * dm.x;
    x3.x += Q11i3j2R * dp.x - Q11i3j2I * dm.y;
    x3.y += Q11i3j2R * dp.y + Q11i3j2I * dm.x;
    x8.x += Q11i3j2R * dp.x + Q11i3j2I * dm.y;
    x8.y += Q11i3j2R * dp.y - Q11i3j2I * dm.x;
    x4.x += Q11i4j2R * dp.x - Q11i4j2I * dm.y;
    x4.y += Q11i4j2R * dp.y + Q11i4j2I * dm.x;
    x7.x += Q11i4j2R * dp.x + Q11i4j2I * dm.y;
    x7.y += Q11i4j2R * dp.y - Q11i4j2I * dm.x;
    x5.x += Q11i5j2R * dp.x - Q11i5j2I * dm.y;
    x5.y += Q11i5j2R * dp.y + Q11i5j2I * dm.x;
    x6.x += Q11i5j2R * dp.x + Q11i5j2I * dm.y;
    x6.y += Q11i5j2R * dp.y - Q11i5j2I * dm.x;
    dp = (*R3) + (*R8);
    dm = (*R3) - (*R8);
    x1.x += Q11i1j3R * dp.x - Q11i1j3I * dm.y;
    x1.y += Q11i1j3R * dp.y + Q11i1j3I * dm.x;
    x10.x += Q11i1j3R * dp.x + Q11i1j3I * dm.y;
    x10.y += Q11i1j3R * dp.y - Q11i1j3I * dm.x;
    x2.x += Q11i2j3R * dp.x - Q11i2j3I * dm.y;
    x2.y += Q11i2j3R * dp.y + Q11i2j3I * dm.x;
    x9.x += Q11i2j3R * dp.x + Q11i2j3I * dm.y;
    x9.y += Q11i2j3R * dp.y - Q11i2j3I * dm.x;
    x3.x += Q11i3j3R * dp.x - Q11i3j3I * dm.y;
    x3.y += Q11i3j3R * dp.y + Q11i3j3I * dm.x;
    x8.x += Q11i3j3R * dp.x + Q11i3j3I * dm.y;
    x8.y += Q11i3j3R * dp.y - Q11i3j3I * dm.x;
    x4.x += Q11i4j3R * dp.x - Q11i4j3I * dm.y;
    x4.y += Q11i4j3R * dp.y + Q11i4j3I * dm.x;
    x7.x += Q11i4j3R * dp.x + Q11i4j3I * dm.y;
    x7.y += Q11i4j3R * dp.y - Q11i4j3I * dm.x;
    x5.x += Q11i5j3R * dp.x - Q11i5j3I * dm.y;
    x5.y += Q11i5j3R * dp.y + Q11i5j3I * dm.x;
    x6.x += Q11i5j3R * dp.x + Q11i5j3I * dm.y;
    x6.y += Q11i5j3R * dp.y - Q11i5j3I * dm.x;
    dp = (*R4) + (*R7);
    dm = (*R4) - (*R7);
    x1.x += Q11i1j4R * dp.x - Q11i1j4I * dm.y;
    x1.y += Q11i1j4R * dp.y + Q11i1j4I * dm.x;
    x10.x += Q11i1j4R * dp.x + Q11i1j4I * dm.y;
    x10.y += Q11i1j4R * dp.y - Q11i1j4I * dm.x;
    x2.x += Q11i2j4R * dp.x - Q11i2j4I * dm.y;
    x2.y += Q11i2j4R * dp.y + Q11i2j4I * dm.x;
    x9.x += Q11i2j4R * dp.x + Q11i2j4I * dm.y;
    x9.y += Q11i2j4R * dp.y - Q11i2j4I * dm.x;
    x3.x += Q11i3j4R * dp.x - Q11i3j4I * dm.y;
    x3.y += Q11i3j4R * dp.y + Q11i3j4I * dm.x;
    x8.x += Q11i3j4R * dp.x + Q11i3j4I * dm.y;
    x8.y += Q11i3j4R * dp.y - Q11i3j4I * dm.x;
    x4.x += Q11i4j4R * dp.x - Q11i4j4I * dm.y;
    x4.y += Q11i4j4R * dp.y + Q11i4j4I * dm.x;
    x7.x += Q11i4j4R * dp.x + Q11i4j4I * dm.y;
    x7.y += Q11i4j4R * dp.y - Q11i4j4I * dm.x;
    x5.x += Q11i5j4R * dp.x - Q11i5j4I * dm.y;
    x5.y += Q11i5j4R * dp.y + Q11i5j4I * dm.x;
    x6.x += Q11i5j4R * dp.x + Q11i5j4I * dm.y;
    x6.y += Q11i5j4R * dp.y - Q11i5j4I * dm.x;
    dp = (*R5) + (*R6);
    dm = (*R5) - (*R6);
    x1.x += Q11i1j5R * dp.x - Q11i1j5I * dm.y;
    x1.y += Q11i1j5R * dp.y + Q11i1j5I * dm.x;
    x10.x += Q11i1j5R * dp.x + Q11i1j5I * dm.y;
    x10.y += Q11i1j5R * dp.y - Q11i1j5I * dm.x;
    x2.x += Q11i2j5R * dp.x - Q11i2j5I * dm.y;
    x2.y += Q11i2j5R * dp.y + Q11i2j5I * dm.x;
    x9.x += Q11i2j5R * dp.x + Q11i2j5I * dm.y;
    x9.y += Q11i2j5R * dp.y - Q11i2j5I * dm.x;
    x3.x += Q11i3j5R * dp.x - Q11i3j5I * dm.y;
    x3.y += Q11i3j5R * dp.y + Q11i3j5I * dm.x;
    x8.x += Q11i3j5R * dp.x + Q11i3j5I * dm.y;
    x8.y += Q11i3j5R * dp.y - Q11i3j5I * dm.x;
    x4.x += Q11i4j5R * dp.x - Q11i4j5I * dm.y;
    x4.y += Q11i4j5R * dp.y + Q11i4j5I * dm.x;
    x7.x += Q11i4j5R * dp.x + Q11i4j5I * dm.y;
    x7.y += Q11i4j5R * dp.y - Q11i4j5I * dm.x;
    x5.x += Q11i5j5R * dp.x - Q11i5j5I * dm.y;
    x5.y += Q11i5j5R * dp.y + Q11i5j5I * dm.x;
    x6.x += Q11i5j5R * dp.x + Q11i5j5I * dm.y;
    x6.y += Q11i5j5R * dp.y - Q11i5j5I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
}

template <typename T>
__device__ void
    InvRad11B1(T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, dp, dm;

    x0  = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    dp  = (*R1) + (*R10);
    dm  = (*R1) - (*R10);
    x1.x += Q11i1j1R * dp.x + Q11i1j1I * dm.y;
    x1.y += Q11i1j1R * dp.y - Q11i1j1I * dm.x;
    x10.x += Q11i1j1R * dp.x - Q11i1j1I * dm.y;
    x10.y += Q11i1j1R * dp.y + Q11i1j1I * dm.x;
    x2.x += Q11i2j1R * dp.x + Q11i2j1I * dm.y;
    x2.y += Q11i2j1R * dp.y - Q11i2j1I * dm.x;
    x9.x += Q11i2j1R * dp.x - Q11i2j1I * dm.y;
    x9.y += Q11i2j1R * dp.y + Q11i2j1I * dm.x;
    x3.x += Q11i3j1R * dp.x + Q11i3j1I * dm.y;
    x3.y += Q11i3j1R * dp.y - Q11i3j1I * dm.x;
    x8.x += Q11i3j1R * dp.x - Q11i3j1I * dm.y;
    x8.y += Q11i3j1R * dp.y + Q11i3j1I * dm.x;
    x4.x += Q11i4j1R * dp.x + Q11i4j1I * dm.y;
    x4.y += Q11i4j1R * dp.y - Q11i4j1I * dm.x;
    x7.x += Q11i4j1R * dp.x - Q11i4j1I * dm.y;
    x7.y += Q11i4j1R * dp.y + Q11i4j1I * dm.x;
    x5.x += Q11i5j1R * dp.x + Q11i5j1I * dm.y;
    x5.y += Q11i5j1R * dp.y - Q11i5j1I * dm.x;
    x6.x += Q11i5j1R * dp.x - Q11i5j1I * dm.y;
    x6.y += Q11i5j1R * dp.y + Q11i5j1I * dm.x;
    dp = (*R2) + (*R9);
    dm = (*R2) - (*R9);
    x1.x += Q11i1j2R * dp.x + Q11i1j2I * dm.y;
    x1.y += Q11i1j2R * dp.y - Q11i1j2I * dm.x;
    x10.x += Q11i1j2R * dp.x - Q11i1j2I * dm.y;
    x10.y += Q11i1j2R * dp.y + Q11i1j2I * dm.x;
    x2.x += Q11i2j2R * dp.x + Q11i2j2I * dm.y;
    x2.y += Q11i2j2R * dp.y - Q11i2j2I * dm.x;
    x9.x += Q11i2j2R * dp.x - Q11i2j2I * dm.y;
    x9.y += Q11i2j2R * dp.y + Q11i2j2I * dm.x;
    x3.x += Q11i3j2R * dp.x + Q11i3j2I * dm.y;
    x3.y += Q11i3j2R * dp.y - Q11i3j2I * dm.x;
    x8.x += Q11i3j2R * dp.x - Q11i3j2I * dm.y;
    x8.y += Q11i3j2R * dp.y + Q11i3j2I * dm.x;
    x4.x += Q11i4j2R * dp.x + Q11i4j2I * dm.y;
    x4.y += Q11i4j2R * dp.y - Q11i4j2I * dm.x;
    x7.x += Q11i4j2R * dp.x - Q11i4j2I * dm.y;
    x7.y += Q11i4j2R * dp.y + Q11i4j2I * dm.x;
    x5.x += Q11i5j2R * dp.x + Q11i5j2I * dm.y;
    x5.y += Q11i5j2R * dp.y - Q11i5j2I * dm.x;
    x6.x += Q11i5j2R * dp.x - Q11i5j2I * dm.y;
    x6.y += Q11i5j2R * dp.y + Q11i5j2I * dm.x;
    dp = (*R3) + (*R8);
    dm = (*R3) - (*R8);
    x1.x += Q11i1j3R * dp.x + Q11i1j3I * dm.y;
    x1.y += Q11i1j3R * dp.y - Q11i1j3I * dm.x;
    x10.x += Q11i1j3R * dp.x - Q11i1j3I * dm.y;
    x10.y += Q11i1j3R * dp.y + Q11i1j3I * dm.x;
    x2.x += Q11i2j3R * dp.x + Q11i2j3I * dm.y;
    x2.y += Q11i2j3R * dp.y - Q11i2j3I * dm.x;
    x9.x += Q11i2j3R * dp.x - Q11i2j3I * dm.y;
    x9.y += Q11i2j3R * dp.y + Q11i2j3I * dm.x;
    x3.x += Q11i3j3R * dp.x + Q11i3j3I * dm.y;
    x3.y += Q11i3j3R * dp.y - Q11i3j3I * dm.x;
    x8.x += Q11i3j3R * dp.x - Q11i3j3I * dm.y;
    x8.y += Q11i3j3R * dp.y + Q11i3j3I * dm.x;
    x4.x += Q11i4j3R * dp.x + Q11i4j3I * dm.y;
    x4.y += Q11i4j3R * dp.y - Q11i4j3I * dm.x;
    x7.x += Q11i4j3R * dp.x - Q11i4j3I * dm.y;
    x7.y += Q11i4j3R * dp.y + Q11i4j3I * dm.x;
    x5.x += Q11i5j3R * dp.x + Q11i5j3I * dm.y;
    x5.y += Q11i5j3R * dp.y - Q11i5j3I * dm.x;
    x6.x += Q11i5j3R * dp.x - Q11i5j3I * dm.y;
    x6.y += Q11i5j3R * dp.y + Q11i5j3I * dm.x;
    dp = (*R4) + (*R7);
    dm = (*R4) - (*R7);
    x1.x += Q11i1j4R * dp.x + Q11i1j4I * dm.y;
    x1.y += Q11i1j4R * dp.y - Q11i1j4I * dm.x;
    x10.x += Q11i1j4R * dp.x - Q11i1j4I * dm.y;
    x10.y += Q11i1j4R * dp.y + Q11i1j4I * dm.x;
    x2.x += Q11i2j4R * dp.x + Q11i2j4I * dm.y;
    x2.y += Q11i2j4R * dp.y - Q11i2j4I * dm.x;
    x9.x += Q11i2j4R * dp.x - Q11i2j4I * dm.y;
    x9.y += Q11i2j4R * dp.y + Q11i2j4I * dm.x;
    x3.x += Q11i3j4R * dp.x + Q11i3j4I * dm.y;
    x3.y += Q11i3j4R * dp.y - Q11i3j4I * dm.x;
    x8.x += Q11i3j4R * dp.x - Q11i3j4I * dm.y;
    x8.y += Q11i3j4R * dp.y + Q11i3j4I * dm.x;
    x4.x += Q11i4j4R * dp.x + Q11i4j4I * dm.y;
    x4.y += Q11i4j4R * dp.y - Q11i4j4I * dm.x;
    x7.x += Q11i4j4R * dp.x - Q11i4j4I * dm.y;
    x7.y += Q11i4j4R * dp.y + Q11i4j4I * dm.x;
    x5.x += Q11i5j4R * dp.x + Q11i5j4I * dm.y;
    x5.y += Q11i5j4R * dp.y - Q11i5j4I * dm.x;
    x6.x += Q11i5j4R * dp.x - Q11i5j4I * dm.y;
    x6.y += Q11i5j4R * dp.y + Q11i5j4I * dm.x;
    dp = (*R5) + (*R6);
    dm = (*R5) - (*R6);
    x1.x += Q11i1j5R * dp.x + Q11i1j5I * dm.y;
    x1.y += Q11i1j5R * dp.y - Q11i1j5I * dm.x;
    x10.x += Q11i1j5R * dp.x - Q11i1j5I * dm.y;
    x10.y += Q11i1j5R * dp.y + Q11i1j5I * dm.x;
    x2.x += Q11i2j5R * dp.x + Q11i2j5I * dm.y;
    x2.y += Q11i2j5R * dp.y - Q11i2j5I * dm.x;
    x9.x += Q11i2j5R * dp.x - Q11i2j5I * dm.y;
    x9.y += Q11i2j5R * dp.y + Q11i2j5I * dm.x;
    x3.x += Q11i3j5R * dp.x + Q11i3j5I * dm.y;
    x3.y += Q11i3j5R * dp.y - Q11i3j5I * dm.x;
    x8.x += Q11i3j5R * dp.x - Q11i3j5I * dm.y;
    x8.y += Q11i3j5R * dp.y + Q11i3j5I * dm.x;
    x4.x += Q11i4j5R * dp.x + Q11i4j5I * dm.y;
    x4.y += Q11i4j5R * dp.y - Q11i4j5I * dm.x;
    x7.x += Q11i4j5R * dp.x - Q11i4j5I * dm.y;
    x7.y += Q11i4j5R * dp.y + Q11i4j5I * dm.x;
    x5.x += Q11i5j5R * dp.x + Q11i5j5I * dm.y;
    x5.y += Q11i5j5R * dp.y - Q11i5j5I * dm.x;
    x6.x += Q11i5j5R * dp.x - Q11i5j5I * dm.y;
    x6.y += Q11i5j5R * dp.y + Q11i5j5I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
}

template <typename T>
__device__ void FwdRad13B1(
    T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10, T* R11, T* R12)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    dp  = (*R1) + (*R12);
    dm  = (*R1) - (*R12);
    x1.x += Q13i1j1R * dp.x - Q13i1j1I * dm.y;
    x1.y += Q13i1j1R * dp.y + Q13i1j1I * dm.x;
    x12.x += Q13i1j1R * dp.x + Q13i1j1I * dm.y;
    x12.y += Q13i1j1R * dp.y - Q13i1j1I * dm.x;
    x2.x += Q13i2j1R * dp.x - Q13i2j1I * dm.y;
    x2.y += Q13i2j1R * dp.y + Q13i2j1I * dm.x;
    x11.x += Q13i2j1R * dp.x + Q13i2j1I * dm.y;
    x11.y += Q13i2j1R * dp.y - Q13i2j1I * dm.x;
    x3.x += Q13i3j1R * dp.x - Q13i3j1I * dm.y;
    x3.y += Q13i3j1R * dp.y + Q13i3j1I * dm.x;
    x10.x += Q13i3j1R * dp.x + Q13i3j1I * dm.y;
    x10.y += Q13i3j1R * dp.y - Q13i3j1I * dm.x;
    x4.x += Q13i4j1R * dp.x - Q13i4j1I * dm.y;
    x4.y += Q13i4j1R * dp.y + Q13i4j1I * dm.x;
    x9.x += Q13i4j1R * dp.x + Q13i4j1I * dm.y;
    x9.y += Q13i4j1R * dp.y - Q13i4j1I * dm.x;
    x5.x += Q13i5j1R * dp.x - Q13i5j1I * dm.y;
    x5.y += Q13i5j1R * dp.y + Q13i5j1I * dm.x;
    x8.x += Q13i5j1R * dp.x + Q13i5j1I * dm.y;
    x8.y += Q13i5j1R * dp.y - Q13i5j1I * dm.x;
    x6.x += Q13i6j1R * dp.x - Q13i6j1I * dm.y;
    x6.y += Q13i6j1R * dp.y + Q13i6j1I * dm.x;
    x7.x += Q13i6j1R * dp.x + Q13i6j1I * dm.y;
    x7.y += Q13i6j1R * dp.y - Q13i6j1I * dm.x;
    dp = (*R2) + (*R11);
    dm = (*R2) - (*R11);
    x1.x += Q13i1j2R * dp.x - Q13i1j2I * dm.y;
    x1.y += Q13i1j2R * dp.y + Q13i1j2I * dm.x;
    x12.x += Q13i1j2R * dp.x + Q13i1j2I * dm.y;
    x12.y += Q13i1j2R * dp.y - Q13i1j2I * dm.x;
    x2.x += Q13i2j2R * dp.x - Q13i2j2I * dm.y;
    x2.y += Q13i2j2R * dp.y + Q13i2j2I * dm.x;
    x11.x += Q13i2j2R * dp.x + Q13i2j2I * dm.y;
    x11.y += Q13i2j2R * dp.y - Q13i2j2I * dm.x;
    x3.x += Q13i3j2R * dp.x - Q13i3j2I * dm.y;
    x3.y += Q13i3j2R * dp.y + Q13i3j2I * dm.x;
    x10.x += Q13i3j2R * dp.x + Q13i3j2I * dm.y;
    x10.y += Q13i3j2R * dp.y - Q13i3j2I * dm.x;
    x4.x += Q13i4j2R * dp.x - Q13i4j2I * dm.y;
    x4.y += Q13i4j2R * dp.y + Q13i4j2I * dm.x;
    x9.x += Q13i4j2R * dp.x + Q13i4j2I * dm.y;
    x9.y += Q13i4j2R * dp.y - Q13i4j2I * dm.x;
    x5.x += Q13i5j2R * dp.x - Q13i5j2I * dm.y;
    x5.y += Q13i5j2R * dp.y + Q13i5j2I * dm.x;
    x8.x += Q13i5j2R * dp.x + Q13i5j2I * dm.y;
    x8.y += Q13i5j2R * dp.y - Q13i5j2I * dm.x;
    x6.x += Q13i6j2R * dp.x - Q13i6j2I * dm.y;
    x6.y += Q13i6j2R * dp.y + Q13i6j2I * dm.x;
    x7.x += Q13i6j2R * dp.x + Q13i6j2I * dm.y;
    x7.y += Q13i6j2R * dp.y - Q13i6j2I * dm.x;
    dp = (*R3) + (*R10);
    dm = (*R3) - (*R10);
    x1.x += Q13i1j3R * dp.x - Q13i1j3I * dm.y;
    x1.y += Q13i1j3R * dp.y + Q13i1j3I * dm.x;
    x12.x += Q13i1j3R * dp.x + Q13i1j3I * dm.y;
    x12.y += Q13i1j3R * dp.y - Q13i1j3I * dm.x;
    x2.x += Q13i2j3R * dp.x - Q13i2j3I * dm.y;
    x2.y += Q13i2j3R * dp.y + Q13i2j3I * dm.x;
    x11.x += Q13i2j3R * dp.x + Q13i2j3I * dm.y;
    x11.y += Q13i2j3R * dp.y - Q13i2j3I * dm.x;
    x3.x += Q13i3j3R * dp.x - Q13i3j3I * dm.y;
    x3.y += Q13i3j3R * dp.y + Q13i3j3I * dm.x;
    x10.x += Q13i3j3R * dp.x + Q13i3j3I * dm.y;
    x10.y += Q13i3j3R * dp.y - Q13i3j3I * dm.x;
    x4.x += Q13i4j3R * dp.x - Q13i4j3I * dm.y;
    x4.y += Q13i4j3R * dp.y + Q13i4j3I * dm.x;
    x9.x += Q13i4j3R * dp.x + Q13i4j3I * dm.y;
    x9.y += Q13i4j3R * dp.y - Q13i4j3I * dm.x;
    x5.x += Q13i5j3R * dp.x - Q13i5j3I * dm.y;
    x5.y += Q13i5j3R * dp.y + Q13i5j3I * dm.x;
    x8.x += Q13i5j3R * dp.x + Q13i5j3I * dm.y;
    x8.y += Q13i5j3R * dp.y - Q13i5j3I * dm.x;
    x6.x += Q13i6j3R * dp.x - Q13i6j3I * dm.y;
    x6.y += Q13i6j3R * dp.y + Q13i6j3I * dm.x;
    x7.x += Q13i6j3R * dp.x + Q13i6j3I * dm.y;
    x7.y += Q13i6j3R * dp.y - Q13i6j3I * dm.x;
    dp = (*R4) + (*R9);
    dm = (*R4) - (*R9);
    x1.x += Q13i1j4R * dp.x - Q13i1j4I * dm.y;
    x1.y += Q13i1j4R * dp.y + Q13i1j4I * dm.x;
    x12.x += Q13i1j4R * dp.x + Q13i1j4I * dm.y;
    x12.y += Q13i1j4R * dp.y - Q13i1j4I * dm.x;
    x2.x += Q13i2j4R * dp.x - Q13i2j4I * dm.y;
    x2.y += Q13i2j4R * dp.y + Q13i2j4I * dm.x;
    x11.x += Q13i2j4R * dp.x + Q13i2j4I * dm.y;
    x11.y += Q13i2j4R * dp.y - Q13i2j4I * dm.x;
    x3.x += Q13i3j4R * dp.x - Q13i3j4I * dm.y;
    x3.y += Q13i3j4R * dp.y + Q13i3j4I * dm.x;
    x10.x += Q13i3j4R * dp.x + Q13i3j4I * dm.y;
    x10.y += Q13i3j4R * dp.y - Q13i3j4I * dm.x;
    x4.x += Q13i4j4R * dp.x - Q13i4j4I * dm.y;
    x4.y += Q13i4j4R * dp.y + Q13i4j4I * dm.x;
    x9.x += Q13i4j4R * dp.x + Q13i4j4I * dm.y;
    x9.y += Q13i4j4R * dp.y - Q13i4j4I * dm.x;
    x5.x += Q13i5j4R * dp.x - Q13i5j4I * dm.y;
    x5.y += Q13i5j4R * dp.y + Q13i5j4I * dm.x;
    x8.x += Q13i5j4R * dp.x + Q13i5j4I * dm.y;
    x8.y += Q13i5j4R * dp.y - Q13i5j4I * dm.x;
    x6.x += Q13i6j4R * dp.x - Q13i6j4I * dm.y;
    x6.y += Q13i6j4R * dp.y + Q13i6j4I * dm.x;
    x7.x += Q13i6j4R * dp.x + Q13i6j4I * dm.y;
    x7.y += Q13i6j4R * dp.y - Q13i6j4I * dm.x;
    dp = (*R5) + (*R8);
    dm = (*R5) - (*R8);
    x1.x += Q13i1j5R * dp.x - Q13i1j5I * dm.y;
    x1.y += Q13i1j5R * dp.y + Q13i1j5I * dm.x;
    x12.x += Q13i1j5R * dp.x + Q13i1j5I * dm.y;
    x12.y += Q13i1j5R * dp.y - Q13i1j5I * dm.x;
    x2.x += Q13i2j5R * dp.x - Q13i2j5I * dm.y;
    x2.y += Q13i2j5R * dp.y + Q13i2j5I * dm.x;
    x11.x += Q13i2j5R * dp.x + Q13i2j5I * dm.y;
    x11.y += Q13i2j5R * dp.y - Q13i2j5I * dm.x;
    x3.x += Q13i3j5R * dp.x - Q13i3j5I * dm.y;
    x3.y += Q13i3j5R * dp.y + Q13i3j5I * dm.x;
    x10.x += Q13i3j5R * dp.x + Q13i3j5I * dm.y;
    x10.y += Q13i3j5R * dp.y - Q13i3j5I * dm.x;
    x4.x += Q13i4j5R * dp.x - Q13i4j5I * dm.y;
    x4.y += Q13i4j5R * dp.y + Q13i4j5I * dm.x;
    x9.x += Q13i4j5R * dp.x + Q13i4j5I * dm.y;
    x9.y += Q13i4j5R * dp.y - Q13i4j5I * dm.x;
    x5.x += Q13i5j5R * dp.x - Q13i5j5I * dm.y;
    x5.y += Q13i5j5R * dp.y + Q13i5j5I * dm.x;
    x8.x += Q13i5j5R * dp.x + Q13i5j5I * dm.y;
    x8.y += Q13i5j5R * dp.y - Q13i5j5I * dm.x;
    x6.x += Q13i6j5R * dp.x - Q13i6j5I * dm.y;
    x6.y += Q13i6j5R * dp.y + Q13i6j5I * dm.x;
    x7.x += Q13i6j5R * dp.x + Q13i6j5I * dm.y;
    x7.y += Q13i6j5R * dp.y - Q13i6j5I * dm.x;
    dp = (*R6) + (*R7);
    dm = (*R6) - (*R7);
    x1.x += Q13i1j6R * dp.x - Q13i1j6I * dm.y;
    x1.y += Q13i1j6R * dp.y + Q13i1j6I * dm.x;
    x12.x += Q13i1j6R * dp.x + Q13i1j6I * dm.y;
    x12.y += Q13i1j6R * dp.y - Q13i1j6I * dm.x;
    x2.x += Q13i2j6R * dp.x - Q13i2j6I * dm.y;
    x2.y += Q13i2j6R * dp.y + Q13i2j6I * dm.x;
    x11.x += Q13i2j6R * dp.x + Q13i2j6I * dm.y;
    x11.y += Q13i2j6R * dp.y - Q13i2j6I * dm.x;
    x3.x += Q13i3j6R * dp.x - Q13i3j6I * dm.y;
    x3.y += Q13i3j6R * dp.y + Q13i3j6I * dm.x;
    x10.x += Q13i3j6R * dp.x + Q13i3j6I * dm.y;
    x10.y += Q13i3j6R * dp.y - Q13i3j6I * dm.x;
    x4.x += Q13i4j6R * dp.x - Q13i4j6I * dm.y;
    x4.y += Q13i4j6R * dp.y + Q13i4j6I * dm.x;
    x9.x += Q13i4j6R * dp.x + Q13i4j6I * dm.y;
    x9.y += Q13i4j6R * dp.y - Q13i4j6I * dm.x;
    x5.x += Q13i5j6R * dp.x - Q13i5j6I * dm.y;
    x5.y += Q13i5j6R * dp.y + Q13i5j6I * dm.x;
    x8.x += Q13i5j6R * dp.x + Q13i5j6I * dm.y;
    x8.y += Q13i5j6R * dp.y - Q13i5j6I * dm.x;
    x6.x += Q13i6j6R * dp.x - Q13i6j6I * dm.y;
    x6.y += Q13i6j6R * dp.y + Q13i6j6I * dm.x;
    x7.x += Q13i6j6R * dp.x + Q13i6j6I * dm.y;
    x7.y += Q13i6j6R * dp.y - Q13i6j6I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
}

template <typename T>
__device__ void InvRad13B1(
    T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10, T* R11, T* R12)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    dp  = (*R1) + (*R12);
    dm  = (*R1) - (*R12);
    x1.x += Q13i1j1R * dp.x + Q13i1j1I * dm.y;
    x1.y += Q13i1j1R * dp.y - Q13i1j1I * dm.x;
    x12.x += Q13i1j1R * dp.x - Q13i1j1I * dm.y;
    x12.y += Q13i1j1R * dp.y + Q13i1j1I * dm.x;
    x2.x += Q13i2j1R * dp.x + Q13i2j1I * dm.y;
    x2.y += Q13i2j1R * dp.y - Q13i2j1I * dm.x;
    x11.x += Q13i2j1R * dp.x - Q13i2j1I * dm.y;
    x11.y += Q13i2j1R * dp.y + Q13i2j1I * dm.x;
    x3.x += Q13i3j1R * dp.x + Q13i3j1I * dm.y;
    x3.y += Q13i3j1R * dp.y - Q13i3j1I * dm.x;
    x10.x += Q13i3j1R * dp.x - Q13i3j1I * dm.y;
    x10.y += Q13i3j1R * dp.y + Q13i3j1I * dm.x;
    x4.x += Q13i4j1R * dp.x + Q13i4j1I * dm.y;
    x4.y += Q13i4j1R * dp.y - Q13i4j1I * dm.x;
    x9.x += Q13i4j1R * dp.x - Q13i4j1I * dm.y;
    x9.y += Q13i4j1R * dp.y + Q13i4j1I * dm.x;
    x5.x += Q13i5j1R * dp.x + Q13i5j1I * dm.y;
    x5.y += Q13i5j1R * dp.y - Q13i5j1I * dm.x;
    x8.x += Q13i5j1R * dp.x - Q13i5j1I * dm.y;
    x8.y += Q13i5j1R * dp.y + Q13i5j1I * dm.x;
    x6.x += Q13i6j1R * dp.x + Q13i6j1I * dm.y;
    x6.y += Q13i6j1R * dp.y - Q13i6j1I * dm.x;
    x7.x += Q13i6j1R * dp.x - Q13i6j1I * dm.y;
    x7.y += Q13i6j1R * dp.y + Q13i6j1I * dm.x;
    dp = (*R2) + (*R11);
    dm = (*R2) - (*R11);
    x1.x += Q13i1j2R * dp.x + Q13i1j2I * dm.y;
    x1.y += Q13i1j2R * dp.y - Q13i1j2I * dm.x;
    x12.x += Q13i1j2R * dp.x - Q13i1j2I * dm.y;
    x12.y += Q13i1j2R * dp.y + Q13i1j2I * dm.x;
    x2.x += Q13i2j2R * dp.x + Q13i2j2I * dm.y;
    x2.y += Q13i2j2R * dp.y - Q13i2j2I * dm.x;
    x11.x += Q13i2j2R * dp.x - Q13i2j2I * dm.y;
    x11.y += Q13i2j2R * dp.y + Q13i2j2I * dm.x;
    x3.x += Q13i3j2R * dp.x + Q13i3j2I * dm.y;
    x3.y += Q13i3j2R * dp.y - Q13i3j2I * dm.x;
    x10.x += Q13i3j2R * dp.x - Q13i3j2I * dm.y;
    x10.y += Q13i3j2R * dp.y + Q13i3j2I * dm.x;
    x4.x += Q13i4j2R * dp.x + Q13i4j2I * dm.y;
    x4.y += Q13i4j2R * dp.y - Q13i4j2I * dm.x;
    x9.x += Q13i4j2R * dp.x - Q13i4j2I * dm.y;
    x9.y += Q13i4j2R * dp.y + Q13i4j2I * dm.x;
    x5.x += Q13i5j2R * dp.x + Q13i5j2I * dm.y;
    x5.y += Q13i5j2R * dp.y - Q13i5j2I * dm.x;
    x8.x += Q13i5j2R * dp.x - Q13i5j2I * dm.y;
    x8.y += Q13i5j2R * dp.y + Q13i5j2I * dm.x;
    x6.x += Q13i6j2R * dp.x + Q13i6j2I * dm.y;
    x6.y += Q13i6j2R * dp.y - Q13i6j2I * dm.x;
    x7.x += Q13i6j2R * dp.x - Q13i6j2I * dm.y;
    x7.y += Q13i6j2R * dp.y + Q13i6j2I * dm.x;
    dp = (*R3) + (*R10);
    dm = (*R3) - (*R10);
    x1.x += Q13i1j3R * dp.x + Q13i1j3I * dm.y;
    x1.y += Q13i1j3R * dp.y - Q13i1j3I * dm.x;
    x12.x += Q13i1j3R * dp.x - Q13i1j3I * dm.y;
    x12.y += Q13i1j3R * dp.y + Q13i1j3I * dm.x;
    x2.x += Q13i2j3R * dp.x + Q13i2j3I * dm.y;
    x2.y += Q13i2j3R * dp.y - Q13i2j3I * dm.x;
    x11.x += Q13i2j3R * dp.x - Q13i2j3I * dm.y;
    x11.y += Q13i2j3R * dp.y + Q13i2j3I * dm.x;
    x3.x += Q13i3j3R * dp.x + Q13i3j3I * dm.y;
    x3.y += Q13i3j3R * dp.y - Q13i3j3I * dm.x;
    x10.x += Q13i3j3R * dp.x - Q13i3j3I * dm.y;
    x10.y += Q13i3j3R * dp.y + Q13i3j3I * dm.x;
    x4.x += Q13i4j3R * dp.x + Q13i4j3I * dm.y;
    x4.y += Q13i4j3R * dp.y - Q13i4j3I * dm.x;
    x9.x += Q13i4j3R * dp.x - Q13i4j3I * dm.y;
    x9.y += Q13i4j3R * dp.y + Q13i4j3I * dm.x;
    x5.x += Q13i5j3R * dp.x + Q13i5j3I * dm.y;
    x5.y += Q13i5j3R * dp.y - Q13i5j3I * dm.x;
    x8.x += Q13i5j3R * dp.x - Q13i5j3I * dm.y;
    x8.y += Q13i5j3R * dp.y + Q13i5j3I * dm.x;
    x6.x += Q13i6j3R * dp.x + Q13i6j3I * dm.y;
    x6.y += Q13i6j3R * dp.y - Q13i6j3I * dm.x;
    x7.x += Q13i6j3R * dp.x - Q13i6j3I * dm.y;
    x7.y += Q13i6j3R * dp.y + Q13i6j3I * dm.x;
    dp = (*R4) + (*R9);
    dm = (*R4) - (*R9);
    x1.x += Q13i1j4R * dp.x + Q13i1j4I * dm.y;
    x1.y += Q13i1j4R * dp.y - Q13i1j4I * dm.x;
    x12.x += Q13i1j4R * dp.x - Q13i1j4I * dm.y;
    x12.y += Q13i1j4R * dp.y + Q13i1j4I * dm.x;
    x2.x += Q13i2j4R * dp.x + Q13i2j4I * dm.y;
    x2.y += Q13i2j4R * dp.y - Q13i2j4I * dm.x;
    x11.x += Q13i2j4R * dp.x - Q13i2j4I * dm.y;
    x11.y += Q13i2j4R * dp.y + Q13i2j4I * dm.x;
    x3.x += Q13i3j4R * dp.x + Q13i3j4I * dm.y;
    x3.y += Q13i3j4R * dp.y - Q13i3j4I * dm.x;
    x10.x += Q13i3j4R * dp.x - Q13i3j4I * dm.y;
    x10.y += Q13i3j4R * dp.y + Q13i3j4I * dm.x;
    x4.x += Q13i4j4R * dp.x + Q13i4j4I * dm.y;
    x4.y += Q13i4j4R * dp.y - Q13i4j4I * dm.x;
    x9.x += Q13i4j4R * dp.x - Q13i4j4I * dm.y;
    x9.y += Q13i4j4R * dp.y + Q13i4j4I * dm.x;
    x5.x += Q13i5j4R * dp.x + Q13i5j4I * dm.y;
    x5.y += Q13i5j4R * dp.y - Q13i5j4I * dm.x;
    x8.x += Q13i5j4R * dp.x - Q13i5j4I * dm.y;
    x8.y += Q13i5j4R * dp.y + Q13i5j4I * dm.x;
    x6.x += Q13i6j4R * dp.x + Q13i6j4I * dm.y;
    x6.y += Q13i6j4R * dp.y - Q13i6j4I * dm.x;
    x7.x += Q13i6j4R * dp.x - Q13i6j4I * dm.y;
    x7.y += Q13i6j4R * dp.y + Q13i6j4I * dm.x;
    dp = (*R5) + (*R8);
    dm = (*R5) - (*R8);
    x1.x += Q13i1j5R * dp.x + Q13i1j5I * dm.y;
    x1.y += Q13i1j5R * dp.y - Q13i1j5I * dm.x;
    x12.x += Q13i1j5R * dp.x - Q13i1j5I * dm.y;
    x12.y += Q13i1j5R * dp.y + Q13i1j5I * dm.x;
    x2.x += Q13i2j5R * dp.x + Q13i2j5I * dm.y;
    x2.y += Q13i2j5R * dp.y - Q13i2j5I * dm.x;
    x11.x += Q13i2j5R * dp.x - Q13i2j5I * dm.y;
    x11.y += Q13i2j5R * dp.y + Q13i2j5I * dm.x;
    x3.x += Q13i3j5R * dp.x + Q13i3j5I * dm.y;
    x3.y += Q13i3j5R * dp.y - Q13i3j5I * dm.x;
    x10.x += Q13i3j5R * dp.x - Q13i3j5I * dm.y;
    x10.y += Q13i3j5R * dp.y + Q13i3j5I * dm.x;
    x4.x += Q13i4j5R * dp.x + Q13i4j5I * dm.y;
    x4.y += Q13i4j5R * dp.y - Q13i4j5I * dm.x;
    x9.x += Q13i4j5R * dp.x - Q13i4j5I * dm.y;
    x9.y += Q13i4j5R * dp.y + Q13i4j5I * dm.x;
    x5.x += Q13i5j5R * dp.x + Q13i5j5I * dm.y;
    x5.y += Q13i5j5R * dp.y - Q13i5j5I * dm.x;
    x8.x += Q13i5j5R * dp.x - Q13i5j5I * dm.y;
    x8.y += Q13i5j5R * dp.y + Q13i5j5I * dm.x;
    x6.x += Q13i6j5R * dp.x + Q13i6j5I * dm.y;
    x6.y += Q13i6j5R * dp.y - Q13i6j5I * dm.x;
    x7.x += Q13i6j5R * dp.x - Q13i6j5I * dm.y;
    x7.y += Q13i6j5R * dp.y + Q13i6j5I * dm.x;
    dp = (*R6) + (*R7);
    dm = (*R6) - (*R7);
    x1.x += Q13i1j6R * dp.x + Q13i1j6I * dm.y;
    x1.y += Q13i1j6R * dp.y - Q13i1j6I * dm.x;
    x12.x += Q13i1j6R * dp.x - Q13i1j6I * dm.y;
    x12.y += Q13i1j6R * dp.y + Q13i1j6I * dm.x;
    x2.x += Q13i2j6R * dp.x + Q13i2j6I * dm.y;
    x2.y += Q13i2j6R * dp.y - Q13i2j6I * dm.x;
    x11.x += Q13i2j6R * dp.x - Q13i2j6I * dm.y;
    x11.y += Q13i2j6R * dp.y + Q13i2j6I * dm.x;
    x3.x += Q13i3j6R * dp.x + Q13i3j6I * dm.y;
    x3.y += Q13i3j6R * dp.y - Q13i3j6I * dm.x;
    x10.x += Q13i3j6R * dp.x - Q13i3j6I * dm.y;
    x10.y += Q13i3j6R * dp.y + Q13i3j6I * dm.x;
    x4.x += Q13i4j6R * dp.x + Q13i4j6I * dm.y;
    x4.y += Q13i4j6R * dp.y - Q13i4j6I * dm.x;
    x9.x += Q13i4j6R * dp.x - Q13i4j6I * dm.y;
    x9.y += Q13i4j6R * dp.y + Q13i4j6I * dm.x;
    x5.x += Q13i5j6R * dp.x + Q13i5j6I * dm.y;
    x5.y += Q13i5j6R * dp.y - Q13i5j6I * dm.x;
    x8.x += Q13i5j6R * dp.x - Q13i5j6I * dm.y;
    x8.y += Q13i5j6R * dp.y + Q13i5j6I * dm.x;
    x6.x += Q13i6j6R * dp.x + Q13i6j6I * dm.y;
    x6.y += Q13i6j6R * dp.y - Q13i6j6I * dm.x;
    x7.x += Q13i6j6R * dp.x - Q13i6j6I * dm.y;
    x7.y += Q13i6j6R * dp.y + Q13i6j6I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
}

template <typename T>
__device__ void FwdRad17B1(T* R0,
                           T* R1,
                           T* R2,
                           T* R3,
                           T* R4,
                           T* R5,
                           T* R6,
                           T* R7,
                           T* R8,
                           T* R9,
                           T* R10,
                           T* R11,
                           T* R12,
                           T* R13,
                           T* R14,
                           T* R15,
                           T* R16)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12) + (*R13) + (*R14) + (*R15) + (*R16);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    x13 = (*R0);
    x14 = (*R0);
    x15 = (*R0);
    x16 = (*R0);
    dp  = (*R1) + (*R16);
    dm  = (*R1) - (*R16);
    x1.x += Q17i1j1R * dp.x - Q17i1j1I * dm.y;
    x1.y += Q17i1j1R * dp.y + Q17i1j1I * dm.x;
    x16.x += Q17i1j1R * dp.x + Q17i1j1I * dm.y;
    x16.y += Q17i1j1R * dp.y - Q17i1j1I * dm.x;
    x2.x += Q17i2j1R * dp.x - Q17i2j1I * dm.y;
    x2.y += Q17i2j1R * dp.y + Q17i2j1I * dm.x;
    x15.x += Q17i2j1R * dp.x + Q17i2j1I * dm.y;
    x15.y += Q17i2j1R * dp.y - Q17i2j1I * dm.x;
    x3.x += Q17i3j1R * dp.x - Q17i3j1I * dm.y;
    x3.y += Q17i3j1R * dp.y + Q17i3j1I * dm.x;
    x14.x += Q17i3j1R * dp.x + Q17i3j1I * dm.y;
    x14.y += Q17i3j1R * dp.y - Q17i3j1I * dm.x;
    x4.x += Q17i4j1R * dp.x - Q17i4j1I * dm.y;
    x4.y += Q17i4j1R * dp.y + Q17i4j1I * dm.x;
    x13.x += Q17i4j1R * dp.x + Q17i4j1I * dm.y;
    x13.y += Q17i4j1R * dp.y - Q17i4j1I * dm.x;
    x5.x += Q17i5j1R * dp.x - Q17i5j1I * dm.y;
    x5.y += Q17i5j1R * dp.y + Q17i5j1I * dm.x;
    x12.x += Q17i5j1R * dp.x + Q17i5j1I * dm.y;
    x12.y += Q17i5j1R * dp.y - Q17i5j1I * dm.x;
    x6.x += Q17i6j1R * dp.x - Q17i6j1I * dm.y;
    x6.y += Q17i6j1R * dp.y + Q17i6j1I * dm.x;
    x11.x += Q17i6j1R * dp.x + Q17i6j1I * dm.y;
    x11.y += Q17i6j1R * dp.y - Q17i6j1I * dm.x;
    x7.x += Q17i7j1R * dp.x - Q17i7j1I * dm.y;
    x7.y += Q17i7j1R * dp.y + Q17i7j1I * dm.x;
    x10.x += Q17i7j1R * dp.x + Q17i7j1I * dm.y;
    x10.y += Q17i7j1R * dp.y - Q17i7j1I * dm.x;
    x8.x += Q17i8j1R * dp.x - Q17i8j1I * dm.y;
    x8.y += Q17i8j1R * dp.y + Q17i8j1I * dm.x;
    x9.x += Q17i8j1R * dp.x + Q17i8j1I * dm.y;
    x9.y += Q17i8j1R * dp.y - Q17i8j1I * dm.x;
    dp = (*R2) + (*R15);
    dm = (*R2) - (*R15);
    x1.x += Q17i1j2R * dp.x - Q17i1j2I * dm.y;
    x1.y += Q17i1j2R * dp.y + Q17i1j2I * dm.x;
    x16.x += Q17i1j2R * dp.x + Q17i1j2I * dm.y;
    x16.y += Q17i1j2R * dp.y - Q17i1j2I * dm.x;
    x2.x += Q17i2j2R * dp.x - Q17i2j2I * dm.y;
    x2.y += Q17i2j2R * dp.y + Q17i2j2I * dm.x;
    x15.x += Q17i2j2R * dp.x + Q17i2j2I * dm.y;
    x15.y += Q17i2j2R * dp.y - Q17i2j2I * dm.x;
    x3.x += Q17i3j2R * dp.x - Q17i3j2I * dm.y;
    x3.y += Q17i3j2R * dp.y + Q17i3j2I * dm.x;
    x14.x += Q17i3j2R * dp.x + Q17i3j2I * dm.y;
    x14.y += Q17i3j2R * dp.y - Q17i3j2I * dm.x;
    x4.x += Q17i4j2R * dp.x - Q17i4j2I * dm.y;
    x4.y += Q17i4j2R * dp.y + Q17i4j2I * dm.x;
    x13.x += Q17i4j2R * dp.x + Q17i4j2I * dm.y;
    x13.y += Q17i4j2R * dp.y - Q17i4j2I * dm.x;
    x5.x += Q17i5j2R * dp.x - Q17i5j2I * dm.y;
    x5.y += Q17i5j2R * dp.y + Q17i5j2I * dm.x;
    x12.x += Q17i5j2R * dp.x + Q17i5j2I * dm.y;
    x12.y += Q17i5j2R * dp.y - Q17i5j2I * dm.x;
    x6.x += Q17i6j2R * dp.x - Q17i6j2I * dm.y;
    x6.y += Q17i6j2R * dp.y + Q17i6j2I * dm.x;
    x11.x += Q17i6j2R * dp.x + Q17i6j2I * dm.y;
    x11.y += Q17i6j2R * dp.y - Q17i6j2I * dm.x;
    x7.x += Q17i7j2R * dp.x - Q17i7j2I * dm.y;
    x7.y += Q17i7j2R * dp.y + Q17i7j2I * dm.x;
    x10.x += Q17i7j2R * dp.x + Q17i7j2I * dm.y;
    x10.y += Q17i7j2R * dp.y - Q17i7j2I * dm.x;
    x8.x += Q17i8j2R * dp.x - Q17i8j2I * dm.y;
    x8.y += Q17i8j2R * dp.y + Q17i8j2I * dm.x;
    x9.x += Q17i8j2R * dp.x + Q17i8j2I * dm.y;
    x9.y += Q17i8j2R * dp.y - Q17i8j2I * dm.x;
    dp = (*R3) + (*R14);
    dm = (*R3) - (*R14);
    x1.x += Q17i1j3R * dp.x - Q17i1j3I * dm.y;
    x1.y += Q17i1j3R * dp.y + Q17i1j3I * dm.x;
    x16.x += Q17i1j3R * dp.x + Q17i1j3I * dm.y;
    x16.y += Q17i1j3R * dp.y - Q17i1j3I * dm.x;
    x2.x += Q17i2j3R * dp.x - Q17i2j3I * dm.y;
    x2.y += Q17i2j3R * dp.y + Q17i2j3I * dm.x;
    x15.x += Q17i2j3R * dp.x + Q17i2j3I * dm.y;
    x15.y += Q17i2j3R * dp.y - Q17i2j3I * dm.x;
    x3.x += Q17i3j3R * dp.x - Q17i3j3I * dm.y;
    x3.y += Q17i3j3R * dp.y + Q17i3j3I * dm.x;
    x14.x += Q17i3j3R * dp.x + Q17i3j3I * dm.y;
    x14.y += Q17i3j3R * dp.y - Q17i3j3I * dm.x;
    x4.x += Q17i4j3R * dp.x - Q17i4j3I * dm.y;
    x4.y += Q17i4j3R * dp.y + Q17i4j3I * dm.x;
    x13.x += Q17i4j3R * dp.x + Q17i4j3I * dm.y;
    x13.y += Q17i4j3R * dp.y - Q17i4j3I * dm.x;
    x5.x += Q17i5j3R * dp.x - Q17i5j3I * dm.y;
    x5.y += Q17i5j3R * dp.y + Q17i5j3I * dm.x;
    x12.x += Q17i5j3R * dp.x + Q17i5j3I * dm.y;
    x12.y += Q17i5j3R * dp.y - Q17i5j3I * dm.x;
    x6.x += Q17i6j3R * dp.x - Q17i6j3I * dm.y;
    x6.y += Q17i6j3R * dp.y + Q17i6j3I * dm.x;
    x11.x += Q17i6j3R * dp.x + Q17i6j3I * dm.y;
    x11.y += Q17i6j3R * dp.y - Q17i6j3I * dm.x;
    x7.x += Q17i7j3R * dp.x - Q17i7j3I * dm.y;
    x7.y += Q17i7j3R * dp.y + Q17i7j3I * dm.x;
    x10.x += Q17i7j3R * dp.x + Q17i7j3I * dm.y;
    x10.y += Q17i7j3R * dp.y - Q17i7j3I * dm.x;
    x8.x += Q17i8j3R * dp.x - Q17i8j3I * dm.y;
    x8.y += Q17i8j3R * dp.y + Q17i8j3I * dm.x;
    x9.x += Q17i8j3R * dp.x + Q17i8j3I * dm.y;
    x9.y += Q17i8j3R * dp.y - Q17i8j3I * dm.x;
    dp = (*R4) + (*R13);
    dm = (*R4) - (*R13);
    x1.x += Q17i1j4R * dp.x - Q17i1j4I * dm.y;
    x1.y += Q17i1j4R * dp.y + Q17i1j4I * dm.x;
    x16.x += Q17i1j4R * dp.x + Q17i1j4I * dm.y;
    x16.y += Q17i1j4R * dp.y - Q17i1j4I * dm.x;
    x2.x += Q17i2j4R * dp.x - Q17i2j4I * dm.y;
    x2.y += Q17i2j4R * dp.y + Q17i2j4I * dm.x;
    x15.x += Q17i2j4R * dp.x + Q17i2j4I * dm.y;
    x15.y += Q17i2j4R * dp.y - Q17i2j4I * dm.x;
    x3.x += Q17i3j4R * dp.x - Q17i3j4I * dm.y;
    x3.y += Q17i3j4R * dp.y + Q17i3j4I * dm.x;
    x14.x += Q17i3j4R * dp.x + Q17i3j4I * dm.y;
    x14.y += Q17i3j4R * dp.y - Q17i3j4I * dm.x;
    x4.x += Q17i4j4R * dp.x - Q17i4j4I * dm.y;
    x4.y += Q17i4j4R * dp.y + Q17i4j4I * dm.x;
    x13.x += Q17i4j4R * dp.x + Q17i4j4I * dm.y;
    x13.y += Q17i4j4R * dp.y - Q17i4j4I * dm.x;
    x5.x += Q17i5j4R * dp.x - Q17i5j4I * dm.y;
    x5.y += Q17i5j4R * dp.y + Q17i5j4I * dm.x;
    x12.x += Q17i5j4R * dp.x + Q17i5j4I * dm.y;
    x12.y += Q17i5j4R * dp.y - Q17i5j4I * dm.x;
    x6.x += Q17i6j4R * dp.x - Q17i6j4I * dm.y;
    x6.y += Q17i6j4R * dp.y + Q17i6j4I * dm.x;
    x11.x += Q17i6j4R * dp.x + Q17i6j4I * dm.y;
    x11.y += Q17i6j4R * dp.y - Q17i6j4I * dm.x;
    x7.x += Q17i7j4R * dp.x - Q17i7j4I * dm.y;
    x7.y += Q17i7j4R * dp.y + Q17i7j4I * dm.x;
    x10.x += Q17i7j4R * dp.x + Q17i7j4I * dm.y;
    x10.y += Q17i7j4R * dp.y - Q17i7j4I * dm.x;
    x8.x += Q17i8j4R * dp.x - Q17i8j4I * dm.y;
    x8.y += Q17i8j4R * dp.y + Q17i8j4I * dm.x;
    x9.x += Q17i8j4R * dp.x + Q17i8j4I * dm.y;
    x9.y += Q17i8j4R * dp.y - Q17i8j4I * dm.x;
    dp = (*R5) + (*R12);
    dm = (*R5) - (*R12);
    x1.x += Q17i1j5R * dp.x - Q17i1j5I * dm.y;
    x1.y += Q17i1j5R * dp.y + Q17i1j5I * dm.x;
    x16.x += Q17i1j5R * dp.x + Q17i1j5I * dm.y;
    x16.y += Q17i1j5R * dp.y - Q17i1j5I * dm.x;
    x2.x += Q17i2j5R * dp.x - Q17i2j5I * dm.y;
    x2.y += Q17i2j5R * dp.y + Q17i2j5I * dm.x;
    x15.x += Q17i2j5R * dp.x + Q17i2j5I * dm.y;
    x15.y += Q17i2j5R * dp.y - Q17i2j5I * dm.x;
    x3.x += Q17i3j5R * dp.x - Q17i3j5I * dm.y;
    x3.y += Q17i3j5R * dp.y + Q17i3j5I * dm.x;
    x14.x += Q17i3j5R * dp.x + Q17i3j5I * dm.y;
    x14.y += Q17i3j5R * dp.y - Q17i3j5I * dm.x;
    x4.x += Q17i4j5R * dp.x - Q17i4j5I * dm.y;
    x4.y += Q17i4j5R * dp.y + Q17i4j5I * dm.x;
    x13.x += Q17i4j5R * dp.x + Q17i4j5I * dm.y;
    x13.y += Q17i4j5R * dp.y - Q17i4j5I * dm.x;
    x5.x += Q17i5j5R * dp.x - Q17i5j5I * dm.y;
    x5.y += Q17i5j5R * dp.y + Q17i5j5I * dm.x;
    x12.x += Q17i5j5R * dp.x + Q17i5j5I * dm.y;
    x12.y += Q17i5j5R * dp.y - Q17i5j5I * dm.x;
    x6.x += Q17i6j5R * dp.x - Q17i6j5I * dm.y;
    x6.y += Q17i6j5R * dp.y + Q17i6j5I * dm.x;
    x11.x += Q17i6j5R * dp.x + Q17i6j5I * dm.y;
    x11.y += Q17i6j5R * dp.y - Q17i6j5I * dm.x;
    x7.x += Q17i7j5R * dp.x - Q17i7j5I * dm.y;
    x7.y += Q17i7j5R * dp.y + Q17i7j5I * dm.x;
    x10.x += Q17i7j5R * dp.x + Q17i7j5I * dm.y;
    x10.y += Q17i7j5R * dp.y - Q17i7j5I * dm.x;
    x8.x += Q17i8j5R * dp.x - Q17i8j5I * dm.y;
    x8.y += Q17i8j5R * dp.y + Q17i8j5I * dm.x;
    x9.x += Q17i8j5R * dp.x + Q17i8j5I * dm.y;
    x9.y += Q17i8j5R * dp.y - Q17i8j5I * dm.x;
    dp = (*R6) + (*R11);
    dm = (*R6) - (*R11);
    x1.x += Q17i1j6R * dp.x - Q17i1j6I * dm.y;
    x1.y += Q17i1j6R * dp.y + Q17i1j6I * dm.x;
    x16.x += Q17i1j6R * dp.x + Q17i1j6I * dm.y;
    x16.y += Q17i1j6R * dp.y - Q17i1j6I * dm.x;
    x2.x += Q17i2j6R * dp.x - Q17i2j6I * dm.y;
    x2.y += Q17i2j6R * dp.y + Q17i2j6I * dm.x;
    x15.x += Q17i2j6R * dp.x + Q17i2j6I * dm.y;
    x15.y += Q17i2j6R * dp.y - Q17i2j6I * dm.x;
    x3.x += Q17i3j6R * dp.x - Q17i3j6I * dm.y;
    x3.y += Q17i3j6R * dp.y + Q17i3j6I * dm.x;
    x14.x += Q17i3j6R * dp.x + Q17i3j6I * dm.y;
    x14.y += Q17i3j6R * dp.y - Q17i3j6I * dm.x;
    x4.x += Q17i4j6R * dp.x - Q17i4j6I * dm.y;
    x4.y += Q17i4j6R * dp.y + Q17i4j6I * dm.x;
    x13.x += Q17i4j6R * dp.x + Q17i4j6I * dm.y;
    x13.y += Q17i4j6R * dp.y - Q17i4j6I * dm.x;
    x5.x += Q17i5j6R * dp.x - Q17i5j6I * dm.y;
    x5.y += Q17i5j6R * dp.y + Q17i5j6I * dm.x;
    x12.x += Q17i5j6R * dp.x + Q17i5j6I * dm.y;
    x12.y += Q17i5j6R * dp.y - Q17i5j6I * dm.x;
    x6.x += Q17i6j6R * dp.x - Q17i6j6I * dm.y;
    x6.y += Q17i6j6R * dp.y + Q17i6j6I * dm.x;
    x11.x += Q17i6j6R * dp.x + Q17i6j6I * dm.y;
    x11.y += Q17i6j6R * dp.y - Q17i6j6I * dm.x;
    x7.x += Q17i7j6R * dp.x - Q17i7j6I * dm.y;
    x7.y += Q17i7j6R * dp.y + Q17i7j6I * dm.x;
    x10.x += Q17i7j6R * dp.x + Q17i7j6I * dm.y;
    x10.y += Q17i7j6R * dp.y - Q17i7j6I * dm.x;
    x8.x += Q17i8j6R * dp.x - Q17i8j6I * dm.y;
    x8.y += Q17i8j6R * dp.y + Q17i8j6I * dm.x;
    x9.x += Q17i8j6R * dp.x + Q17i8j6I * dm.y;
    x9.y += Q17i8j6R * dp.y - Q17i8j6I * dm.x;
    dp = (*R7) + (*R10);
    dm = (*R7) - (*R10);
    x1.x += Q17i1j7R * dp.x - Q17i1j7I * dm.y;
    x1.y += Q17i1j7R * dp.y + Q17i1j7I * dm.x;
    x16.x += Q17i1j7R * dp.x + Q17i1j7I * dm.y;
    x16.y += Q17i1j7R * dp.y - Q17i1j7I * dm.x;
    x2.x += Q17i2j7R * dp.x - Q17i2j7I * dm.y;
    x2.y += Q17i2j7R * dp.y + Q17i2j7I * dm.x;
    x15.x += Q17i2j7R * dp.x + Q17i2j7I * dm.y;
    x15.y += Q17i2j7R * dp.y - Q17i2j7I * dm.x;
    x3.x += Q17i3j7R * dp.x - Q17i3j7I * dm.y;
    x3.y += Q17i3j7R * dp.y + Q17i3j7I * dm.x;
    x14.x += Q17i3j7R * dp.x + Q17i3j7I * dm.y;
    x14.y += Q17i3j7R * dp.y - Q17i3j7I * dm.x;
    x4.x += Q17i4j7R * dp.x - Q17i4j7I * dm.y;
    x4.y += Q17i4j7R * dp.y + Q17i4j7I * dm.x;
    x13.x += Q17i4j7R * dp.x + Q17i4j7I * dm.y;
    x13.y += Q17i4j7R * dp.y - Q17i4j7I * dm.x;
    x5.x += Q17i5j7R * dp.x - Q17i5j7I * dm.y;
    x5.y += Q17i5j7R * dp.y + Q17i5j7I * dm.x;
    x12.x += Q17i5j7R * dp.x + Q17i5j7I * dm.y;
    x12.y += Q17i5j7R * dp.y - Q17i5j7I * dm.x;
    x6.x += Q17i6j7R * dp.x - Q17i6j7I * dm.y;
    x6.y += Q17i6j7R * dp.y + Q17i6j7I * dm.x;
    x11.x += Q17i6j7R * dp.x + Q17i6j7I * dm.y;
    x11.y += Q17i6j7R * dp.y - Q17i6j7I * dm.x;
    x7.x += Q17i7j7R * dp.x - Q17i7j7I * dm.y;
    x7.y += Q17i7j7R * dp.y + Q17i7j7I * dm.x;
    x10.x += Q17i7j7R * dp.x + Q17i7j7I * dm.y;
    x10.y += Q17i7j7R * dp.y - Q17i7j7I * dm.x;
    x8.x += Q17i8j7R * dp.x - Q17i8j7I * dm.y;
    x8.y += Q17i8j7R * dp.y + Q17i8j7I * dm.x;
    x9.x += Q17i8j7R * dp.x + Q17i8j7I * dm.y;
    x9.y += Q17i8j7R * dp.y - Q17i8j7I * dm.x;
    dp = (*R8) + (*R9);
    dm = (*R8) - (*R9);
    x1.x += Q17i1j8R * dp.x - Q17i1j8I * dm.y;
    x1.y += Q17i1j8R * dp.y + Q17i1j8I * dm.x;
    x16.x += Q17i1j8R * dp.x + Q17i1j8I * dm.y;
    x16.y += Q17i1j8R * dp.y - Q17i1j8I * dm.x;
    x2.x += Q17i2j8R * dp.x - Q17i2j8I * dm.y;
    x2.y += Q17i2j8R * dp.y + Q17i2j8I * dm.x;
    x15.x += Q17i2j8R * dp.x + Q17i2j8I * dm.y;
    x15.y += Q17i2j8R * dp.y - Q17i2j8I * dm.x;
    x3.x += Q17i3j8R * dp.x - Q17i3j8I * dm.y;
    x3.y += Q17i3j8R * dp.y + Q17i3j8I * dm.x;
    x14.x += Q17i3j8R * dp.x + Q17i3j8I * dm.y;
    x14.y += Q17i3j8R * dp.y - Q17i3j8I * dm.x;
    x4.x += Q17i4j8R * dp.x - Q17i4j8I * dm.y;
    x4.y += Q17i4j8R * dp.y + Q17i4j8I * dm.x;
    x13.x += Q17i4j8R * dp.x + Q17i4j8I * dm.y;
    x13.y += Q17i4j8R * dp.y - Q17i4j8I * dm.x;
    x5.x += Q17i5j8R * dp.x - Q17i5j8I * dm.y;
    x5.y += Q17i5j8R * dp.y + Q17i5j8I * dm.x;
    x12.x += Q17i5j8R * dp.x + Q17i5j8I * dm.y;
    x12.y += Q17i5j8R * dp.y - Q17i5j8I * dm.x;
    x6.x += Q17i6j8R * dp.x - Q17i6j8I * dm.y;
    x6.y += Q17i6j8R * dp.y + Q17i6j8I * dm.x;
    x11.x += Q17i6j8R * dp.x + Q17i6j8I * dm.y;
    x11.y += Q17i6j8R * dp.y - Q17i6j8I * dm.x;
    x7.x += Q17i7j8R * dp.x - Q17i7j8I * dm.y;
    x7.y += Q17i7j8R * dp.y + Q17i7j8I * dm.x;
    x10.x += Q17i7j8R * dp.x + Q17i7j8I * dm.y;
    x10.y += Q17i7j8R * dp.y - Q17i7j8I * dm.x;
    x8.x += Q17i8j8R * dp.x - Q17i8j8I * dm.y;
    x8.y += Q17i8j8R * dp.y + Q17i8j8I * dm.x;
    x9.x += Q17i8j8R * dp.x + Q17i8j8I * dm.y;
    x9.y += Q17i8j8R * dp.y - Q17i8j8I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
    (*R13) = x13;
    (*R14) = x14;
    (*R15) = x15;
    (*R16) = x16;
}

template <typename T>
__device__ void InvRad17B1(T* R0,
                           T* R1,
                           T* R2,
                           T* R3,
                           T* R4,
                           T* R5,
                           T* R6,
                           T* R7,
                           T* R8,
                           T* R9,
                           T* R10,
                           T* R11,
                           T* R12,
                           T* R13,
                           T* R14,
                           T* R15,
                           T* R16)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12) + (*R13) + (*R14) + (*R15) + (*R16);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    x13 = (*R0);
    x14 = (*R0);
    x15 = (*R0);
    x16 = (*R0);
    dp  = (*R1) + (*R16);
    dm  = (*R1) - (*R16);
    x1.x += Q17i1j1R * dp.x + Q17i1j1I * dm.y;
    x1.y += Q17i1j1R * dp.y - Q17i1j1I * dm.x;
    x16.x += Q17i1j1R * dp.x - Q17i1j1I * dm.y;
    x16.y += Q17i1j1R * dp.y + Q17i1j1I * dm.x;
    x2.x += Q17i2j1R * dp.x + Q17i2j1I * dm.y;
    x2.y += Q17i2j1R * dp.y - Q17i2j1I * dm.x;
    x15.x += Q17i2j1R * dp.x - Q17i2j1I * dm.y;
    x15.y += Q17i2j1R * dp.y + Q17i2j1I * dm.x;
    x3.x += Q17i3j1R * dp.x + Q17i3j1I * dm.y;
    x3.y += Q17i3j1R * dp.y - Q17i3j1I * dm.x;
    x14.x += Q17i3j1R * dp.x - Q17i3j1I * dm.y;
    x14.y += Q17i3j1R * dp.y + Q17i3j1I * dm.x;
    x4.x += Q17i4j1R * dp.x + Q17i4j1I * dm.y;
    x4.y += Q17i4j1R * dp.y - Q17i4j1I * dm.x;
    x13.x += Q17i4j1R * dp.x - Q17i4j1I * dm.y;
    x13.y += Q17i4j1R * dp.y + Q17i4j1I * dm.x;
    x5.x += Q17i5j1R * dp.x + Q17i5j1I * dm.y;
    x5.y += Q17i5j1R * dp.y - Q17i5j1I * dm.x;
    x12.x += Q17i5j1R * dp.x - Q17i5j1I * dm.y;
    x12.y += Q17i5j1R * dp.y + Q17i5j1I * dm.x;
    x6.x += Q17i6j1R * dp.x + Q17i6j1I * dm.y;
    x6.y += Q17i6j1R * dp.y - Q17i6j1I * dm.x;
    x11.x += Q17i6j1R * dp.x - Q17i6j1I * dm.y;
    x11.y += Q17i6j1R * dp.y + Q17i6j1I * dm.x;
    x7.x += Q17i7j1R * dp.x + Q17i7j1I * dm.y;
    x7.y += Q17i7j1R * dp.y - Q17i7j1I * dm.x;
    x10.x += Q17i7j1R * dp.x - Q17i7j1I * dm.y;
    x10.y += Q17i7j1R * dp.y + Q17i7j1I * dm.x;
    x8.x += Q17i8j1R * dp.x + Q17i8j1I * dm.y;
    x8.y += Q17i8j1R * dp.y - Q17i8j1I * dm.x;
    x9.x += Q17i8j1R * dp.x - Q17i8j1I * dm.y;
    x9.y += Q17i8j1R * dp.y + Q17i8j1I * dm.x;
    dp = (*R2) + (*R15);
    dm = (*R2) - (*R15);
    x1.x += Q17i1j2R * dp.x + Q17i1j2I * dm.y;
    x1.y += Q17i1j2R * dp.y - Q17i1j2I * dm.x;
    x16.x += Q17i1j2R * dp.x - Q17i1j2I * dm.y;
    x16.y += Q17i1j2R * dp.y + Q17i1j2I * dm.x;
    x2.x += Q17i2j2R * dp.x + Q17i2j2I * dm.y;
    x2.y += Q17i2j2R * dp.y - Q17i2j2I * dm.x;
    x15.x += Q17i2j2R * dp.x - Q17i2j2I * dm.y;
    x15.y += Q17i2j2R * dp.y + Q17i2j2I * dm.x;
    x3.x += Q17i3j2R * dp.x + Q17i3j2I * dm.y;
    x3.y += Q17i3j2R * dp.y - Q17i3j2I * dm.x;
    x14.x += Q17i3j2R * dp.x - Q17i3j2I * dm.y;
    x14.y += Q17i3j2R * dp.y + Q17i3j2I * dm.x;
    x4.x += Q17i4j2R * dp.x + Q17i4j2I * dm.y;
    x4.y += Q17i4j2R * dp.y - Q17i4j2I * dm.x;
    x13.x += Q17i4j2R * dp.x - Q17i4j2I * dm.y;
    x13.y += Q17i4j2R * dp.y + Q17i4j2I * dm.x;
    x5.x += Q17i5j2R * dp.x + Q17i5j2I * dm.y;
    x5.y += Q17i5j2R * dp.y - Q17i5j2I * dm.x;
    x12.x += Q17i5j2R * dp.x - Q17i5j2I * dm.y;
    x12.y += Q17i5j2R * dp.y + Q17i5j2I * dm.x;
    x6.x += Q17i6j2R * dp.x + Q17i6j2I * dm.y;
    x6.y += Q17i6j2R * dp.y - Q17i6j2I * dm.x;
    x11.x += Q17i6j2R * dp.x - Q17i6j2I * dm.y;
    x11.y += Q17i6j2R * dp.y + Q17i6j2I * dm.x;
    x7.x += Q17i7j2R * dp.x + Q17i7j2I * dm.y;
    x7.y += Q17i7j2R * dp.y - Q17i7j2I * dm.x;
    x10.x += Q17i7j2R * dp.x - Q17i7j2I * dm.y;
    x10.y += Q17i7j2R * dp.y + Q17i7j2I * dm.x;
    x8.x += Q17i8j2R * dp.x + Q17i8j2I * dm.y;
    x8.y += Q17i8j2R * dp.y - Q17i8j2I * dm.x;
    x9.x += Q17i8j2R * dp.x - Q17i8j2I * dm.y;
    x9.y += Q17i8j2R * dp.y + Q17i8j2I * dm.x;
    dp = (*R3) + (*R14);
    dm = (*R3) - (*R14);
    x1.x += Q17i1j3R * dp.x + Q17i1j3I * dm.y;
    x1.y += Q17i1j3R * dp.y - Q17i1j3I * dm.x;
    x16.x += Q17i1j3R * dp.x - Q17i1j3I * dm.y;
    x16.y += Q17i1j3R * dp.y + Q17i1j3I * dm.x;
    x2.x += Q17i2j3R * dp.x + Q17i2j3I * dm.y;
    x2.y += Q17i2j3R * dp.y - Q17i2j3I * dm.x;
    x15.x += Q17i2j3R * dp.x - Q17i2j3I * dm.y;
    x15.y += Q17i2j3R * dp.y + Q17i2j3I * dm.x;
    x3.x += Q17i3j3R * dp.x + Q17i3j3I * dm.y;
    x3.y += Q17i3j3R * dp.y - Q17i3j3I * dm.x;
    x14.x += Q17i3j3R * dp.x - Q17i3j3I * dm.y;
    x14.y += Q17i3j3R * dp.y + Q17i3j3I * dm.x;
    x4.x += Q17i4j3R * dp.x + Q17i4j3I * dm.y;
    x4.y += Q17i4j3R * dp.y - Q17i4j3I * dm.x;
    x13.x += Q17i4j3R * dp.x - Q17i4j3I * dm.y;
    x13.y += Q17i4j3R * dp.y + Q17i4j3I * dm.x;
    x5.x += Q17i5j3R * dp.x + Q17i5j3I * dm.y;
    x5.y += Q17i5j3R * dp.y - Q17i5j3I * dm.x;
    x12.x += Q17i5j3R * dp.x - Q17i5j3I * dm.y;
    x12.y += Q17i5j3R * dp.y + Q17i5j3I * dm.x;
    x6.x += Q17i6j3R * dp.x + Q17i6j3I * dm.y;
    x6.y += Q17i6j3R * dp.y - Q17i6j3I * dm.x;
    x11.x += Q17i6j3R * dp.x - Q17i6j3I * dm.y;
    x11.y += Q17i6j3R * dp.y + Q17i6j3I * dm.x;
    x7.x += Q17i7j3R * dp.x + Q17i7j3I * dm.y;
    x7.y += Q17i7j3R * dp.y - Q17i7j3I * dm.x;
    x10.x += Q17i7j3R * dp.x - Q17i7j3I * dm.y;
    x10.y += Q17i7j3R * dp.y + Q17i7j3I * dm.x;
    x8.x += Q17i8j3R * dp.x + Q17i8j3I * dm.y;
    x8.y += Q17i8j3R * dp.y - Q17i8j3I * dm.x;
    x9.x += Q17i8j3R * dp.x - Q17i8j3I * dm.y;
    x9.y += Q17i8j3R * dp.y + Q17i8j3I * dm.x;
    dp = (*R4) + (*R13);
    dm = (*R4) - (*R13);
    x1.x += Q17i1j4R * dp.x + Q17i1j4I * dm.y;
    x1.y += Q17i1j4R * dp.y - Q17i1j4I * dm.x;
    x16.x += Q17i1j4R * dp.x - Q17i1j4I * dm.y;
    x16.y += Q17i1j4R * dp.y + Q17i1j4I * dm.x;
    x2.x += Q17i2j4R * dp.x + Q17i2j4I * dm.y;
    x2.y += Q17i2j4R * dp.y - Q17i2j4I * dm.x;
    x15.x += Q17i2j4R * dp.x - Q17i2j4I * dm.y;
    x15.y += Q17i2j4R * dp.y + Q17i2j4I * dm.x;
    x3.x += Q17i3j4R * dp.x + Q17i3j4I * dm.y;
    x3.y += Q17i3j4R * dp.y - Q17i3j4I * dm.x;
    x14.x += Q17i3j4R * dp.x - Q17i3j4I * dm.y;
    x14.y += Q17i3j4R * dp.y + Q17i3j4I * dm.x;
    x4.x += Q17i4j4R * dp.x + Q17i4j4I * dm.y;
    x4.y += Q17i4j4R * dp.y - Q17i4j4I * dm.x;
    x13.x += Q17i4j4R * dp.x - Q17i4j4I * dm.y;
    x13.y += Q17i4j4R * dp.y + Q17i4j4I * dm.x;
    x5.x += Q17i5j4R * dp.x + Q17i5j4I * dm.y;
    x5.y += Q17i5j4R * dp.y - Q17i5j4I * dm.x;
    x12.x += Q17i5j4R * dp.x - Q17i5j4I * dm.y;
    x12.y += Q17i5j4R * dp.y + Q17i5j4I * dm.x;
    x6.x += Q17i6j4R * dp.x + Q17i6j4I * dm.y;
    x6.y += Q17i6j4R * dp.y - Q17i6j4I * dm.x;
    x11.x += Q17i6j4R * dp.x - Q17i6j4I * dm.y;
    x11.y += Q17i6j4R * dp.y + Q17i6j4I * dm.x;
    x7.x += Q17i7j4R * dp.x + Q17i7j4I * dm.y;
    x7.y += Q17i7j4R * dp.y - Q17i7j4I * dm.x;
    x10.x += Q17i7j4R * dp.x - Q17i7j4I * dm.y;
    x10.y += Q17i7j4R * dp.y + Q17i7j4I * dm.x;
    x8.x += Q17i8j4R * dp.x + Q17i8j4I * dm.y;
    x8.y += Q17i8j4R * dp.y - Q17i8j4I * dm.x;
    x9.x += Q17i8j4R * dp.x - Q17i8j4I * dm.y;
    x9.y += Q17i8j4R * dp.y + Q17i8j4I * dm.x;
    dp = (*R5) + (*R12);
    dm = (*R5) - (*R12);
    x1.x += Q17i1j5R * dp.x + Q17i1j5I * dm.y;
    x1.y += Q17i1j5R * dp.y - Q17i1j5I * dm.x;
    x16.x += Q17i1j5R * dp.x - Q17i1j5I * dm.y;
    x16.y += Q17i1j5R * dp.y + Q17i1j5I * dm.x;
    x2.x += Q17i2j5R * dp.x + Q17i2j5I * dm.y;
    x2.y += Q17i2j5R * dp.y - Q17i2j5I * dm.x;
    x15.x += Q17i2j5R * dp.x - Q17i2j5I * dm.y;
    x15.y += Q17i2j5R * dp.y + Q17i2j5I * dm.x;
    x3.x += Q17i3j5R * dp.x + Q17i3j5I * dm.y;
    x3.y += Q17i3j5R * dp.y - Q17i3j5I * dm.x;
    x14.x += Q17i3j5R * dp.x - Q17i3j5I * dm.y;
    x14.y += Q17i3j5R * dp.y + Q17i3j5I * dm.x;
    x4.x += Q17i4j5R * dp.x + Q17i4j5I * dm.y;
    x4.y += Q17i4j5R * dp.y - Q17i4j5I * dm.x;
    x13.x += Q17i4j5R * dp.x - Q17i4j5I * dm.y;
    x13.y += Q17i4j5R * dp.y + Q17i4j5I * dm.x;
    x5.x += Q17i5j5R * dp.x + Q17i5j5I * dm.y;
    x5.y += Q17i5j5R * dp.y - Q17i5j5I * dm.x;
    x12.x += Q17i5j5R * dp.x - Q17i5j5I * dm.y;
    x12.y += Q17i5j5R * dp.y + Q17i5j5I * dm.x;
    x6.x += Q17i6j5R * dp.x + Q17i6j5I * dm.y;
    x6.y += Q17i6j5R * dp.y - Q17i6j5I * dm.x;
    x11.x += Q17i6j5R * dp.x - Q17i6j5I * dm.y;
    x11.y += Q17i6j5R * dp.y + Q17i6j5I * dm.x;
    x7.x += Q17i7j5R * dp.x + Q17i7j5I * dm.y;
    x7.y += Q17i7j5R * dp.y - Q17i7j5I * dm.x;
    x10.x += Q17i7j5R * dp.x - Q17i7j5I * dm.y;
    x10.y += Q17i7j5R * dp.y + Q17i7j5I * dm.x;
    x8.x += Q17i8j5R * dp.x + Q17i8j5I * dm.y;
    x8.y += Q17i8j5R * dp.y - Q17i8j5I * dm.x;
    x9.x += Q17i8j5R * dp.x - Q17i8j5I * dm.y;
    x9.y += Q17i8j5R * dp.y + Q17i8j5I * dm.x;
    dp = (*R6) + (*R11);
    dm = (*R6) - (*R11);
    x1.x += Q17i1j6R * dp.x + Q17i1j6I * dm.y;
    x1.y += Q17i1j6R * dp.y - Q17i1j6I * dm.x;
    x16.x += Q17i1j6R * dp.x - Q17i1j6I * dm.y;
    x16.y += Q17i1j6R * dp.y + Q17i1j6I * dm.x;
    x2.x += Q17i2j6R * dp.x + Q17i2j6I * dm.y;
    x2.y += Q17i2j6R * dp.y - Q17i2j6I * dm.x;
    x15.x += Q17i2j6R * dp.x - Q17i2j6I * dm.y;
    x15.y += Q17i2j6R * dp.y + Q17i2j6I * dm.x;
    x3.x += Q17i3j6R * dp.x + Q17i3j6I * dm.y;
    x3.y += Q17i3j6R * dp.y - Q17i3j6I * dm.x;
    x14.x += Q17i3j6R * dp.x - Q17i3j6I * dm.y;
    x14.y += Q17i3j6R * dp.y + Q17i3j6I * dm.x;
    x4.x += Q17i4j6R * dp.x + Q17i4j6I * dm.y;
    x4.y += Q17i4j6R * dp.y - Q17i4j6I * dm.x;
    x13.x += Q17i4j6R * dp.x - Q17i4j6I * dm.y;
    x13.y += Q17i4j6R * dp.y + Q17i4j6I * dm.x;
    x5.x += Q17i5j6R * dp.x + Q17i5j6I * dm.y;
    x5.y += Q17i5j6R * dp.y - Q17i5j6I * dm.x;
    x12.x += Q17i5j6R * dp.x - Q17i5j6I * dm.y;
    x12.y += Q17i5j6R * dp.y + Q17i5j6I * dm.x;
    x6.x += Q17i6j6R * dp.x + Q17i6j6I * dm.y;
    x6.y += Q17i6j6R * dp.y - Q17i6j6I * dm.x;
    x11.x += Q17i6j6R * dp.x - Q17i6j6I * dm.y;
    x11.y += Q17i6j6R * dp.y + Q17i6j6I * dm.x;
    x7.x += Q17i7j6R * dp.x + Q17i7j6I * dm.y;
    x7.y += Q17i7j6R * dp.y - Q17i7j6I * dm.x;
    x10.x += Q17i7j6R * dp.x - Q17i7j6I * dm.y;
    x10.y += Q17i7j6R * dp.y + Q17i7j6I * dm.x;
    x8.x += Q17i8j6R * dp.x + Q17i8j6I * dm.y;
    x8.y += Q17i8j6R * dp.y - Q17i8j6I * dm.x;
    x9.x += Q17i8j6R * dp.x - Q17i8j6I * dm.y;
    x9.y += Q17i8j6R * dp.y + Q17i8j6I * dm.x;
    dp = (*R7) + (*R10);
    dm = (*R7) - (*R10);
    x1.x += Q17i1j7R * dp.x + Q17i1j7I * dm.y;
    x1.y += Q17i1j7R * dp.y - Q17i1j7I * dm.x;
    x16.x += Q17i1j7R * dp.x - Q17i1j7I * dm.y;
    x16.y += Q17i1j7R * dp.y + Q17i1j7I * dm.x;
    x2.x += Q17i2j7R * dp.x + Q17i2j7I * dm.y;
    x2.y += Q17i2j7R * dp.y - Q17i2j7I * dm.x;
    x15.x += Q17i2j7R * dp.x - Q17i2j7I * dm.y;
    x15.y += Q17i2j7R * dp.y + Q17i2j7I * dm.x;
    x3.x += Q17i3j7R * dp.x + Q17i3j7I * dm.y;
    x3.y += Q17i3j7R * dp.y - Q17i3j7I * dm.x;
    x14.x += Q17i3j7R * dp.x - Q17i3j7I * dm.y;
    x14.y += Q17i3j7R * dp.y + Q17i3j7I * dm.x;
    x4.x += Q17i4j7R * dp.x + Q17i4j7I * dm.y;
    x4.y += Q17i4j7R * dp.y - Q17i4j7I * dm.x;
    x13.x += Q17i4j7R * dp.x - Q17i4j7I * dm.y;
    x13.y += Q17i4j7R * dp.y + Q17i4j7I * dm.x;
    x5.x += Q17i5j7R * dp.x + Q17i5j7I * dm.y;
    x5.y += Q17i5j7R * dp.y - Q17i5j7I * dm.x;
    x12.x += Q17i5j7R * dp.x - Q17i5j7I * dm.y;
    x12.y += Q17i5j7R * dp.y + Q17i5j7I * dm.x;
    x6.x += Q17i6j7R * dp.x + Q17i6j7I * dm.y;
    x6.y += Q17i6j7R * dp.y - Q17i6j7I * dm.x;
    x11.x += Q17i6j7R * dp.x - Q17i6j7I * dm.y;
    x11.y += Q17i6j7R * dp.y + Q17i6j7I * dm.x;
    x7.x += Q17i7j7R * dp.x + Q17i7j7I * dm.y;
    x7.y += Q17i7j7R * dp.y - Q17i7j7I * dm.x;
    x10.x += Q17i7j7R * dp.x - Q17i7j7I * dm.y;
    x10.y += Q17i7j7R * dp.y + Q17i7j7I * dm.x;
    x8.x += Q17i8j7R * dp.x + Q17i8j7I * dm.y;
    x8.y += Q17i8j7R * dp.y - Q17i8j7I * dm.x;
    x9.x += Q17i8j7R * dp.x - Q17i8j7I * dm.y;
    x9.y += Q17i8j7R * dp.y + Q17i8j7I * dm.x;
    dp = (*R8) + (*R9);
    dm = (*R8) - (*R9);
    x1.x += Q17i1j8R * dp.x + Q17i1j8I * dm.y;
    x1.y += Q17i1j8R * dp.y - Q17i1j8I * dm.x;
    x16.x += Q17i1j8R * dp.x - Q17i1j8I * dm.y;
    x16.y += Q17i1j8R * dp.y + Q17i1j8I * dm.x;
    x2.x += Q17i2j8R * dp.x + Q17i2j8I * dm.y;
    x2.y += Q17i2j8R * dp.y - Q17i2j8I * dm.x;
    x15.x += Q17i2j8R * dp.x - Q17i2j8I * dm.y;
    x15.y += Q17i2j8R * dp.y + Q17i2j8I * dm.x;
    x3.x += Q17i3j8R * dp.x + Q17i3j8I * dm.y;
    x3.y += Q17i3j8R * dp.y - Q17i3j8I * dm.x;
    x14.x += Q17i3j8R * dp.x - Q17i3j8I * dm.y;
    x14.y += Q17i3j8R * dp.y + Q17i3j8I * dm.x;
    x4.x += Q17i4j8R * dp.x + Q17i4j8I * dm.y;
    x4.y += Q17i4j8R * dp.y - Q17i4j8I * dm.x;
    x13.x += Q17i4j8R * dp.x - Q17i4j8I * dm.y;
    x13.y += Q17i4j8R * dp.y + Q17i4j8I * dm.x;
    x5.x += Q17i5j8R * dp.x + Q17i5j8I * dm.y;
    x5.y += Q17i5j8R * dp.y - Q17i5j8I * dm.x;
    x12.x += Q17i5j8R * dp.x - Q17i5j8I * dm.y;
    x12.y += Q17i5j8R * dp.y + Q17i5j8I * dm.x;
    x6.x += Q17i6j8R * dp.x + Q17i6j8I * dm.y;
    x6.y += Q17i6j8R * dp.y - Q17i6j8I * dm.x;
    x11.x += Q17i6j8R * dp.x - Q17i6j8I * dm.y;
    x11.y += Q17i6j8R * dp.y + Q17i6j8I * dm.x;
    x7.x += Q17i7j8R * dp.x + Q17i7j8I * dm.y;
    x7.y += Q17i7j8R * dp.y - Q17i7j8I * dm.x;
    x10.x += Q17i7j8R * dp.x - Q17i7j8I * dm.y;
    x10.y += Q17i7j8R * dp.y + Q17i7j8I * dm.x;
    x8.x += Q17i8j8R * dp.x + Q17i8j8I * dm.y;
    x8.y += Q17i8j8R * dp.y - Q17i8j8I * dm.x;
    x9.x += Q17i8j8R * dp.x - Q17i8j8I * dm.y;
    x9.y += Q17i8j8R * dp.y + Q17i8j8I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
    (*R13) = x13;
    (*R14) = x14;
    (*R15) = x15;
    (*R16) = x16;
}

#endif // ROCFFT_BUTTERFLY_TEMPLATE_H

// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef REAL_TO_COMPLEX_DEVICE_H
#define REAL_TO_COMPLEX_DEVICE_H

// The even-length real to complex post process device kernel
template <typename Tcomplex, bool Ndiv4, CallbackType cbtype, bool SCALE = false>
__device__ inline void post_process_interleaved(const size_t    idx_p,
                                                const size_t    idx_q,
                                                const size_t    half_N,
                                                const size_t    quarter_N,
                                                const Tcomplex* input,
                                                Tcomplex*       output,
                                                size_t          output_base,
                                                const Tcomplex* twiddles,
                                                void* __restrict__ load_cb_fn,
                                                void* __restrict__ load_cb_data,
                                                uint32_t load_cb_lds_bytes,
                                                void* __restrict__ store_cb_fn,
                                                void* __restrict__ store_cb_data,
                                                const real_type_t<Tcomplex> scale_factor = 0.0)
{
    // post process can't be the first kernel, so don't bother
    // going through the load cb to read global memory
    auto store_cb = get_store_cb<Tcomplex, cbtype>(store_cb_fn);

    Tcomplex outval;

    if(idx_p == 0)
    {
        outval.x = input[0].x - input[0].y;
        outval.y = 0;
        store_cb(output,
                 output_base + half_N,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);

        outval.x = input[0].x + input[0].y;
        outval.y = 0;
        store_cb(output,
                 output_base + 0,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);

        if(Ndiv4)
        {
            outval.x = input[quarter_N].x;
            outval.y = -input[quarter_N].y;

            store_cb(output,
                     output_base + quarter_N,
                     SCALE ? (outval * scale_factor) : outval,
                     store_cb_data,
                     nullptr);
        }
    }
    else
    {
        const Tcomplex p = input[idx_p];
        const Tcomplex q = input[idx_q];
        const Tcomplex u = 0.5 * (p + q);
        const Tcomplex v = 0.5 * (p - q);

        const Tcomplex twd_p = twiddles[idx_p];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        outval.x = u.x + v.x * twd_p.y + u.y * twd_p.x;
        outval.y = v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(output,
                 output_base + idx_p,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);

        outval.x = u.x - v.x * twd_p.y - u.y * twd_p.x;
        outval.y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(output,
                 output_base + idx_q,
                 SCALE ? (outval * scale_factor) : outval,
                 store_cb_data,
                 nullptr);
    }
}

// TODO: rework pre/post processing
template <typename T, bool Ndiv4, CallbackType cbtype>
__device__ inline void post_process_interleaved_inplace(const size_t idx_p,
                                                        const size_t idx_q,
                                                        const size_t half_N,
                                                        const size_t quarter_N,
                                                        T*           inout,
                                                        size_t       offset_base,
                                                        const T*     twiddles,
                                                        void* __restrict__ load_cb_fn,
                                                        void* __restrict__ load_cb_data,
                                                        uint32_t load_cb_lds_bytes,
                                                        void* __restrict__ store_cb_fn,
                                                        void* __restrict__ store_cb_data)
{
    // post process can't be the first kernel, so don't bother
    // going through the load cb to read global memory
    auto store_cb = get_store_cb<T, cbtype>(store_cb_fn);

    T p, q, outval;
    if(idx_p < quarter_N)
    {
        p = inout[offset_base + idx_p];
        q = inout[offset_base + idx_q];
    }

    __syncthreads();

    if(idx_p == 0)
    {
        outval.x = p.x + p.y;
        outval.y = 0;
        store_cb(inout, offset_base + idx_p, outval, store_cb_data, nullptr);

        outval.x = p.x - p.y;
        outval.y = 0;
        store_cb(inout, offset_base + idx_q, outval, store_cb_data, nullptr);

        if(Ndiv4)
        {
            outval   = inout[offset_base + quarter_N];
            outval.y = -outval.y;
            store_cb(inout, offset_base + quarter_N, outval, store_cb_data, nullptr);
        }
    }
    else if(idx_p < quarter_N)
    {
        const T u = 0.5 * (p + q);
        const T v = 0.5 * (p - q);

        const T twd_p = twiddles[idx_p];
        // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

        outval.x = u.x + v.x * twd_p.y + u.y * twd_p.x;
        outval.y = v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(inout, offset_base + idx_p, outval, store_cb_data, nullptr);

        outval.x = u.x - v.x * twd_p.y - u.y * twd_p.x;
        outval.y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        store_cb(inout, offset_base + idx_q, outval, store_cb_data, nullptr);
    }
}

// The below 2 functions are only for inplace in lds. So no callback.
template <typename Tcomplex, bool Ndiv4>
__device__ inline void real_post_process_kernel_inplace(const size_t    idx_p,
                                                        const size_t    idx_q,
                                                        const size_t    quarter_N,
                                                        Tcomplex*       inout,
                                                        size_t          offset_base,
                                                        const Tcomplex* twiddles)
{
    if(idx_p < quarter_N)
    {
        Tcomplex p = inout[offset_base + idx_p];
        Tcomplex q = inout[offset_base + idx_q];

        if(idx_p == 0)
        {
            inout[offset_base + idx_p].x = p.x + p.y;
            inout[offset_base + idx_p].y = 0;

            inout[offset_base + idx_q].x = p.x - p.y;
            inout[offset_base + idx_q].y = 0;

            if(Ndiv4)
            {
                inout[offset_base + quarter_N].y = -inout[offset_base + quarter_N].y;
            }
        }
        else
        {
            const Tcomplex u = 0.5 * (p + q);
            const Tcomplex v = 0.5 * (p - q);

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p) = (-twd_p.x, twd_p.y);

            inout[offset_base + idx_p].x = u.x + v.x * twd_p.y + u.y * twd_p.x;
            inout[offset_base + idx_p].y = v.y + u.y * twd_p.y - v.x * twd_p.x;

            inout[offset_base + idx_q].x = u.x - v.x * twd_p.y - u.y * twd_p.x;
            inout[offset_base + idx_q].y = -v.y + u.y * twd_p.y - v.x * twd_p.x;
        }
    }
}

template <typename Tcomplex, bool Ndiv4>
__device__ inline void real_pre_process_kernel_inplace(const size_t    idx_p,
                                                       const size_t    idx_q,
                                                       const size_t    quarter_N,
                                                       Tcomplex*       inout,
                                                       size_t          offset_base,
                                                       const Tcomplex* twiddles)
{
    if(idx_p < quarter_N)
    {
        Tcomplex p = inout[offset_base + idx_p];
        Tcomplex q = inout[offset_base + idx_q];

        if(idx_p == 0)
        {
            // NB: multi-dimensional transforms may have non-zero
            // imaginary part at index 0 or at the Nyquist frequency.
            inout[offset_base + idx_p].x = p.x - p.y + q.x + q.y;
            inout[offset_base + idx_p].y = p.x + p.y - q.x + q.y;

            if(Ndiv4)
            {
                auto quarter_elem                = inout[offset_base + quarter_N];
                inout[offset_base + quarter_N].x = 2.0 * quarter_elem.x;
                inout[offset_base + quarter_N].y = -2.0 * quarter_elem.y;
            }
        }
        else
        {
            const Tcomplex u = p + q;
            const Tcomplex v = p - q;

            const Tcomplex twd_p = twiddles[idx_p];
            // NB: twd_q = -conj(twd_p);

            inout[offset_base + idx_p].x = u.x + v.x * twd_p.y - u.y * twd_p.x;
            inout[offset_base + idx_p].y = v.y + u.y * twd_p.y + v.x * twd_p.x;

            inout[offset_base + idx_q].x = u.x - v.x * twd_p.y + u.y * twd_p.x;
            inout[offset_base + idx_q].y = -v.y + u.y * twd_p.y + v.x * twd_p.x;
        }
    }
}

#endif

// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// complex number operators that are not present in hipRTC

#ifndef ROCFFT_RTC_WORKAROUND_H
#define ROCFFT_RTC_WORKAROUND_H

__device__ float2& operator*=(float2& f2, const float f)
{
    return f2 *= float2{f};
}

__device__ double2& operator*=(double2& f2, const double f)
{
    return f2 *= double2{f};
}

__device__ float2 operator-(float2 f2)
{
    return float2{-f2.x, -f2.y};
}

__device__ double2 operator-(double2 f2)
{
    return double2{-f2.x, -f2.y};
}

#endif // ROCFFT_RTC_WORKAROUND_H

#ifndef RIDER_LDS_TO_REG
#define RIDER_LDS_TO_REG
template <typename scalar_type, StrideBin sb>
__device__ void lds_to_reg_input_length125_device(scalar_type* R,
                                                  scalar_type* __restrict__ lds_complex,
                                                  unsigned int stride_lds,
                                                  unsigned int offset_lds,
                                                  unsigned int thread,
                                                  bool         write)
{
    const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    unsigned int       l_offset;
    __syncthreads();
    l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
    R[0]     = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 25) * lstride;
    R[1]     = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 50) * lstride;
    R[2]     = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 75) * lstride;
    R[3]     = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 100) * lstride;
    R[4]     = lds_complex[l_offset];
}
template <typename scalar_type, StrideBin sb>
__device__ void lds_from_reg_output_length125_device(scalar_type* R,
                                                     scalar_type* __restrict__ lds_complex,
                                                     unsigned int stride_lds,
                                                     unsigned int offset_lds,
                                                     unsigned int thread,
                                                     bool         write)
{
    const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    unsigned int       l_offset;
    __syncthreads();
    l_offset = offset_lds + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 0) * lstride;
    lds_complex[l_offset] = R[0];
    l_offset = offset_lds + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 25) * lstride;
    lds_complex[l_offset] = R[1];
    l_offset = offset_lds + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 50) * lstride;
    lds_complex[l_offset] = R[2];
    l_offset = offset_lds + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 75) * lstride;
    lds_complex[l_offset] = R[3];
    l_offset = offset_lds + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 100) * lstride;
    lds_complex[l_offset] = R[4];
}

#endif

template <typename scalar_type,
          const bool lds_is_real,
          StrideBin  sb,
          const bool lds_linear,
          const bool direct_load_to_reg,
          bool       apply_large_twiddle,
          size_t     large_twiddle_steps = 3,
          size_t     large_twiddle_base  = 8>
__device__ void forward_length125_SBCC_device(scalar_type* R,
                                              real_type_t<scalar_type>* __restrict__ lds_real,
                                              scalar_type* __restrict__ lds_complex,
                                              const scalar_type* __restrict__ twiddles,
                                              unsigned int       stride_lds,
                                              unsigned int       offset_lds,
                                              unsigned int       thread,
                                              bool               write,
                                              const scalar_type* large_twiddles,
                                              size_t             trans_local)
{
    scalar_type        W;
    scalar_type        t;
    const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    unsigned int       l_offset;

    // pass 0, width 5
    // using 25 threads we need to do 25 radix-5 butterflies
    // therefore each thread will do 1.000000 butterflies
    FwdRad5B1(&R[0], &R[1], &R[2], &R[3], &R[4]);
    if(!lds_is_real)
    {
        if(!direct_load_to_reg)
        {
            __syncthreads();
        }

        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 0) * lstride;
        lds_complex[l_offset] = R[0];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 1) * lstride;
        lds_complex[l_offset] = R[1];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 2) * lstride;
        lds_complex[l_offset] = R[2];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 3) * lstride;
        lds_complex[l_offset] = R[3];
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 4) * lstride;
        lds_complex[l_offset] = R[4];
    }

    else
    {
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 0) * lstride;
        lds_real[l_offset] = R[0].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 1) * lstride;
        lds_real[l_offset] = R[1].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 2) * lstride;
        lds_real[l_offset] = R[2].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 3) * lstride;
        lds_real[l_offset] = R[3].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 4) * lstride;
        lds_real[l_offset] = R[4].x;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 25) * lstride;
        R[1].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 50) * lstride;
        R[2].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 75) * lstride;
        R[3].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 100) * lstride;
        R[4].x   = lds_real[l_offset];
        __syncthreads();
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 0) * lstride;
        lds_real[l_offset] = R[0].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 1) * lstride;
        lds_real[l_offset] = R[1].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 2) * lstride;
        lds_real[l_offset] = R[2].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 3) * lstride;
        lds_real[l_offset] = R[3].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 1) * 5 + (thread + 0 + 0) % 1 + 4) * lstride;
        lds_real[l_offset] = R[4].y;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 25) * lstride;
        R[1].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 50) * lstride;
        R[2].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 75) * lstride;
        R[3].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 100) * lstride;
        R[4].y   = lds_real[l_offset];
    }

    // pass 1, width 5
    // using 25 threads we need to do 25 radix-5 butterflies
    // therefore each thread will do 1.000000 butterflies
    if(!lds_is_real)
    {
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 25) * lstride;
        R[1]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 50) * lstride;
        R[2]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 75) * lstride;
        R[3]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 100) * lstride;
        R[4]     = lds_complex[l_offset];
    }

    W    = twiddles[0 + 4 * ((thread + 0 + 0) % 5)];
    t    = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
    R[1] = t;
    W    = twiddles[1 + 4 * ((thread + 0 + 0) % 5)];
    t    = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
    R[2] = t;
    W    = twiddles[2 + 4 * ((thread + 0 + 0) % 5)];
    t    = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
    R[3] = t;
    W    = twiddles[3 + 4 * ((thread + 0 + 0) % 5)];
    t    = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
    R[4] = t;
    FwdRad5B1(&R[0], &R[1], &R[2], &R[3], &R[4]);
    if(!lds_is_real)
    {
        __syncthreads();
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 0) * lstride;
        lds_complex[l_offset] = R[0];
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 5) * lstride;
        lds_complex[l_offset] = R[1];
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 10) * lstride;
        lds_complex[l_offset] = R[2];
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 15) * lstride;
        lds_complex[l_offset] = R[3];
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 20) * lstride;
        lds_complex[l_offset] = R[4];
    }

    else
    {
        __syncthreads();
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 0) * lstride;
        lds_real[l_offset] = R[0].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 5) * lstride;
        lds_real[l_offset] = R[1].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 10) * lstride;
        lds_real[l_offset] = R[2].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 15) * lstride;
        lds_real[l_offset] = R[3].x;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 20) * lstride;
        lds_real[l_offset] = R[4].x;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 25) * lstride;
        R[1].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 50) * lstride;
        R[2].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 75) * lstride;
        R[3].x   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 100) * lstride;
        R[4].x   = lds_real[l_offset];
        __syncthreads();
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 0) * lstride;
        lds_real[l_offset] = R[0].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 5) * lstride;
        lds_real[l_offset] = R[1].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 10) * lstride;
        lds_real[l_offset] = R[2].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 15) * lstride;
        lds_real[l_offset] = R[3].y;
        l_offset = offset_lds + (((thread + 0 + 0) / 5) * 25 + (thread + 0 + 0) % 5 + 20) * lstride;
        lds_real[l_offset] = R[4].y;
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 25) * lstride;
        R[1].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 50) * lstride;
        R[2].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 75) * lstride;
        R[3].y   = lds_real[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 100) * lstride;
        R[4].y   = lds_real[l_offset];
    }

    // pass 2, width 5
    // using 25 threads we need to do 25 radix-5 butterflies
    // therefore each thread will do 1.000000 butterflies
    if(!lds_is_real)
    {
        __syncthreads();
        l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
        R[0]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 25) * lstride;
        R[1]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 50) * lstride;
        R[2]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 75) * lstride;
        R[3]     = lds_complex[l_offset];
        l_offset = offset_lds + ((thread + 0 + 0) + 100) * lstride;
        R[4]     = lds_complex[l_offset];
    }

    W    = twiddles[20 + 4 * ((thread + 0 + 0) % 25)];
    t    = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
    R[1] = t;
    W    = twiddles[21 + 4 * ((thread + 0 + 0) % 25)];
    t    = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
    R[2] = t;
    W    = twiddles[22 + 4 * ((thread + 0 + 0) % 25)];
    t    = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
    R[3] = t;
    W    = twiddles[23 + 4 * ((thread + 0 + 0) % 25)];
    t    = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
    R[4] = t;
    FwdRad5B1(&R[0], &R[1], &R[2], &R[3], &R[4]);
    if(apply_large_twiddle)
    {
        // large twiddle multiplication
        W = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 25) + 0 * 25) * trans_local);
        t    = {R[0].x * W.x - R[0].y * W.y, R[0].y * W.x + R[0].x * W.y};
        R[0] = t;
        W    = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 25) + 1 * 25) * trans_local);
        t    = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
        R[1] = t;
        W    = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 25) + 2 * 25) * trans_local);
        t    = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
        R[2] = t;
        W    = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 25) + 3 * 25) * trans_local);
        t    = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
        R[3] = t;
        W    = TW_NSteps<scalar_type, large_twiddle_base, large_twiddle_steps>(
            large_twiddles, (((int)(thread + 0 + 0) % 25) + 4 * 25) * trans_local);
        t    = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
        R[4] = t;
    }
}
typedef float2                   scalar_type;
static const StrideBin           sb                  = SB_NONUNIT;
static const EmbeddedType        ebtype              = EmbeddedType::NONE;
static const SBRC_TYPE           sbrc_type           = SBRC_2D;
static const SBRC_TRANSPOSE_TYPE transpose_type      = NONE;
static const CallbackType        cbtype              = CallbackType::NONE;
static const DirectRegType       drtype              = DirectRegType::TRY_ENABLE_IF_SUPPORT;
static const bool                apply_large_twiddle = false;
static const IntrinsicAccessType intrinsic_mode      = IntrinsicAccessType::DISABLE_BOTH;
static const size_t              large_twiddle_base  = 8;
static const size_t              large_twiddle_steps = 0;
extern "C" __global__ __launch_bounds__(400) void fft_rtc_fwd_len125_sp_ip_CI_sbcc_dirReg(
    const scalar_type* __restrict__ twiddles,
    const scalar_type* large_twiddles,
    const size_t       dim,
    const size_t* __restrict__ lengths,
    const size_t* __restrict__ stride,
    const size_t       nbatch,
    const unsigned int lds_padding,
    void* __restrict__ load_cb_fn,
    void* __restrict__ load_cb_data,
    uint32_t load_cb_lds_bytes,
    void* __restrict__ store_cb_fn,
    void* __restrict__ store_cb_data,
    scalar_type* __restrict__ buf)
{
    // this kernel:
    //   uses 25 threads per transform
    //   does 16 transforms per thread block
    // therefore it should be called with 400 threads per thread block
    scalar_type R[5];
    extern __shared__ unsigned char __attribute__((aligned(sizeof(scalar_type)))) lds_uchar[];
    real_type_t<scalar_type>* __restrict__ lds_real
        = reinterpret_cast<real_type_t<scalar_type>*>(lds_uchar);
    scalar_type* __restrict__ lds_complex = reinterpret_cast<scalar_type*>(lds_uchar);
    size_t       offset                   = 0;
    unsigned int offset_lds;
    unsigned int stride_lds;
    size_t       batch;
    size_t       transform;
    const bool   direct_load_to_reg    = drtype == DirectRegType::TRY_ENABLE_IF_SUPPORT;
    const bool   direct_store_from_reg = direct_load_to_reg;
    const bool   lds_linear            = !direct_load_to_reg;
    const bool   lds_is_real           = false;
    auto         load_cb               = get_load_cb<scalar_type, cbtype>(load_cb_fn);
    auto         store_cb              = get_store_cb<scalar_type, cbtype>(store_cb_fn);

    // large twiddles
    __shared__ scalar_type large_twd_lds[(apply_large_twiddle && large_twiddle_base < 8)
                                             ? ((1 << large_twiddle_base) * 3)
                                             : (0)];
    if(apply_large_twiddle && large_twiddle_base < 8)
    {
        size_t ltwd_id = threadIdx.x;
        while(ltwd_id < (1 << large_twiddle_base) * 3)
        {
            large_twd_lds[ltwd_id] = large_twiddles[ltwd_id];
            ltwd_id += 400;
        }
    }

    // offsets
    const size_t stride0 = (sb == SB_UNIT) ? (1) : (stride[0]);
    size_t       tile_index;
    size_t       num_of_tiles;

    // calculate offset for each tile:
    //   tile_index  now means index of the tile along dim1
    //   num_of_tiles now means number of tiles along dim1
    size_t plength = 1;
    size_t remaining;
    size_t index_along_d;
    num_of_tiles = (lengths[1] - 1) / 16 + 1;
    plength      = num_of_tiles;
    tile_index   = blockIdx.x % num_of_tiles;
    remaining    = blockIdx.x / num_of_tiles;
    offset       = tile_index * 16 * stride[1];
    for(int d = 2; d < dim; ++d)
    {
        plength       = plength * lengths[d];
        index_along_d = remaining % lengths[d];
        remaining     = remaining / lengths[d];
        offset        = offset + index_along_d * stride[d];
    }

    batch  = blockIdx.x / plength;
    offset = offset + batch * stride[dim];
    transform
        = lds_linear ? tile_index * 16 + threadIdx.x / 25 : tile_index * 16 + threadIdx.x % 16;
    stride_lds            = lds_linear ? 125 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding)
                                       : 16 + (ebtype == EmbeddedType::NONE ? 0 : lds_padding);
    offset_lds            = lds_linear ? stride_lds * (transform % 16) : threadIdx.x % 16;
    bool         in_bound = ((tile_index + 1) * 16 > lengths[1]) ? false : true;
    unsigned int thread   = threadIdx.x / 16;
    unsigned int tid_hor  = threadIdx.x % 16;

    if(direct_load_to_reg)
    {
        // load global into registers
        if(intrinsic_mode != IntrinsicAccessType::DISABLE_BOTH)
        {
            // use intrinsic load
            // evaluate all flags as one rw argument
            R[0] = intrinsic_load(buf,
                                  tid_hor * stride[1] + (((thread + 0 + 0) + 0)) * stride0,
                                  offset,
                                  (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            R[1] = intrinsic_load(buf,
                                  tid_hor * stride[1] + (((thread + 0 + 0) + 25)) * stride0,
                                  offset,
                                  (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            R[2] = intrinsic_load(buf,
                                  tid_hor * stride[1] + (((thread + 0 + 0) + 50)) * stride0,
                                  offset,
                                  (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            R[3] = intrinsic_load(buf,
                                  tid_hor * stride[1] + (((thread + 0 + 0) + 75)) * stride0,
                                  offset,
                                  (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            R[4] = intrinsic_load(buf,
                                  tid_hor * stride[1] + (((thread + 0 + 0) + 100)) * stride0,
                                  offset,
                                  (in_bound || tile_index * 16 + tid_hor < lengths[1]));
        }

        else
        {
            // can't use intrinsic load
            if(in_bound)
            {
                R[0] = load_cb(buf,
                               offset + tid_hor * stride[1] + (((thread + 0 + 0) + 0)) * stride0,
                               load_cb_data,
                               nullptr);
                R[1] = load_cb(buf,
                               offset + tid_hor * stride[1] + (((thread + 0 + 0) + 25)) * stride0,
                               load_cb_data,
                               nullptr);
                R[2] = load_cb(buf,
                               offset + tid_hor * stride[1] + (((thread + 0 + 0) + 50)) * stride0,
                               load_cb_data,
                               nullptr);
                R[3] = load_cb(buf,
                               offset + tid_hor * stride[1] + (((thread + 0 + 0) + 75)) * stride0,
                               load_cb_data,
                               nullptr);
                R[4] = load_cb(buf,
                               offset + tid_hor * stride[1] + (((thread + 0 + 0) + 100)) * stride0,
                               load_cb_data,
                               nullptr);
            }

            if(!in_bound)
            {
                if(tile_index * 16 + tid_hor < lengths[1])
                {
                    R[0]
                        = load_cb(buf,
                                  offset + tid_hor * stride[1] + (((thread + 0 + 0) + 0)) * stride0,
                                  load_cb_data,
                                  nullptr);
                    R[1] = load_cb(buf,
                                   offset + tid_hor * stride[1]
                                       + (((thread + 0 + 0) + 25)) * stride0,
                                   load_cb_data,
                                   nullptr);
                    R[2] = load_cb(buf,
                                   offset + tid_hor * stride[1]
                                       + (((thread + 0 + 0) + 50)) * stride0,
                                   load_cb_data,
                                   nullptr);
                    R[3] = load_cb(buf,
                                   offset + tid_hor * stride[1]
                                       + (((thread + 0 + 0) + 75)) * stride0,
                                   load_cb_data,
                                   nullptr);
                    R[4] = load_cb(buf,
                                   offset + tid_hor * stride[1]
                                       + (((thread + 0 + 0) + 100)) * stride0,
                                   load_cb_data,
                                   nullptr);
                }
            }
        }
    }

    else
    {
        // load global into lds
        // no intrinsic when load to lds. FIXME- check why use nested branch is better
        if(in_bound)
        {
            lds_complex[tid_hor * stride_lds + (thread + 0) * 1] = load_cb(
                buf, offset + tid_hor * stride[1] + (thread + 0) * stride0, load_cb_data, nullptr);
            lds_complex[tid_hor * stride_lds + (thread + 25) * 1] = load_cb(
                buf, offset + tid_hor * stride[1] + (thread + 25) * stride0, load_cb_data, nullptr);
            lds_complex[tid_hor * stride_lds + (thread + 50) * 1] = load_cb(
                buf, offset + tid_hor * stride[1] + (thread + 50) * stride0, load_cb_data, nullptr);
            lds_complex[tid_hor * stride_lds + (thread + 75) * 1] = load_cb(
                buf, offset + tid_hor * stride[1] + (thread + 75) * stride0, load_cb_data, nullptr);
            lds_complex[tid_hor * stride_lds + (thread + 100) * 1]
                = load_cb(buf,
                          offset + tid_hor * stride[1] + (thread + 100) * stride0,
                          load_cb_data,
                          nullptr);
        }

        if(!in_bound)
        {
            if(tile_index * 16 + tid_hor < lengths[1])
            {
                lds_complex[tid_hor * stride_lds + (thread + 0) * 1]
                    = load_cb(buf,
                              offset + tid_hor * stride[1] + (thread + 0) * stride0,
                              load_cb_data,
                              nullptr);
                lds_complex[tid_hor * stride_lds + (thread + 25) * 1]
                    = load_cb(buf,
                              offset + tid_hor * stride[1] + (thread + 25) * stride0,
                              load_cb_data,
                              nullptr);
                lds_complex[tid_hor * stride_lds + (thread + 50) * 1]
                    = load_cb(buf,
                              offset + tid_hor * stride[1] + (thread + 50) * stride0,
                              load_cb_data,
                              nullptr);
                lds_complex[tid_hor * stride_lds + (thread + 75) * 1]
                    = load_cb(buf,
                              offset + tid_hor * stride[1] + (thread + 75) * stride0,
                              load_cb_data,
                              nullptr);
                lds_complex[tid_hor * stride_lds + (thread + 100) * 1]
                    = load_cb(buf,
                              offset + tid_hor * stride[1] + (thread + 100) * stride0,
                              load_cb_data,
                              nullptr);
            }
        }
    }

    // calc the thread_in_device value once and for all device funcs
    unsigned int thread_in_device = lds_linear ? threadIdx.x % 25 : threadIdx.x / 16;

    // call a pre-load from lds to registers (if necessary)
    if(!direct_load_to_reg)
    {
        lds_to_reg_input_length125_device<scalar_type, lds_linear ? SB_UNIT : SB_NONUNIT>(
            R, lds_complex, stride_lds, offset_lds, thread_in_device, true);
    }

    // transform
    forward_length125_SBCC_device<scalar_type,
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
        thread_in_device,
        true,
        (apply_large_twiddle && large_twiddle_base < 8) ? (large_twd_lds) : (large_twiddles),
        transform);

    // call a post-store from registers to lds (if necessary)
    if(!direct_store_from_reg)
    {
        lds_from_reg_output_length125_device<scalar_type, lds_linear ? SB_UNIT : SB_NONUNIT>(
            R, lds_complex, stride_lds, offset_lds, thread_in_device, true);
    }

    if(direct_store_from_reg)
    {
        // store registers into global
        if(intrinsic_mode == IntrinsicAccessType::ENABLE_BOTH)
        {
            // use intrinsic store
            store_intrinsic(buf,
                            tid_hor * stride[1]
                                + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 0)
                                      * stride0,
                            offset,
                            R[0],
                            (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            store_intrinsic(buf,
                            tid_hor * stride[1]
                                + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 25)
                                      * stride0,
                            offset,
                            R[1],
                            (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            store_intrinsic(buf,
                            tid_hor * stride[1]
                                + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 50)
                                      * stride0,
                            offset,
                            R[2],
                            (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            store_intrinsic(buf,
                            tid_hor * stride[1]
                                + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 75)
                                      * stride0,
                            offset,
                            R[3],
                            (in_bound || tile_index * 16 + tid_hor < lengths[1]));
            store_intrinsic(buf,
                            tid_hor * stride[1]
                                + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 100)
                                      * stride0,
                            offset,
                            R[4],
                            (in_bound || tile_index * 16 + tid_hor < lengths[1]));
        }

        else
        {
            // can't use intrinsic store
            if(in_bound)
            {
                store_cb(buf,
                         offset + tid_hor * stride[1]
                             + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 0)
                                   * stride0,
                         R[0],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1]
                             + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 25)
                                   * stride0,
                         R[1],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1]
                             + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 50)
                                   * stride0,
                         R[2],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1]
                             + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 75)
                                   * stride0,
                         R[3],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1]
                             + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 100)
                                   * stride0,
                         R[4],
                         store_cb_data,
                         nullptr);
            }

            if(!in_bound)
            {
                if(tile_index * 16 + tid_hor < lengths[1])
                {
                    store_cb(buf,
                             offset + tid_hor * stride[1]
                                 + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 0)
                                       * stride0,
                             R[0],
                             store_cb_data,
                             nullptr);
                    store_cb(buf,
                             offset + tid_hor * stride[1]
                                 + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 25)
                                       * stride0,
                             R[1],
                             store_cb_data,
                             nullptr);
                    store_cb(buf,
                             offset + tid_hor * stride[1]
                                 + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 50)
                                       * stride0,
                             R[2],
                             store_cb_data,
                             nullptr);
                    store_cb(buf,
                             offset + tid_hor * stride[1]
                                 + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 75)
                                       * stride0,
                             R[3],
                             store_cb_data,
                             nullptr);
                    store_cb(buf,
                             offset + tid_hor * stride[1]
                                 + (((thread + 0 + 0) / 25) * 125 + (thread + 0 + 0) % 25 + 100)
                                       * stride0,
                             R[4],
                             store_cb_data,
                             nullptr);
                }
            }
        }
    }

    else
    {

        // store global
        __syncthreads();
        // no intrinsic when store from lds. FIXME- check why use nested branch is better
        if(in_bound)
        {
            store_cb(buf,
                     offset + tid_hor * stride[1] + (thread + 0) * stride0,
                     lds_complex[tid_hor * stride_lds + (thread + 0) * 1],
                     store_cb_data,
                     nullptr);
            store_cb(buf,
                     offset + tid_hor * stride[1] + (thread + 25) * stride0,
                     lds_complex[tid_hor * stride_lds + (thread + 25) * 1],
                     store_cb_data,
                     nullptr);
            store_cb(buf,
                     offset + tid_hor * stride[1] + (thread + 50) * stride0,
                     lds_complex[tid_hor * stride_lds + (thread + 50) * 1],
                     store_cb_data,
                     nullptr);
            store_cb(buf,
                     offset + tid_hor * stride[1] + (thread + 75) * stride0,
                     lds_complex[tid_hor * stride_lds + (thread + 75) * 1],
                     store_cb_data,
                     nullptr);
            store_cb(buf,
                     offset + tid_hor * stride[1] + (thread + 100) * stride0,
                     lds_complex[tid_hor * stride_lds + (thread + 100) * 1],
                     store_cb_data,
                     nullptr);
        }

        if(!in_bound)
        {
            if(tile_index * 16 + tid_hor < lengths[1])
            {
                store_cb(buf,
                         offset + tid_hor * stride[1] + (thread + 0) * stride0,
                         lds_complex[tid_hor * stride_lds + (thread + 0) * 1],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1] + (thread + 25) * stride0,
                         lds_complex[tid_hor * stride_lds + (thread + 25) * 1],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1] + (thread + 50) * stride0,
                         lds_complex[tid_hor * stride_lds + (thread + 50) * 1],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1] + (thread + 75) * stride0,
                         lds_complex[tid_hor * stride_lds + (thread + 75) * 1],
                         store_cb_data,
                         nullptr);
                store_cb(buf,
                         offset + tid_hor * stride[1] + (thread + 100) * stride0,
                         lds_complex[tid_hor * stride_lds + (thread + 100) * 1],
                         store_cb_data,
                         nullptr);
            }
        }
    }
}
// ROCFFT_RTC_END fft_rtc_fwd_len125_sp_ip_CI_sbcc_dirReg

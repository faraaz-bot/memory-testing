// Case: rocfft-bench --length 200 200 200 -t 2 -o -b 10

/// Scheme Log
//
// Work buffer size: 0
// Work buffer ratio: 0.0
// Assignment strategy: BALANCE_BUFFER_FUSION

// scheme: CS_REAL_3D_EVEN
// dimension: 3
// batch: 10
// length: 200 200 200
// outputLength: 101 200 200
// iStrides: 1 200 40000
// oStrides: 1 101 20200
// iDist: 8000000
// oDist: 4040000
// direction: -1
// placement: OP
// single-precision
// array type: R -> HI
// SBRC_Trans_Type: NONE
// Direct_to_from_Reg: FORCE_OFF_OR_NOT_SUPPORT
// OB_USER_IN -> OB_USER_OUT
// A -> B

//     scheme: CS_REAL_TRANSFORM_EVEN
//     dimension: 1
//     batch: 10
//     length: 200 200 200
//     outputLength: 101 200 200
//     iStrides: 1 200 40000
//     oStrides: 1 101 20200
//     iDist: 8000000
//     oDist: 4040000
//     direction: -1
//     placement: OP
//     single-precision
//     array type: R -> CI
//     SBRC_Trans_Type: NONE
//     Direct_to_from_Reg: FORCE_OFF_OR_NOT_SUPPORT
//     OB_USER_IN -> OB_USER_OUT
//     A -> B

//         scheme: CS_KERNEL_STOCKHAM
//         dimension: 1
//         batch: 400000
//         length: 100
//         outputLength: 101
//         iStrides: 1
//         oStrides: 1
//         iDist: 100
//         oDist: 101
//         direction: -1
//         placement: OP
//         single-precision
//         array type: CI -> CI
//         twiddle table length: 140
//         EmbeddedType: Real2C_POST
//         SBRC_Trans_Type: NONE
//         Direct_to_from_Reg: TRY_ENABLE_IF_SUPPORT
//         OB_USER_IN -> OB_USER_OUT
//         A -> B
//         comment: collapsed contiguous high length(s) 200 200 into batch
//         Leaf-Node: external-kernel configuration:
//             workgroup_size: 60
//             trans_per_block: 6
//             radices: [ 10 10 ]

//     scheme: CS_KERNEL_STOCKHAM_BLOCK_CC
//     dimension: 1
//     batch: 10
//     length: 200 101 200
//     outputLength: 101 200 200
//     iStrides: 20200 1 101
//     oStrides: 20200 1 101
//     iDist: 4040000
//     oDist: 4040000
//     direction: -1
//     placement: IP
//     single-precision
//     array type: CI -> CI
//     twiddle table length: 195
//     SBRC_Trans_Type: NONE
//     Intrinsic Mode: LOAD_ONLY
//     Direct_to_from_Reg: TRY_ENABLE_IF_SUPPORT
//     OB_USER_OUT -> OB_USER_OUT
//     B -> B
//     Leaf-Node: external-kernel configuration:
//         workgroup_size: 400
//         trans_per_block: 10
//         radices: [ 5 8 5 ]

//     scheme: CS_KERNEL_STOCKHAM_BLOCK_CC
//     dimension: 1
//     batch: 2000
//     length: 200 101
//     outputLength: 101 200
//     iStrides: 101 1
//     oStrides: 101 1
//     iDist: 20200
//     oDist: 20200
//     direction: -1
//     placement: IP
//     single-precision
//     array type: CI -> HI
//     twiddle table length: 195
//     SBRC_Trans_Type: NONE
//     Intrinsic Mode: LOAD_ONLY
//     Direct_to_from_Reg: TRY_ENABLE_IF_SUPPORT
//     OB_USER_OUT -> OB_USER_OUT
//     B -> B
//     comment: collapsed contiguous high length(s) 200 into batch
//     Leaf-Node: external-kernel configuration:
//         workgroup_size: 400
//         trans_per_block: 10
//         radices: [ 5 8 5 ]
// GridParams
//   b[66667,1,1] wgs[60,1,1], dy_lds bytes 4848
//   b[22000,1,1] wgs[400,1,1], dy_lds bytes 16000
//   b[22000,1,1] wgs[400,1,1], dy_lds bytes 16000
// End GridParams
// ===============================================================================

#include <hip/hip_runtime.h>

#include "rtc_compile.h"
#include "rtc_kernel.h"
#include <hip/hiprtc.h>

#include <fstream>
#include <functional>
#include <future>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#define ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS

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

#ifndef ROCFFT_GPUBUF_H
#define ROCFFT_GPUBUF_H

// Kernel occupancy: 13
// Kernel occupancy: 4
// Kernel occupancy: 4

// Simple RAII class for GPU buffers.  T is the type of pointer that
// data() returns
template <class T = void>
class gpubuf_t
{
public:
    gpubuf_t() {}
    // buffers are movable but not copyable
    gpubuf_t(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(bsize, other.bsize);
    }
    gpubuf_t& operator=(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(bsize, other.bsize);
        return *this;
    }
    gpubuf_t(const gpubuf_t&)            = delete;
    gpubuf_t& operator=(const gpubuf_t&) = delete;

    ~gpubuf_t()
    {
        free();
    }

    static bool use_alloc_managed()
    {
        return std::getenv("ROCFFT_MALLOC_MANAGED");
    }

    hipError_t alloc(const size_t size)
    {
        bsize                     = size;
        static bool alloc_managed = use_alloc_managed();
        free();
        auto ret = alloc_managed ? hipMallocManaged(&buf, bsize) : hipMalloc(&buf, bsize);
        if(ret != hipSuccess)
        {
            buf   = nullptr;
            bsize = 0;
        }
        return ret;
    }

    size_t size() const
    {
        return bsize;
    }

    void free()
    {
        if(buf != nullptr)
        {
            (void)hipFree(buf);
            buf = nullptr;
        }
    }

    T* data() const
    {
        return static_cast<T*>(buf);
    }

    // equality/bool tests
    bool operator==(std::nullptr_t n) const
    {
        return buf == n;
    }
    bool operator!=(std::nullptr_t n) const
    {
        return buf != n;
    }
    operator bool() const
    {
        return buf;
    }

private:
    // The GPU buffer
    void*  buf   = nullptr;
    size_t bsize = 0;
};

// default gpubuf that gives out void* pointers
typedef gpubuf_t<> gpubuf;
#endif

// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

// utility code to embed into generated test harnesses, to simplify
// allocating and initializing device memory

// copy a host vector to the device
template <typename T>
gpubuf_t<T> host_vec_to_dev(const std::vector<T>& hvec)
{
    gpubuf_t<T> ret;
    if(ret.alloc(sizeof(T) * hvec.size()) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemcpy(ret.data(), hvec.data(), sizeof(T) * hvec.size(), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to memcpy");
    return ret;
}

template <typename T1, typename T2>
T1 ceildiv(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

// generate random complex input
template <typename Tcomplex>
gpubuf_t<Tcomplex> random_complex_device(unsigned int count)
{
    std::vector<Tcomplex> hostBuf(count);

    auto partitions     = std::max<size_t>(std::thread::hardware_concurrency(), 32);
    auto partition_size = ceildiv(count, partitions);

#pragma omp parallel for
    for(unsigned int partition = 0; partition < partitions; ++partition)
    {
        std::mt19937                           gen(partition);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        auto begin = partition * partition_size;
        if(begin >= count)
            continue;
        auto end = std::min(begin + partition_size, count);

        for(auto d = hostBuf.begin() + begin; d != hostBuf.begin() + end; ++d)
        {
            d->x = dis(gen);
            d->y = dis(gen);
        }
    }
    return host_vec_to_dev(hostBuf);
}

// generate random real input
template <typename Treal>
gpubuf_t<Treal> random_real_device(unsigned int count)
{
    std::vector<Treal> hostBuf(count);

    auto partitions     = std::max<size_t>(std::thread::hardware_concurrency(), 32);
    auto partition_size = ceildiv(count, partitions);

#pragma omp parallel for
    for(unsigned int partition = 0; partition < partitions; ++partition)
    {
        std::mt19937                           gen(partition);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        auto begin = partition * partition_size;
        if(begin >= count)
            continue;
        auto end = std::min(begin + partition_size, count);

        for(auto d = hostBuf.begin() + begin; d != hostBuf.begin() + end; ++d)
        {
            *d = dis(gen);
        }
    }
    return host_vec_to_dev(hostBuf);
}

// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_COMPLEX_H
#define ROCFFT_COMPLEX_H

#if !defined(__HIPCC_RTC__)
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
typedef __half _Float16;
#endif

template <typename Treal>
struct rocfft_complex
{

    Treal x; // Real part
    Treal y; // Imaginary part

    // Constructors
    // Do not initialize the members x or y by default, to ensure that it can
    // be used in __shared__ and that it is a trivial class compatible with C.
    __device__ __host__                 rocfft_complex()                       = default;
    __device__ __host__                 rocfft_complex(const rocfft_complex&)  = default;
    __device__ __host__                 rocfft_complex(rocfft_complex&&)       = default;
    __device__ __host__ rocfft_complex& operator=(const rocfft_complex& rhs) & = default;
    __device__ __host__ rocfft_complex& operator=(rocfft_complex&& rhs) &      = default;
    __device__                          __host__ ~rocfft_complex()             = default;

    // Constructor from real and imaginary parts
    __device__ __host__ constexpr rocfft_complex(Treal real, Treal imag)
        : x{real}
        , y{imag}
    {
    }

    // Conversion from different precision
    template <typename U>
    __device__ __host__ explicit constexpr rocfft_complex(const rocfft_complex<U>& z)
        : x(z.x)
        , y(z.y)
    {
    }

    // Accessors
    __device__ __host__ constexpr Treal real() const
    {
        return x;
    }

    __device__ __host__ constexpr Treal imag() const
    {
        return y;
    }

    // Unary operations
    __forceinline__ __device__ __host__ rocfft_complex operator-() const
    {
        return {-x, -y};
    }

    __forceinline__ __device__ __host__ rocfft_complex operator+() const
    {
        return *this;
    }

    __device__ __host__ Treal asum(const rocfft_complex& z)
    {
        return abs(z.x) + abs(z.y);
    }

    // Internal real functions
    static __forceinline__ __device__ __host__ Treal abs(Treal x)
    {
        return x < 0 ? -x : x;
    }

    static __forceinline__ __device__ __host__ float sqrt(float x)
    {
        return ::sqrtf(x);
    }

    static __forceinline__ __device__ __host__ double sqrt(double x)
    {
        return ::sqrt(x);
    }

    // Addition operators
    __device__ __host__ auto& operator+=(const rocfft_complex& rhs)
    {
        return *this = {x + rhs.x, y + rhs.y};
    }

    __device__ __host__ auto operator+(const rocfft_complex& rhs) const
    {
        auto lhs = *this;
        return lhs += rhs;
    }

    // Subtraction operators
    __device__ __host__ auto& operator-=(const rocfft_complex& rhs)
    {
        return *this = {x - rhs.x, y - rhs.y};
    }

    __device__ __host__ auto operator-(const rocfft_complex& rhs) const
    {
        auto lhs = *this;
        return lhs -= rhs;
    }

    // Multiplication operators
    __device__ __host__ auto& operator*=(const rocfft_complex& rhs)
    {
        return *this = {x * rhs.x - y * rhs.y, y * rhs.x + x * rhs.y};
    }

    __device__ __host__ auto operator*(const rocfft_complex& rhs) const
    {
        auto lhs = *this;
        return lhs *= rhs;
    }

    // Division operators
    __device__ __host__ auto& operator/=(const rocfft_complex& rhs)
    {
        // Form of Robert L. Smith's Algorithm 116
        if(abs(rhs.x) > abs(rhs.y))
        {
            Treal ratio = rhs.y / rhs.x;
            Treal scale = 1 / (rhs.x + rhs.y * ratio);
            *this       = {(x + y * ratio) * scale, (y - x * ratio) * scale};
        }
        else
        {
            Treal ratio = rhs.x / rhs.y;
            Treal scale = 1 / (rhs.x * ratio + rhs.y);
            *this       = {(y + x * ratio) * scale, (y * ratio - x) * scale};
        }
        return *this;
    }

    __device__ __host__ auto operator/(const rocfft_complex& rhs) const
    {
        auto lhs = *this;
        return lhs /= rhs;
    }

    // Comparison operators
    __device__ __host__ constexpr bool operator==(const rocfft_complex& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }

    __device__ __host__ constexpr bool operator!=(const rocfft_complex& rhs) const
    {
        return !(*this == rhs);
    }

    // Operators for complex-real computations
    template <typename U>
    __device__ __host__ auto& operator+=(const U& rhs)
    {
        return (x += Treal(rhs)), *this;
    }

    template <typename U>
    __device__ __host__ auto& operator-=(const U& rhs)
    {
        return (x -= Treal(rhs)), *this;
    }

    __device__ __host__ auto operator+(const Treal& rhs)
    {
        auto lhs = *this;
        return lhs += rhs;
    }

    __device__ __host__ auto operator-(const Treal& rhs)
    {
        auto lhs = *this;
        return lhs -= rhs;
    }

    template <typename U>
    __device__ __host__ auto& operator*=(const U& rhs)
    {
        return (x *= Treal(rhs)), (y *= Treal(rhs)), *this;
    }

    template <typename U>
    __device__ __host__ auto operator*(const U& rhs) const
    {
        auto lhs = *this;
        return lhs *= Treal(rhs);
    }

    template <typename U>
    __device__ __host__ auto& operator/=(const U& rhs)
    {
        return (x /= Treal(rhs)), (y /= Treal(rhs)), *this;
    }

    template <typename U>
    __device__ __host__ auto operator/(const U& rhs) const
    {
        auto lhs = *this;
        return lhs /= Treal(rhs);
    }

    template <typename U>
    __device__ __host__ constexpr bool operator==(const U& rhs) const
    {
        return x == Treal(rhs) && y == 0;
    }

    template <typename U>
    __device__ __host__ constexpr bool operator!=(const U& rhs) const
    {
        return !(*this == rhs);
    }
};

// Stream operators
#if !defined(__HIPCC_RTC__)
static std::ostream& operator<<(std::ostream& stream, const _Float16& f)
{
    return stream << static_cast<double>(f);
}

template <typename Treal>
std::ostream& operator<<(std::ostream& out, const rocfft_complex<Treal>& z)
{
    return out << '(' << static_cast<double>(z.x) << ',' << static_cast<double>(z.y) << ')';
}
#endif

// Operators for real-complex computations
template <typename U, typename Treal>
__device__ __host__ rocfft_complex<Treal> operator+(const U& lhs, const rocfft_complex<Treal>& rhs)
{
    return {Treal(lhs) + rhs.x, rhs.y};
}

template <typename U, typename Treal>
__device__ __host__ rocfft_complex<Treal> operator-(const U& lhs, const rocfft_complex<Treal>& rhs)
{
    return {Treal(lhs) - rhs.x, -rhs.y};
}

template <typename U, typename Treal>
__device__ __host__ rocfft_complex<Treal> operator*(const U& lhs, const rocfft_complex<Treal>& rhs)
{
    return {Treal(lhs) * rhs.x, Treal(lhs) * rhs.y};
}

template <typename U, typename Treal>
__device__ __host__ rocfft_complex<Treal> operator/(const U& lhs, const rocfft_complex<Treal>& rhs)
{
    // Form of Robert L. Smith's Algorithm 116
    if(rocfft_complex<Treal>::abs(rhs.x) > rocfft_complex<Treal>::abs(rhs.y))
    {
        Treal ratio = rhs.y / rhs.x;
        Treal scale = Treal(lhs) / (rhs.x + rhs.y * ratio);
        return {scale, -scale * ratio};
    }
    else
    {
        Treal ratio = rhs.x / rhs.y;
        Treal scale = Treal(lhs) / (rhs.x * ratio + rhs.y);
        return {ratio * scale, -scale};
    }
}

template <typename U, typename Treal>
__device__ __host__ constexpr bool operator==(const U& lhs, const rocfft_complex<Treal>& rhs)
{
    return Treal(lhs) == rhs.x && 0 == rhs.y;
}

template <typename U, typename Treal>
__device__ __host__ constexpr bool operator!=(const U& lhs, const rocfft_complex<Treal>& rhs)
{
    return !(lhs == rhs);
}

// Extending std namespace to handle rocfft_complex datatype
namespace std
{
    template <typename Treal>
    __device__ __host__ constexpr Treal real(const rocfft_complex<Treal>& z)
    {
        return z.x;
    }

    template <typename Treal>
    __device__ __host__ constexpr Treal imag(const rocfft_complex<Treal>& z)
    {
        return z.y;
    }

    template <typename Treal>
    __device__ __host__ constexpr rocfft_complex<Treal> conj(const rocfft_complex<Treal>& z)
    {
        return {z.x, -z.y};
    }

    template <typename Treal>
    __device__ __host__ inline Treal norm(const rocfft_complex<Treal>& z)
    {
        return (z.x * z.x) + (z.y * z.y);
    }

    template <typename Treal>
    __device__ __host__ inline Treal abs(const rocfft_complex<Treal>& z)
    {
        Treal tr = rocfft_complex<Treal>::abs(z.x), ti = rocfft_complex<Treal>::abs(z.y);
        return tr > ti ? (ti /= tr, tr * rocfft_complex<Treal>::sqrt(ti * ti + 1))
               : ti    ? (tr /= ti, ti * rocfft_complex<Treal>::sqrt(tr * tr + 1))
                       : 0;
    }
}

#endif // ROCFFT_COMPLEX_H

typedef rocfft_complex<float> scalar_type;
static const size_t           NUM_ELEMS = 101 * 200 * 200 * 10;

gpubuf_t<scalar_type> twiddles140;
gpubuf_t<scalar_type> twiddles195;
size_t                dim0;
size_t                dim1;
size_t                dim2;
gpubuf_t<size_t>      lengths0;
gpubuf_t<size_t>      lengths1;
gpubuf_t<size_t>      lengths2;
gpubuf_t<size_t>      stride_in_0;
gpubuf_t<size_t>      stride_out_0;
gpubuf_t<size_t>      stride1;
gpubuf_t<size_t>      stride2;
size_t                nbatch0;
size_t                nbatch1;
size_t                nbatch2;
unsigned int          lds_padding0;
unsigned int          lds_padding1;
unsigned int          lds_padding2;
gpubuf_t<scalar_type> buf_in;
gpubuf_t<scalar_type> buf_out;
gpubuf_t<scalar_type> buf_orig;
dim3                  gridDim0;
dim3                  gridDim1;
dim3                  gridDim2;
dim3                  blockDim0;
dim3                  blockDim1;
dim3                  blockDim2;
unsigned int          lds_bytes0;
unsigned int          lds_bytes1;
unsigned int          lds_bytes2;

RTCKernelArgs kargs0, kargs1, kargs2;
RTCKernel     rtc_kerenl0("fft_rtc_fwd_len100_factors_10_10_wgs_60_tpt_10_halfLds_sp_op_CI_CI_"
                          "unitstride_sbrr_R2C_dirReg");
RTCKernel     rtc_kerenl1("fft_rtc_fwd_len200_factors_5_8_5_wgs_400_tpt_40_sp_ip_CI_sbcc");
RTCKernel     rtc_kerenl2("fft_rtc_fwd_len200_factors_5_8_5_wgs_400_tpt_40_sp_ip_CI_sbcc");

void init_kernel()
{
    buf_orig = random_complex_device<scalar_type>(NUM_ELEMS);
    buf_in.alloc(NUM_ELEMS * sizeof(scalar_type));
    buf_out.alloc(NUM_ELEMS * sizeof(scalar_type));

    twiddles140  = random_complex_device<scalar_type>(140);
    dim0         = 1;
    lengths0     = host_vec_to_dev<size_t>({100});
    stride_in_0  = host_vec_to_dev<size_t>({1, 100});
    stride_out_0 = host_vec_to_dev<size_t>({1, 101});
    nbatch0      = 400000;
    lds_padding0 = 1;
    gridDim0     = {66667, 1, 1};
    blockDim0    = {60, 1, 1};
    lds_bytes0   = 4848;

    twiddles195  = random_complex_device<scalar_type>(195);
    dim1         = 3;
    lengths1     = host_vec_to_dev<size_t>({200, 101, 200});
    stride1      = host_vec_to_dev<size_t>({20200, 1, 101, 4040000});
    nbatch1      = 10;
    lds_padding1 = 0;
    gridDim1     = {22000, 1, 1};
    blockDim1    = {400, 1, 1};
    lds_bytes1   = 16000;

    // twiddles195  = random_complex_device<scalar_type>(195);
    dim2         = 2;
    lengths2     = host_vec_to_dev<size_t>({200, 101});
    stride2      = host_vec_to_dev<size_t>({101, 1, 20200});
    nbatch2      = 2000;
    lds_padding2 = 0;
    gridDim2     = {22000, 1, 1};
    blockDim2    = {400, 1, 1};
    lds_bytes2   = 16000;
}

void init_rtc(std::string const& gpu_arch)
{
    //-----------------------------------------------------------
    //kernel0

    // twiddles
    kargs0.append_ptr(twiddles140.data());

    // large 1D twiddles for CS_KERNEL_STOCKHAM_BLOCK_CC only
    //     kargs.append_ptr(data.node->twiddles_large);

    // if(!hardcoded_dim)
    {
        kargs0.append_size_t(lengths0.size());
    }
    // lengths
    kargs0.append_ptr(lengths0.data());
    // stride in/out
    kargs0.append_ptr(stride_in_0.data());
    kargs0.append_ptr(stride_out_0.data());
    // nbatch
    kargs0.append_size_t(nbatch0);
    // lds padding
    kargs0.append_unsigned_int(lds_bytes0);
    // callback params
    kargs0.append_ptr(nullptr);
    kargs0.append_ptr(nullptr);
    kargs0.append_unsigned_int(0);
    kargs0.append_ptr(nullptr);
    kargs0.append_ptr(nullptr);

    // buffer pointers
    kargs0.append_ptr(buf_in.data());
    kargs0.append_ptr(buf_out.data());

    rtc_kerenl0.gridDim  = gridDim0;
    rtc_kerenl0.blockDim = blockDim0;

    std::ifstream kernel0_file("fft_rtc_fwd_len100_factors_10_10_wgs_60_tpt_10_halfLds_sp_op_CI_CI_"
                               "unitstride_sbrr_R2C_dirReg.h");
    if(!kernel0_file.is_open())
    {
        std::cerr << "Can not open kernel0 source file!" << std::endl;
        return;
    }

    std::string kernel0_content((std::istreambuf_iterator<char>(kernel0_file)),
                                std::istreambuf_iterator<char>());
    kernel0_file.close();
    rtc_kerenl0.init(compile_inprocess(kernel0_content, gpu_arch));

    //-----------------------------------------------------------
    //kernel1

    // twiddles
    kargs1.append_ptr(twiddles195.data());

    // large 1D twiddles for CS_KERNEL_STOCKHAM_BLOCK_CC only
    kargs1.append_ptr(nullptr);

    // if(!hardcoded_dim)
    {
        kargs1.append_size_t(lengths1.size());
    }
    // lengths
    kargs1.append_ptr(lengths1.data());
    // stride in/out
    kargs1.append_ptr(stride1.data());
    // nbatch
    kargs1.append_size_t(nbatch1);
    // lds padding
    kargs1.append_unsigned_int(lds_bytes1);
    // callback params
    kargs1.append_ptr(nullptr);
    kargs1.append_ptr(nullptr);
    kargs1.append_unsigned_int(0);
    kargs1.append_ptr(nullptr);
    kargs1.append_ptr(nullptr);

    // buffer pointers
    kargs1.append_ptr(buf_in.data());
    kargs1.append_ptr(buf_out.data());

    rtc_kerenl1.gridDim  = gridDim1;
    rtc_kerenl1.blockDim = blockDim1;

    std::ifstream kernel1_file("fft_rtc_fwd_len200_factors_5_8_5_wgs_400_tpt_40_sp_ip_CI_sbcc.h");
    if(!kernel1_file.is_open())
    {
        std::cerr << "Can not open kernel1 source file!" << std::endl;
        return;
    }

    std::string kernel1_content((std::istreambuf_iterator<char>(kernel1_file)),
                                std::istreambuf_iterator<char>());
    kernel1_file.close();
    rtc_kerenl1.init(compile_inprocess(kernel1_content, gpu_arch));

    //-----------------------------------------------------------
    //kernel2

    // twiddles
    kargs2.append_ptr(twiddles195.data());

    // large 1D twiddles for CS_KERNEL_STOCKHAM_BLOCK_CC only
    kargs2.append_ptr(nullptr);

    // if(!hardcoded_dim)
    {
        kargs2.append_size_t(lengths2.size());
    }
    // lengths
    kargs2.append_ptr(lengths2.data());
    // stride in/out
    kargs2.append_ptr(stride2.data());
    // nbatch
    kargs2.append_size_t(nbatch2);
    // lds padding
    kargs2.append_unsigned_int(lds_bytes2);
    // callback params
    kargs2.append_ptr(nullptr);
    kargs2.append_ptr(nullptr);
    kargs2.append_unsigned_int(0);
    kargs2.append_ptr(nullptr);
    kargs2.append_ptr(nullptr);

    // buffer pointers
    kargs2.append_ptr(buf_in.data());
    kargs2.append_ptr(buf_out.data());

    rtc_kerenl2.gridDim  = gridDim2;
    rtc_kerenl2.blockDim = blockDim2;

    std::ifstream kernel2_file("fft_rtc_fwd_len200_factors_5_8_5_wgs_400_tpt_40_sp_ip_CI_sbcc.h");
    if(!kernel2_file.is_open())
    {
        std::cerr << "Can not open kernel2 source file!" << std::endl;
        return;
    }

    std::string kernel2_content((std::istreambuf_iterator<char>(kernel2_file)),
                                std::istreambuf_iterator<char>());
    kernel2_file.close();
    rtc_kerenl2.init(compile_inprocess(kernel2_content, gpu_arch));
}

// #include "fft_rtc_fwd_len100_factors_10_10_wgs_60_tpt_10_halfLds_sp_op_CI_CI_unitstride_sbrr_R2C_dirReg.h"
#include "fft_rtc_fwd_len200_factors_5_8_5_wgs_400_tpt_40_sp_ip_CI_sbcc.h"

// this kernel:
void launch_kernel0(bool isRTC = false)
{
    // if(isRTC)
    // {
    //     rtc_kerenl0.launch(kargs0, rtc_kerenl0.gridDim, rtc_kerenl0.blockDim, lds_bytes0, 0);
    // }
    // else
    // {
    //     fft_rtc_fwd_len100_factors_10_10_wgs_60_tpt_10_halfLds_sp_op_CI_CI_unitstride_sbrr_R2C_dirReg<<<
    //         gridDim0,
    //         blockDim0,
    //         lds_bytes0>>>(twiddles140.data(),
    //                       dim0,
    //                       lengths0.data(),
    //                       stride_in_0.data(),
    //                       stride_out_0.data(),
    //                       nbatch0,
    //                       lds_padding0,
    //                       nullptr,
    //                       nullptr,
    //                       0,
    //                       nullptr,
    //                       nullptr,
    //                       buf_in.data(),
    //                       buf_out.data());
    // }
}

void launch_kernel1(bool isRTC = false)
{
    if(isRTC)
    {
        rtc_kerenl1.launch(kargs1, rtc_kerenl1.gridDim, rtc_kerenl1.blockDim, lds_bytes1, 0);
    }
    else
    {
        fft_rtc_fwd_len200_factors_5_8_5_wgs_400_tpt_40_sp_ip_CI_sbcc<<<gridDim1,
                                                                        blockDim1,
                                                                        lds_bytes1>>>(
            twiddles195.data(),
            nullptr,
            dim1,
            lengths1.data(),
            stride1.data(),
            nbatch1,
            lds_padding1,
            nullptr,
            nullptr,
            0,
            nullptr,
            nullptr,
            buf_out.data());
    }
}

void launch_kernel2(bool isRTC = false)
{
    if(isRTC)
    {
        rtc_kerenl2.launch(kargs2, rtc_kerenl2.gridDim, rtc_kerenl2.blockDim, lds_bytes2, 0);
    }
    else
    {
        fft_rtc_fwd_len200_factors_5_8_5_wgs_400_tpt_40_sp_ip_CI_sbcc<<<gridDim2,
                                                                        blockDim2,
                                                                        lds_bytes2>>>(
            twiddles195.data(),
            nullptr,
            dim2,
            lengths2.data(),
            stride2.data(),
            nbatch2,
            lds_padding2,
            nullptr,
            nullptr,
            0,
            nullptr,
            nullptr,
            buf_out.data());
    }
}

int main()
{
    // initialize arguments, grid
    init_kernel();
    init_rtc("gfx90a");

    unsigned int num_trials = 10;

    std::vector<scalar_type> hostbuf_prev;
    std::vector<scalar_type> hostbuf(NUM_ELEMS);

    static const size_t BUF_BYTES = NUM_ELEMS * sizeof(scalar_type);

    hipEvent_t timer_start, timer_stop;
    hipEventCreate(&timer_start);
    hipEventCreate(&timer_stop);

    std::cout << "AOT-------------------\n";

    for(unsigned int trial = 0; trial < num_trials; ++trial)
    {
        // initialize input buffer, since each trial will mutate the
        // input buffer
        if(hipMemcpy(buf_in.data(), buf_orig.data(), BUF_BYTES, hipMemcpyDeviceToDevice)
           != hipSuccess)
            throw std::runtime_error("initial hipMemcpy failed");

        // For test sbcc only
        if(hipMemcpy(buf_out.data(), buf_orig.data(), BUF_BYTES, hipMemcpyDeviceToDevice)
           != hipSuccess)
            throw std::runtime_error("initial hipMemcpy failed");

        hipEventRecord(timer_start);

        // launch the kernels that make up the FFT
        launch_kernel0();
        launch_kernel1();
        launch_kernel2();

        hipEventRecord(timer_stop);
        hipEventSynchronize(timer_stop);
        float gpu_time;
        hipEventElapsedTime(&gpu_time, timer_start, timer_stop);
        std::cout << ", " << gpu_time;

        // copy results back to host for comparison
        if(hipMemcpy(hostbuf.data(), buf_out.data(), BUF_BYTES, hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy of results failed");
    }

    std::cout << " ms\n";

    std::cout << "RTC-------------------\n";

    for(unsigned int trial = 0; trial < num_trials; ++trial)
    {
        // initialize input buffer, since each trial will mutate the
        // input buffer
        if(hipMemcpy(buf_in.data(), buf_orig.data(), BUF_BYTES, hipMemcpyDeviceToDevice)
           != hipSuccess)
            throw std::runtime_error("initial hipMemcpy failed");

        // For test sbcc only
        if(hipMemcpy(buf_out.data(), buf_orig.data(), BUF_BYTES, hipMemcpyDeviceToDevice)
           != hipSuccess)
            throw std::runtime_error("initial hipMemcpy failed");

        hipEventRecord(timer_start);

        // launch the kernels that make up the FFT
        launch_kernel0(true);
        launch_kernel1(true);
        launch_kernel2(true);

        hipEventRecord(timer_stop);
        hipEventSynchronize(timer_stop);
        float gpu_time;
        hipEventElapsedTime(&gpu_time, timer_start, timer_stop);
        std::cout << ", " << gpu_time;

        // copy results back to host for comparison
        if(hipMemcpy(hostbuf.data(), buf_out.data(), BUF_BYTES, hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy of results failed");
    }

    std::cout << " ms\n";

    hipEventDestroy(timer_start);
    hipEventDestroy(timer_stop);

    std::cout << num_trials << "trials finished successfully.\n";
    return 0;
}

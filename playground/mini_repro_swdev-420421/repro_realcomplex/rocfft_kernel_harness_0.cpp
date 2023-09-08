#include <hip/hip_runtime.h>

#include <fstream>
#include <functional>
#include <future>
#include <hip/hip_runtime_api.h>
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
    gpubuf_t(const gpubuf_t&) = delete;
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
    __device__ __host__ rocfft_complex()                      = default;
    __device__ __host__ rocfft_complex(const rocfft_complex&) = default;
    __device__ __host__ rocfft_complex(rocfft_complex&&)      = default;
    __device__ __host__ rocfft_complex& operator=(const rocfft_complex& rhs) & = default;
    __device__ __host__ rocfft_complex& operator=(rocfft_complex&& rhs) & = default;
    __device__                          __host__ ~rocfft_complex()        = default;

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
static const size_t           NUM_ELEMS = 129 * 128 * 256;

gpubuf_t<scalar_type> twiddles176;
gpubuf_t<scalar_type> twiddles248;
gpubuf_t<scalar_type> twiddles112;
size_t                dim0;
size_t                dim1;
size_t                dim2;
gpubuf_t<size_t>      lengths0;
gpubuf_t<size_t>      lengths1;
gpubuf_t<size_t>      lengths2;
gpubuf_t<size_t>      stride0;
gpubuf_t<size_t>      stride1;
gpubuf_t<size_t>      stride2;
size_t                nbatch0;
size_t                nbatch1;
size_t                nbatch2;
unsigned int          lds_padding0;
unsigned int          lds_padding1;
unsigned int          lds_padding2;
gpubuf_t<scalar_type> buf;
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
void                  init_kernel()
{
    buf_orig = random_complex_device<scalar_type>(NUM_ELEMS);
    buf.alloc(NUM_ELEMS * sizeof(scalar_type));

    twiddles176  = random_complex_device<scalar_type>(176);
    dim0         = 1;
    lengths0     = host_vec_to_dev<size_t>({128});
    stride0      = host_vec_to_dev<size_t>({1, 129});
    nbatch0      = 32768;
    lds_padding0 = 1;
    gridDim0     = {2048, 1, 1};
    blockDim0    = {256, 1, 1};
    lds_bytes0   = 16512;

    twiddles248  = random_complex_device<scalar_type>(248);
    dim1         = 3;
    lengths1     = host_vec_to_dev<size_t>({256, 129, 128});
    stride1      = host_vec_to_dev<size_t>({16512, 1, 129, 4227072});
    nbatch1      = 1;
    lds_padding1 = 0;
    gridDim1     = {2176, 1, 1};
    blockDim1    = {256, 1, 1};
    lds_bytes1   = 16384;

    twiddles112  = random_complex_device<scalar_type>(112);
    dim2         = 2;
    lengths2     = host_vec_to_dev<size_t>({128, 129});
    stride2      = host_vec_to_dev<size_t>({129, 1, 16512});
    nbatch2      = 256;
    lds_padding2 = 0;
    gridDim2     = {2304, 1, 1};
    blockDim2    = {256, 1, 1};
    lds_bytes2   = 16384;
}

#include "fft_rtc_fwd_len128_factors_16_8_wgs_256_tpt_16_halfLds_sp_ip_CI_unitstride_sbrr_R2C_dirReg.h"
#include "fft_rtc_fwd_len128_factors_16_8_wgs_256_tpt_16_sp_ip_CI_sbcc_dirReg.h"
#include "fft_rtc_fwd_len256_factors_8_4_8_wgs_256_tpt_32_sp_ip_CI_sbcc_dirReg.h"

void launch_kernel0()
{
    fft_rtc_fwd_len128_factors_16_8_wgs_256_tpt_16_halfLds_sp_ip_CI_unitstride_sbrr_R2C_dirReg<<<
        gridDim0,
        blockDim0,
        lds_bytes0>>>(twiddles176.data(),
                      dim0,
                      lengths0.data(),
                      stride0.data(),
                      nbatch0,
                      lds_padding0,
                      nullptr,
                      nullptr,
                      0,
                      nullptr,
                      nullptr,
                      buf.data());
}

void launch_kernel1()
{
    fft_rtc_fwd_len256_factors_8_4_8_wgs_256_tpt_32_sp_ip_CI_sbcc_dirReg<<<gridDim1,
                                                                           blockDim1,
                                                                           lds_bytes1>>>(
        twiddles248.data(),
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
        buf.data());
}

void launch_kernel2()
{
    fft_rtc_fwd_len128_factors_16_8_wgs_256_tpt_16_sp_ip_CI_sbcc_dirReg<<<gridDim2,
                                                                          blockDim2,
                                                                          lds_bytes2>>>(
        twiddles112.data(),
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
        buf.data());
}

void check_diff(const std::vector<scalar_type>& buf0,
                const std::vector<scalar_type>& buf1,
                unsigned int                    trial)
{
    static const double single_epsilon = 3.75e-5;

    std::mutex          diff_failures_lock;
    std::vector<size_t> diff_failures;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(size_t i = 0; i < buf0.size(); ++i)
    {
        if(std::abs(buf0[i].real() - buf1[i].real()) > single_epsilon
           || std::abs(buf0[i].imag() - buf1[i].imag()) > single_epsilon)
        {
            diff_failures_lock.lock();
            diff_failures.push_back(i);
            diff_failures_lock.unlock();
        }
    }
    if(!diff_failures.empty())
    {
        std::sort(diff_failures.begin(), diff_failures.end());

        printf("trial %u failure indexes:\n", trial);
        for(auto f : diff_failures)
        {
            printf(" %zu", f);
        }
        puts("");
        fflush(stdout);
        throw std::runtime_error("diffs present between runs");
    }
}

int main()
{
    // initialize arguments, grid
    init_kernel();
    unsigned int num_trials = 100;

    std::vector<scalar_type> hostbuf_prev;
    std::vector<scalar_type> hostbuf(NUM_ELEMS);

    static const size_t BUF_BYTES = NUM_ELEMS * sizeof(scalar_type);

    for(unsigned int trial = 0; trial < num_trials; ++trial)
    {
        // initialize input buffer, since each trial will mutate the
        // input buffer
        if(hipMemcpy(buf.data(), buf_orig.data(), BUF_BYTES, hipMemcpyDeviceToDevice) != hipSuccess)
            throw std::runtime_error("initial hipMemcpy failed");

        // launch the kernels that make up the FFT
        launch_kernel0();
        launch_kernel1();
        launch_kernel2();

        // copy results back to host for comparison
        if(hipMemcpy(hostbuf.data(), buf.data(), BUF_BYTES, hipMemcpyDeviceToHost) != hipSuccess)
            throw std::runtime_error("hipMemcpy of results failed");

        // save results and compare against previous trial
        if(hostbuf_prev.empty())
        {
            hostbuf_prev.resize(NUM_ELEMS);
        }
        else
        {
            check_diff(hostbuf_prev, hostbuf, trial);
        }
        hostbuf.swap(hostbuf_prev);
    }

    printf("%u trials finished successfully.\n", num_trials);
    return 0;
}

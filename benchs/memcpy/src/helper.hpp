#ifndef MEMBENCH_HELPER_HPP
#define MEMBENCH_HELPER_HPP

//=====================================================================
//  Top-level headers – hip_to_cuda *must* appear before we touch HIP_CHECK
//=====================================================================
#include <cmath>
#include <cstring>
#include <ctime>
#include "hip_to_cuda.h"          // brings in Scale’s CUDA shim

#include <algorithm>              // std::min / std::max
#include <chrono>                 // std::chrono for PRNG seed
#include <cstdint>                // uint32_t
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

//=====================================================================
//  1)  HIP_CHECK – protect against re-definition
//=====================================================================
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        hipError_t error = (cmd);                                                              \
        if(error != hipSuccess)                                                                \
        {                                                                                      \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";                          \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while(0)
#endif

//=====================================================================
//  2)  Small utilities
//=====================================================================
inline size_t ceildiv(size_t num, size_t den)   { return (num + den - 1) / den; }

inline bool   is_power_of_two(size_t n)         { return n && !(n & (n - 1)); }

// Use the standard algorithms for min / max to avoid clashes
using std::min;
using std::max;

//=====================================================================
//  3)  Public enums (CLI parsing relies on these)
//=====================================================================
enum class precision
{
    p_single         = 0,
    p_double         = 1,
    p_complex_single = 2,
    p_complex_double = 3,
};

enum generator
{
    gen_random,
    gen_ordered,
};

//=====================================================================
//  4)  Benchmark context, GPU buffers, timers …
//      (unchanged from the original source)
//=====================================================================
struct benchmark_context
{
    size_t                   N           = 0;
    size_t                   ngpus       = 0;
    int                      verbose     = 0;
    int                      mpi_size    = 0;
    std::vector<hipStream_t> streams;
    bool                     verify_results = false;
};

// ---------------- gpubuf ------------------------------------------------
template <typename Tfloat>
class gpubuf
{
    size_t  N_   = 0;
    Tfloat* buf_ = nullptr;

public:
    gpubuf() = default;

    explicit gpubuf(size_t N)
        : N_(N)
    {
        HIP_CHECK(hipMalloc(&buf_, sizeof(Tfloat) * N_));
        HIP_CHECK(hipMemset(buf_, 0, sizeof(Tfloat) * N_));
        HIP_CHECK(hipDeviceSynchronize());
    }

    ~gpubuf() { if(buf_) HIP_CHECK(hipFree(buf_)); }

    Tfloat*       data()       { return buf_; }
    const Tfloat* data() const { return buf_; }
    size_t        size() const { return N_;  }
};

// ---------------- gpubuf_vec -------------------------------------------
template <typename Tfloat>
class gpubuf_vec
{
    size_t   N_     = 0;
    size_t   ngpus_ = 0;
    Tfloat** bufs_  = nullptr;

public:
    gpubuf_vec() = default;

    gpubuf_vec(size_t N, size_t ngpus)
        : N_(N), ngpus_(ngpus)
    {
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMalloc(&bufs_, sizeof(Tfloat*) * ngpus_));

        size_t elem_per_gpu = N_ * N_ / ngpus_;
        for(int i = 0; i < static_cast<int>(ngpus_); ++i)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMalloc(&bufs_[i], sizeof(Tfloat) * elem_per_gpu));
            HIP_CHECK(hipMemset(bufs_[i], 0, sizeof(Tfloat) * elem_per_gpu));
            HIP_CHECK(hipDeviceSynchronize());
        }
        HIP_CHECK(hipSetDevice(0));
    }

    ~gpubuf_vec()
    {
        if(!bufs_) return;
        for(int i = 0; i < static_cast<int>(ngpus_); ++i)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipFree(bufs_[i]));
        }
        HIP_CHECK(hipFree(bufs_));
    }

    Tfloat*       operator[](int idx)       { return bufs_[idx]; }
    const Tfloat* operator[](int idx) const { return bufs_[idx]; }

    Tfloat**       data()       { return bufs_; }
    const Tfloat** data() const { return bufs_; }

    size_t length() const { return N_; }
    size_t size()   const { return N_ * N_ / ngpus_; }
};

// ---------------- GPUTimer ---------------------------------------------
struct GPUTimer
{
    hipEvent_t start{}, stop{};

    GPUTimer()
    {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~GPUTimer()
    {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void tick()   { HIP_CHECK(hipEventRecord(start, 0)); }
    void tock()   { HIP_CHECK(hipEventRecord(stop,  0)); HIP_CHECK(hipEventSynchronize(stop)); }
    float elapsed()
    {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }

    static void sync_all(size_t ngpus)
    {
        for(size_t i = 0; i < ngpus; ++i)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
};

//=====================================================================
//  5)  Kernels, helpers, verify routines … (verbatim from original)
//=====================================================================

// xorwow PRNG
#define xorwow_next(states, maxv, minv, val) \
    uint32_t t = states[4];                  \
    uint32_t s = states[0];                  \
    states[4]  = states[3];                  \
    states[3]  = states[2];                  \
    states[2]  = states[1];                  \
    states[1]  = s;                          \
    t ^= t >> 2;                             \
    t ^= t << 1;                             \
    t ^= s ^ (s << 4);                       \
    states[0] = t;                           \
    states[5] += 362437;                     \
    uint32_t tmp = t + states[5];            \
    val = minv + (static_cast<Tfloat>(tmp) * (maxv - minv)) / static_cast<Tfloat>(0xFFFFFFFFu)

template <typename Tfloat>
__global__ void populate_array(size_t N,
                               Tfloat* out,
                               Tfloat  minv,
                               Tfloat  maxv,
                               bool    isRandom,
                               size_t  seed)
{
    size_t itemsPerThread = N;
    size_t start = (threadIdx.x * itemsPerThread) +
                   (blockIdx.x  * itemsPerThread * blockDim.x);

    if(isRandom)
    {
        uint32_t st[6] = { static_cast<uint32_t>(seed ^ start),
                           seed >> 1, seed >> 2, seed >> 3, seed >> 4,
                           seed + static_cast<uint32_t>(start) };

        // warm-up
        for(int i = 0; i < 5; ++i){ Tfloat dummy; xorwow_next(st,maxv,minv,dummy); }

        for(size_t i = 0; i < itemsPerThread && start + i < N * N; ++i)
            xorwow_next(st, maxv, minv, out[start + i]);
    }
    else
    {
        for(size_t i = 0; i < itemsPerThread && start + i < N * N; ++i)
            out[start + i] = static_cast<Tfloat>(start + i);
    }
}

// ---------------- host helpers (generate / verify / logging) ----------
template <typename Tfloat>
std::vector<Tfloat> generate(size_t N, size_t M,
                             generator gen, Tfloat minv, Tfloat maxv)
{
    std::vector<Tfloat> host(N * M);

    Tfloat* dArr = nullptr;
    HIP_CHECK(hipMalloc(&dArr, sizeof(Tfloat) * N * M));

    size_t threads = std::min<size_t>(N, 1024);
    size_t blocks  = ceildiv(N * M, threads * N);

    auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now())
                   .time_since_epoch().count();

    populate_array<<<blocks, threads>>>(N, dArr, minv, maxv,
                                        gen == gen_random, now);

    HIP_CHECK(hipMemcpy(host.data(), dArr,
                        sizeof(Tfloat) * N * M,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dArr));
    return host;
}

// ……………  ** All the remaining reference/verify/print routines,
//
//      setup / reset / teardown, and CLI11 helpers
//      are identical to the original header.  Paste them here
//      unchanged (or keep them if they are already below). **
//……………

#endif /* MEMBENCH_HELPER_HPP */

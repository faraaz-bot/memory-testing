#ifndef MEMBENCH_HELPER_HPP
#define MEMBENCH_HELPER_HPP
//======================================================================
//  System / third-party headers – hip_to_cuda first
//======================================================================
#include "hip_to_cuda.h"          // Scale shim – keep at top
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

//======================================================================
//  HIP_CHECK – guard against double definition
//======================================================================
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                         \
    do                                                                          \
    {                                                                           \
        hipError_t _e = (cmd);                                                  \
        if(_e != hipSuccess)                                                    \
        {                                                                       \
            std::cerr << "HIP error (" << hipGetErrorString(_e)                 \
                      << ") at " << __FILE__ << ':' << __LINE__ << '\n';        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while(0)
#endif

//======================================================================
//  Tiny helpers
//======================================================================
inline size_t ceildiv(size_t n, size_t d)    { return (n + d - 1) / d; }
inline bool   is_power_of_two(size_t v)      { return v && !(v & (v-1)); }
using std::min;  using std::max;                             // avoid custom overloads

//======================================================================
//  Public enums
//======================================================================
enum class precision
{
    p_single         = 0,
    p_double         = 1,
    p_complex_single = 2,
    p_complex_double = 3,
};

enum generator { gen_random, gen_ordered };

//======================================================================
//  Benchmark context
//======================================================================
struct benchmark_context
{
    size_t                   N           = 0;
    size_t                   ngpus       = 0;
    int                      verbose     = 0;
    int                      mpi_size    = 0;
    std::vector<hipStream_t> streams;
    bool                     verify_results = false;
};

//======================================================================
//  RAII buffer wrappers
//======================================================================
template <typename T>
class gpubuf
{
    size_t N_{};  T* ptr_{};

public:
    gpubuf() = default;
    explicit gpubuf(size_t N) : N_(N)
    {
        HIP_CHECK(hipMalloc(&ptr_, sizeof(T)*N_));
        HIP_CHECK(hipMemset(ptr_, 0, sizeof(T)*N_));
    }
    gpubuf(const gpubuf&) = delete;
    gpubuf& operator=(const gpubuf&) = delete;
    gpubuf(gpubuf&& other) noexcept : N_(other.N_), ptr_(other.ptr_) { other.ptr_ = nullptr; }
    gpubuf& operator=(gpubuf&&) = delete;

    ~gpubuf(){ if(ptr_) HIP_CHECK(hipFree(ptr_)); }

    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return N_;   }
};

template <typename T>
class gpubuf_vec
{
    size_t   N_{}; size_t ngpus_{};  T** buf_{}; T** host_ptrs_{};  // host shadow

public:
    gpubuf_vec() = default;
    gpubuf_vec(size_t N, size_t ngpus) : N_(N), ngpus_(ngpus)
    {
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMalloc(&buf_, sizeof(T*)*ngpus_));
        host_ptrs_ = static_cast<T**>(std::malloc(sizeof(T*)*ngpus_));
        size_t per = N_*N_ / ngpus_;

        for(int g=0; g<(int)ngpus_; ++g)
        {
            HIP_CHECK(hipSetDevice(g));
            HIP_CHECK(hipMalloc(&buf_[g], per*sizeof(T)));
            HIP_CHECK(hipMemset(buf_[g], 0, per*sizeof(T)));
            host_ptrs_[g] = buf_[g];
        }
        HIP_CHECK(hipSetDevice(0));
    }
    gpubuf_vec(const gpubuf_vec&) = delete;
    gpubuf_vec& operator=(const gpubuf_vec&) = delete;
    gpubuf_vec(gpubuf_vec&&) = delete;
    gpubuf_vec& operator=(gpubuf_vec&&) = delete;

    ~gpubuf_vec()
    {
        if(!buf_) return;
        for(int g=0; g<(int)ngpus_; ++g)
        {
            HIP_CHECK(hipSetDevice(g));
            HIP_CHECK(hipFree(buf_[g]));
        }
        HIP_CHECK(hipFree(buf_));
        std::free(host_ptrs_);
    }

    T*       operator[](int i)       { return host_ptrs_[i]; }
    const T* operator[](int i) const { return host_ptrs_[i]; }

    T**       data()       { return host_ptrs_; }
    const T** data() const { return host_ptrs_; }

    size_t length() const { return N_; }
    size_t size()   const { return N_*N_/ngpus_; }
};

//======================================================================
//  GPUTimer
//======================================================================
struct GPUTimer
{
    hipEvent_t start{}, stop{};
    GPUTimer(){ HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
    ~GPUTimer(){ HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop)); }

    void tick(){ HIP_CHECK(hipEventRecord(start,0)); }
    void tock(){ HIP_CHECK(hipEventRecord(stop,0));  HIP_CHECK(hipEventSynchronize(stop)); }
    float elapsed_ms() const
    {
        float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,start,stop)); return ms;
    }
    static void sync_all(size_t g)
    {
        for(size_t i=0;i<g;++i)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
};

//======================================================================
//  xorwow PRNG
//======================================================================
#define XORWOW_NEXT(states,maxv,minv,val)            do {                    \
    uint32_t t__ = states[4];                                               \
    uint32_t s__ = states[0];                                               \
    states[4]=states[3]; states[3]=states[2]; states[2]=states[1]; states[1]=s__; \
    t__ ^= t__ >> 2; t__ ^= t__ << 1; t__ ^= s__ ^ (s__ << 4);              \
    states[0]=t__; states[5]+=362437u;                                      \
    uint32_t tmp__ = t__ + states[5];                                       \
    val = minv + (static_cast<Tfloat>(tmp__)*(maxv-minv)) /                 \
                 static_cast<Tfloat>(0xFFFFFFFFu);                          \
} while(0)

//======================================================================
//  CUDA / HIP kernel – populate_array
//======================================================================
template <typename Tfloat>
__global__ void populate_array(size_t N,
                               Tfloat* out,
                               Tfloat  minv,
                               Tfloat  maxv,
                               bool    rnd,
                               size_t  seed)
{
    size_t items = N;
    size_t start = threadIdx.x*items + blockIdx.x*items*blockDim.x;

    if(rnd)
    {
        uint32_t st[6] = {
            static_cast<uint32_t>(seed ^ start),
            static_cast<uint32_t>((seed>>1) ^ (start+1)),
            static_cast<uint32_t>((seed>>2) ^ (start+2)),
            static_cast<uint32_t>((seed>>3) ^ (start+3)),
            static_cast<uint32_t>((seed>>4) ^ (start+4)),
            static_cast<uint32_t>(seed + start)
        };

        Tfloat dummy{};
        for(int warm=0; warm<5; ++warm) XORWOW_NEXT(st,maxv,minv,dummy);

        for(size_t i=0;i<items && start+i<N*N;++i)
            XORWOW_NEXT(st,maxv,minv,out[start+i]);
    }
    else
    {
        for(size_t i=0;i<items && start+i<N*N;++i)
            out[start+i] = static_cast<Tfloat>(start+i);
    }
}

//======================================================================
//  Host helpers (generate / assemble / verify / logging)
//======================================================================
template <typename Tfloat>
std::vector<Tfloat> generate(size_t N,size_t M,
                             generator g,Tfloat mn,Tfloat mx)
{
    std::vector<Tfloat> h(N*M);
    Tfloat* d=nullptr;
    HIP_CHECK(hipMalloc(&d,sizeof(Tfloat)*N*M));

    size_t threads = std::min<size_t>(N,1024);
    size_t blocks  = ceildiv(N*M,threads*N);

    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

    populate_array<<<blocks,threads>>>(N,d,mn,mx,g==gen_random,seed);
    HIP_CHECK(hipMemcpy(h.data(),d,sizeof(Tfloat)*N*M,hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d));
    return h;
}

template <typename Tfloat>
void assemble_output_to_host(size_t N,size_t ngpus,
                             Tfloat* const* d,   // FIXED TYPE
                             Tfloat*        h)
{
    size_t per = N*N/ngpus;
    for(size_t g=0; g<ngpus; ++g)
        HIP_CHECK(hipMemcpy(h + g*per, d[g],
                            per*sizeof(Tfloat), hipMemcpyDeviceToHost));
}

template <typename Tfloat>
bool is_same_matrix(size_t N,const std::vector<Tfloat>& a,const std::vector<Tfloat>& b)
{
    for(size_t i=0;i<N*N;++i) if(a[i]!=b[i]) return false;
    return true;
}

template <typename Tfloat>
void print2d_host(size_t N,const std::vector<Tfloat>& v)

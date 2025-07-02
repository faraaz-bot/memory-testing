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
//  HIP_CHECK – guard against double-definition
//======================================================================
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        hipError_t _e = (cmd);                                                                 \
        if(_e != hipSuccess)                                                                   \
        {                                                                                      \
            std::cerr << "HIP error (" << hipGetErrorString(_e)                                \
                      << ") at " << __FILE__ << ':' << __LINE__ << '\n';                       \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
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
    ~gpubuf(){ if(ptr_) HIP_CHECK(hipFree(ptr_)); }

    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return N_;   }
};

template <typename T>
class gpubuf_vec
{
    size_t   N_{}; size_t ngpus_{};  T** buf_{};

public:
    gpubuf_vec() = default;
    gpubuf_vec(size_t N, size_t ngpus) : N_(N), ngpus_(ngpus)
    {
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMalloc(&buf_, sizeof(T*)*ngpus_));
        size_t per = N_*N_/ngpus_;
        for(int g=0; g<(int)ngpus_; ++g)
        {
            HIP_CHECK(hipSetDevice(g));
            HIP_CHECK(hipMalloc(&buf_[g], per*sizeof(T)));
            HIP_CHECK(hipMemset(buf_[g], 0, per*sizeof(T)));
        }
        HIP_CHECK(hipSetDevice(0));
    }
    ~gpubuf_vec()
    {
        if(!buf_) return;
        for(int g=0; g<(int)ngpus_; ++g){ HIP_CHECK(hipSetDevice(g)); HIP_CHECK(hipFree(buf_[g])); }
        HIP_CHECK(hipFree(buf_));
    }

    T*       operator[](int i)       { return buf_[i]; }
    const T* operator[](int i) const { return buf_[i]; }
    T**       data()       { return buf_; }
    const T** data() const { return buf_; }
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
    static void sync_all(size_t g){ for(size_t i=0;i<g;++i){ HIP_CHECK(hipSetDevice(i)); HIP_CHECK(hipDeviceSynchronize()); } }
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
//  CUDA / HIP kernels
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
//  Host-side helpers  (generate / assemble / verify / logging)
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
void assemble_output_to_host(size_t N,size_t ngpus,Tfloat** d,Tfloat* h)
{
    size_t per = N*N/ngpus;
    for(size_t g=0; g<ngpus; ++g)
        HIP_CHECK(hipMemcpy(h+g*per, d[g], per*sizeof(Tfloat), hipMemcpyDeviceToHost));
}

template <typename Tfloat>
bool is_same_matrix(size_t N,const std::vector<Tfloat>& a,const std::vector<Tfloat>& b)
{
    for(size_t i=0;i<N*N;++i) if(a[i]!=b[i]) return false;
    return true;
}

template <typename Tfloat>
void print2d_host(size_t N,const std::vector<Tfloat>& v)
{
    std::cout<<"[\n";
    for(size_t r=0;r<N;++r)
    {
        std::cout<<"  [ ";
        for(size_t c=0;c<N;++c) std::cout<<std::setw(6)<<v[r*N+c]<<' ';
        std::cout<<"]\n";
    }
    std::cout<<"]\n";
}

//----------------  NEW: log_matrices  ---------------------------------
template <typename Tfloat>
void log_matrices(const benchmark_context& ctx,
                  const std::vector<Tfloat>& original_input,
                  gpubuf_vec<Tfloat>&        device_output,
                  std::vector<Tfloat>&       assembled_output)
{
    assemble_output_to_host<Tfloat>(
        ctx.N, ctx.ngpus, device_output.data(), assembled_output.data());

    std::cout << "Original Input:\n";
    print2d_host<Tfloat>(ctx.N, original_input);

    std::cout << "------------------------\nDevice Side Computation:\n";
    print2d_host<Tfloat>(ctx.N, assembled_output);
}
//-----------------------------------------------------------------------

//======================================================================
//  I/O helpers (setup / reset / teardown) – unchanged
//======================================================================
template <typename Tfloat>
void setup(size_t N,size_t ngpus,
           gpubuf_vec<Tfloat>& in,
           gpubuf_vec<Tfloat>& out,
           const std::vector<Tfloat>& h,
           std::vector<hipStream_t>& streams)
{
    size_t per = N*N/ngpus;
    in   = gpubuf_vec<Tfloat>(N,ngpus);
    out  = gpubuf_vec<Tfloat>(N,ngpus);
    streams.resize(ngpus*ngpus);

    for(size_t g=0; g<ngpus; ++g)
    {
        HIP_CHECK(hipSetDevice(g));
        HIP_CHECK(hipMemcpy(in[g], h.data()+g*per, per*sizeof(Tfloat), hipMemcpyHostToDevice));
        for(size_t s=0;s<ngpus;++s) HIP_CHECK(hipStreamCreate(&streams[g*ngpus+s]));
    }
    HIP_CHECK(hipSetDevice(0));
}

template <typename Tfloat>
void reset(size_t N,size_t ngpus,
           gpubuf_vec<Tfloat>& out,
           std::vector<Tfloat>& h)
{
    size_t per=N*N/ngpus;
    for(size_t g=0;g<ngpus;++g)
    {
        HIP_CHECK(hipSetDevice(g));
        HIP_CHECK(hipMemset(out[g],0,per*sizeof(Tfloat)));
    }
    std::fill(h.begin(),h.end(),0);
}

template <typename Tfloat>
void teardown(size_t ngpus,
              gpubuf_vec<Tfloat>& in,
              gpubuf_vec<Tfloat>& out,
              std::vector<hipStream_t>& streams)
{
    for(size_t g=0; g<ngpus; ++g)
    {
        HIP_CHECK(hipSetDevice(g));
        for(size_t s=0;s<ngpus;++s) HIP_CHECK(hipStreamDestroy(streams[g*ngpus+s]));
    }
    streams.clear();
    in  = gpubuf_vec<Tfloat>();
    out = gpubuf_vec<Tfloat>();
}

//======================================================================
//  Device-side print helper (for verbose debugging)
//======================================================================
template <typename Tfloat>
__global__ void print2d(int rows,int cols,const Tfloat* d)
{
    printf("[\n");
    for(int r=0;r<rows;++r)
    {
        printf("  [ ");
        for(int c=0;c<cols;++c) printf("%6.3f ", (double)d[r*cols+c]);
        printf("]\n");
    }
    printf("]\n");
}

//======================================================================
//  CLI lexical cast helpers
//======================================================================
inline bool lexical_cast(const std::string& w, precision& p)
{
    if(w=="single"||w=="0")            p=precision::p_single;
    else if(w=="double"||w=="1")       p=precision::p_double;
    else if(w=="c_single"||w=="2")     p=precision::p_complex_single;
    else if(w=="c_double"||w=="3")     p=precision::p_complex_double;
    else throw std::runtime_error("invalid precision");
    return true;
}
inline bool lexical_cast(const std::string& w, generator& g)
{
    if(w=="random"||w=="0")        g=gen_random;
    else if(w=="ordered"||w=="1")  g=gen_ordered;
    else throw std::runtime_error("invalid generator");
    return true;
}

#endif /* MEMBENCH_HELPER_HPP */

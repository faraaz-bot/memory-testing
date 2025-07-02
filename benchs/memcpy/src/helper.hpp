#ifndef MEMBENCH_HELPER_HPP
#define MEMBENCH_HELPER_HPP
//──────────────────────────────────────────────────────────────────────
// 1.  Headers – keep hip_to_cuda.h first
//──────────────────────────────────────────────────────────────────────
#include "hip_to_cuda.h"
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

//──────────────────────────────────────────────────────────────────────
// 2.  HIP_CHECK  (protected against double definition)
//──────────────────────────────────────────────────────────────────────
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                       \
    do {                                                                     \
        hipError_t _e = (cmd);                                               \
        if(_e != hipSuccess) {                                               \
            std::cerr << "HIP error (" << hipGetErrorString(_e)              \
                      << ") at " << __FILE__ << ':' << __LINE__ << '\n';     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while(0)
#endif

// tiny helpers
inline size_t ceildiv(size_t n,size_t d){ return (n+d-1)/d; }
inline bool   is_power_of_two(size_t v){ return v && !(v&(v-1)); }
using std::min;  using std::max;

//──────────────────────────────────────────────────────────────────────
// 3.  Public enums / context
//──────────────────────────────────────────────────────────────────────
enum class precision { p_single, p_double, p_complex_single, p_complex_double };
enum generator { gen_random, gen_ordered };

struct benchmark_context
{
    size_t                   N{};
    size_t                   ngpus{};
    int                      verbose{};
    int                      mpi_size{};
    std::vector<hipStream_t> streams;
    bool                     verify_results{};
};

//──────────────────────────────────────────────────────────────────────
// 4.  Simple RAII buffer
//──────────────────────────────────────────────────────────────────────
template <typename T>
class gpubuf
{
    size_t N_{};  T* ptr_{};
public:
    gpubuf() = default;
    explicit gpubuf(size_t N): N_(N){
        HIP_CHECK(hipMalloc(&ptr_,sizeof(T)*N_));
        HIP_CHECK(hipMemset(ptr_,0,sizeof(T)*N_));
    }
    ~gpubuf(){ if(ptr_) HIP_CHECK(hipFree(ptr_)); }

    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return N_;   }
};

/*─────────────────────────────────────────────────────────────────────
 5.  gpubuf_vec – **dual pointer table**
     host_ptrs_ : ordinary RAM (CPU code uses it)
     dev_ptrs_  : device memory  (kernels use it)
─────────────────────────────────────────────────────────────────────*/
template <typename T>
class gpubuf_vec
{
    size_t  N_{};  size_t ngpus_{};
    T** host_ptrs_{};   // CPU-side
    T** dev_ptrs_{};    // GPU-side (copied from host_ptrs_)

public:
    gpubuf_vec() = default;

    gpubuf_vec(size_t N,size_t ngpus): N_(N), ngpus_(ngpus)
    {
        host_ptrs_ = static_cast<T**>(std::malloc(sizeof(T*)*ngpus_));
        if(!host_ptrs_) throw std::bad_alloc();

        HIP_CHECK(hipMalloc(&dev_ptrs_, sizeof(T*)*ngpus_));

        size_t per = N_*N_/ngpus_;
        for(int g=0; g<(int)ngpus_; ++g)
        {
            HIP_CHECK(hipSetDevice(g));
            HIP_CHECK(hipMalloc(&host_ptrs_[g], per*sizeof(T)));
            HIP_CHECK(hipMemset(host_ptrs_[g], 0, per*sizeof(T)));
        }
        HIP_CHECK(hipMemcpy(dev_ptrs_, host_ptrs_,
                            sizeof(T*)*ngpus_, hipMemcpyHostToDevice));
        HIP_CHECK(hipSetDevice(0));
    }

    ~gpubuf_vec()
    {
        if(!host_ptrs_) return;
        for(int g=0; g<(int)ngpus_; ++g){
            HIP_CHECK(hipSetDevice(g));
            HIP_CHECK(hipFree(host_ptrs_[g]));
        }
        HIP_CHECK(hipFree(dev_ptrs_));
        std::free(host_ptrs_);
    }

    /* --------------- accessors ---------------- */
    T*       operator[](int i)       { return host_ptrs_[i]; }
    const T* operator[](int i) const { return host_ptrs_[i]; }

    /* what kernels expect */
    T**       data()       { return dev_ptrs_; }
    const T** data() const { return dev_ptrs_; }

    /* what host helpers need */
    T** host_ptrs()       { return host_ptrs_; }
    const T** host_ptrs() const { return host_ptrs_; }

    size_t length() const { return N_; }
    size_t size()   const { return N_*N_/ngpus_; }
};

//──────────────────────────────────────────────────────────────────────
// 6.  GPUTimer – unchanged
//──────────────────────────────────────────────────────────────────────
struct GPUTimer
{
    hipEvent_t start{}, stop{};
    GPUTimer(){ HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
    ~GPUTimer(){ HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop)); }

    void tick(){ HIP_CHECK(hipEventRecord(start,0)); }
    void tock(){ HIP_CHECK(hipEventRecord(stop,0));  HIP_CHECK(hipEventSynchronize(stop)); }
    float elapsed_ms() const { float ms{}; HIP_CHECK(hipEventElapsedTime(&ms,start,stop)); return ms; }

    static void sync_all(size_t g){
        for(size_t i=0;i<g;++i){
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
};

//──────────────────────────────────────────────────────────────────────
// 7.  Forward decls so main.cpp sees them early
//──────────────────────────────────────────────────────────────────────
template <typename T> void setup(size_t,size_t,
                                 gpubuf_vec<T>&,gpubuf_vec<T>&,
                                 const std::vector<T>&,std::vector<hipStream_t>&);
template <typename T> void reset(size_t,size_t,gpubuf_vec<T>&,std::vector<T>&);
template <typename T> void teardown(size_t,gpubuf_vec<T>&,gpubuf_vec<T>&,std::vector<hipStream_t>&);
template <typename T> std::vector<T> generate(size_t,size_t,generator,T,T);

//──────────────────────────────────────────────────────────────────────
// 8.  xorwow PRNG kernel
//──────────────────────────────────────────────────────────────────────
#define XORWOW_NEXT(st,maxv,minv,val)                   do{                 \
    uint32_t t = st[4];                                                     \
    uint32_t s = st[0];                                                     \
    st[4]=st[3]; st[3]=st[2]; st[2]=st[1]; st[1]=s;                         \
    t ^= t>>2; t ^= t<<1; t ^= s ^ (s<<4);                                  \
    st[0]=t; st[5]+=362437u;                                                \
    uint32_t tmp=t+st[5];                                                   \
    val = minv + (static_cast<Tfloat>(tmp)*(maxv-minv))/                    \
                  static_cast<Tfloat>(0xFFFFFFFFu);                         \
}while(0)

template <typename Tfloat>
__global__ void populate_array(size_t N,Tfloat* out,
                               Tfloat minv,Tfloat maxv,
                               bool rnd,size_t seed)
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
        for(int i=0;i<5;++i) XORWOW_NEXT(st,maxv,minv,dummy);
        for(size_t i=0;i<items && start+i<N*N;++i)
            XORWOW_NEXT(st,maxv,minv,out[start+i]);
    }
    else
    {
        for(size_t i=0;i<items && start+i<N*N;++i)
            out[start+i] = static_cast<Tfloat>(start+i);
    }
}

//──────────────────────────────────────────────────────────────────────
// 9.  Host helpers  (generate / assemble / verify / logging)
//──────────────────────────────────────────────────────────────────────
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
                             Tfloat** dev_table,  // device ptr table
                             Tfloat* host_out,
                             const gpubuf_vec<Tfloat>& buffers)
{
    (void)dev_table; // not needed on host
    size_t per = N*N/ngpus;
    for(size_t g=0; g<ngpus; ++g)
        HIP_CHECK(hipMemcpy(host_out+g*per,
                            buffers.host_ptrs()[g],
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

template <typename Tfloat>
void log_matrices(const benchmark_context& ctx,
                  const std::vector<Tfloat>& original,
                  gpubuf_vec<Tfloat>& buffers,
                  std::vector<Tfloat>& assembled)
{
    assemble_output_to_host<Tfloat>(ctx.N, ctx.ngpus,
                                    buffers.data(), assembled.data(), buffers);

    std::cout<<"Original input:\n";
    print2d_host(ctx.N,original);
    std::cout<<"------------------------\nDevice computation:\n";
    print2d_host(ctx.N,assembled);
}

//──────────────────────────────────────────────────────────────────────
// 10.  I/O helpers (setup / reset / teardown)
//──────────────────────────────────────────────────────────────────────
template <typename Tfloat>
void setup(size_t N,size_t ngpus,
           gpubuf_vec<Tfloat>& in,
           gpubuf_vec<Tfloat>& out,
           const std::vector<Tfloat>& h,
           std::vector<hipStream_t>& streams)
{
    size_t per=N*N/ngpus;
    in   = gpubuf_vec<Tfloat>(N,ngpus);
    out  = gpubuf_vec<Tfloat>(N,ngpus);
    streams.resize(ngpus*ngpus);

    for(size_t g=0; g<ngpus; ++g){
        HIP_CHECK(hipSetDevice(g));
        HIP_CHECK(hipMemcpy(in[g], h.data()+g*per, per*sizeof(Tfloat),
                            hipMemcpyHostToDevice));
        for(size_t s=0;s<ngpus;++s)
            HIP_CHECK(hipStreamCreate(&streams[g*ngpus+s]));
    }
    HIP_CHECK(hipSetDevice(0));
}

template <typename Tfloat>
void reset(size_t N,size_t ngpus,
           gpubuf_vec<Tfloat>& out,
           std::vector<Tfloat>& h)
{
    size_t per=N*N/ngpus;
    for(size_t g=0; g<ngpus; ++g){
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
    for(size_t g=0; g<ngpus; ++g){
        HIP_CHECK(hipSetDevice(g));
        for(size_t s=0;s<ngpus;++s)
            HIP_CHECK(hipStreamDestroy(streams[g*ngpus+s]));
    }
    streams.clear();
    in  = gpubuf_vec<Tfloat>();
    out = gpubuf_vec<Tfloat>();
}

//──────────────────────────────────────────────────────────────────────
// 11.  Device print (optional)
//──────────────────────────────────────────────────────────────────────
template <typename Tfloat>
__global__ void print2d(int rows,int cols,const Tfloat* d)
{
    printf("[\n");
    for(int r=0;r<rows;++r){
        printf("  [ ");
        for(int c=0;c<cols;++c) printf("%6.3f ",(double)d[r*cols+c]);
        printf("]\n");
    }
    printf("]\n");
}

//──────────────────────────────────────────────────────────────────────
// 12.  CLI helpers
//──────────────────────────────────────────────────────────────────────
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
    if(w=="random"||w=="0")       g=gen_random;
    else if(w=="ordered"||w=="1") g=gen_ordered;
    else throw std::runtime_error("invalid generator");
    return true;
}

#endif /* MEMBENCH_HELPER_HPP */

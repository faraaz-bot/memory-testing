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
#include <cstdlib>     // malloc / free   <-- CHANGED
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

//======================================================================
//  HIP_CHECK – guard against double-definition
//======================================================================
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t _e = (cmd);                                             \
        if(_e != hipSuccess) {                                             \
            std::cerr << "HIP error (" << hipGetErrorString(_e)            \
                      << ") at " << __FILE__ << ':' << __LINE__ << '\n';   \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while(0)
#endif

//======================================================================
//  Tiny helpers
//======================================================================
inline size_t ceildiv(size_t n, size_t d)    { return (n + d - 1) / d; }
inline bool   is_power_of_two(size_t v)      { return v && !(v & (v-1)); }
using std::min;  using std::max;

//======================================================================
//  Public enums
//======================================================================
enum class precision {
    p_single         = 0,
    p_double         = 1,
    p_complex_single = 2,
    p_complex_double = 3,
};
enum generator { gen_random, gen_ordered };

//======================================================================
//  Benchmark context
//======================================================================
struct benchmark_context {
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
    ~gpubuf() { if(ptr_) HIP_CHECK(hipFree(ptr_)); }

    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return N_;   }
};

/* =====================================================================
 *  gpubuf_vec  –  pointer-table now lives in host RAM  (SEGFAULT FIX)
 * ===================================================================*/
template <typename T>
class gpubuf_vec
{
    size_t   N_{};        // matrix dimension
    size_t   ngpus_{};    // number of GPUs
    T**      buf_{};      // host array of device pointers   <-- CHANGED

public:
    gpubuf_vec() = default;

    gpubuf_vec(size_t N, size_t ngpus) : N_(N), ngpus_(ngpus)
    {
        /* host-allocate the pointer table */
        buf_ = static_cast<T**>(std::malloc(sizeof(T*) * ngpus_));
        if(!buf_) throw std::bad_alloc();

        size_t per = N_ * N_ / ngpus_;

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
        for(int g=0; g<(int)ngpus_; ++g){
            HIP_CHECK(hipSetDevice(g));
            HIP_CHECK(hipFree(buf_[g]));
        }
        std::free(buf_);                         // <-- CHANGED
    }

    T*       operator[](int i)       { return buf_[i]; }
    const T* operator[](int i) const { return buf_[i]; }
    T**      data()                  { return buf_;    }
    const T**data() const            { return buf_;    }

    size_t length() const { return N_; }
    size_t size()   const { return N_*N_/ngpus_; }
};

//======================================================================
//  GPUTimer   (unchanged)
//======================================================================
struct GPUTimer
{
    hipEvent_t start{}, stop{};
    GPUTimer(){ HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
    ~GPUTimer(){ HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop)); }

    void tick(){ HIP_CHECK(hipEventRecord(start,0)); }
    void tock(){ HIP_CHECK(hipEventRecord(stop,0));  HIP_CHECK(hipEventSynchronize(stop)); }
    float elapsed_ms() const
    { float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,start,stop)); return ms; }

    static void sync_all(size_t g){
        for(size_t i=0;i<g;++i){
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
};

/* --------------------------------------------------------------------
 *  =====  everything below here is IDENTICAL to your previous file  ===
 *  (kernels, generate(), assemble, verify, setup/reset/teardown …)
 * ------------------------------------------------------------------ */

/* …  (keep the rest of the file exactly as you already have it) … */

#endif /* MEMBENCH_HELPER_HPP */

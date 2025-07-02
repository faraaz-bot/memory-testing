#pragma once


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>      // <── NEW: malloc / free
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "hip_to_cuda.h"   // must precede our HIP_CHECK

/* ------------------------------------------------------------------ */
/*  HIP_CHECK (keep first definition only)                            */
/* ------------------------------------------------------------------ */
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                   \
    do {                                                                 \
        hipError_t _hip_err = (cmd);                                     \
        if(_hip_err != hipSuccess) {                                     \
            std::cerr << "HIP error (" << hipGetErrorString(_hip_err)    \
                      << ") at " << __FILE__ << ':' << __LINE__ << "\n"; \
            std::exit(-1);                                               \
        }                                                                \
    } while(0)
#endif

/* ------------------------------------------------------------------ */
/*  small helpers                                                     */
/* ------------------------------------------------------------------ */
inline size_t ceildiv(size_t n, size_t d)       { return (n+d-1)/d; }
inline bool   is_power_of_two(size_t n)         { return n && !(n&(n-1)); }
using std::min;   using std::max;

/* ------------------------------------------------------------------ */
/*  enums                                                              */
/* ------------------------------------------------------------------ */
enum class precision   { p_single, p_double, p_complex_single, p_complex_double };
enum generator         { gen_random, gen_ordered };

/* ------------------------------------------------------------------ */
/*  benchmark-context                                                 */
/* ------------------------------------------------------------------ */
struct benchmark_context
{
    size_t                   N{};
    size_t                   ngpus{};
    int                      verbose{};
    int                      mpi_size{};
    std::vector<hipStream_t> streams;
    bool                     verify_results{false};   // used in mpi variant
};

/* ------------------------------------------------------------------ */
/*  RAII wrappers                                                     */
/* ------------------------------------------------------------------ */
template <typename Tfloat>
class gpubuf
{
    size_t  N{};
    Tfloat* buf{};
public:
    gpubuf() = default;
    explicit gpubuf(size_t elems) : N(elems)
    {
        HIP_CHECK(hipMalloc(&buf, sizeof(Tfloat)*N));
        HIP_CHECK(hipMemset(buf, 0, sizeof(Tfloat)*N));
        HIP_CHECK(hipDeviceSynchronize());
    }
    ~gpubuf() { if(buf) HIP_CHECK(hipFree(buf)); }

    Tfloat*       data()       { return buf; }
    const Tfloat* data() const { return buf; }
    size_t        size() const { return N;   }
};

/* ---------- gpubuf_vec – HOST list of DEVICE buffers -------------- */
template <typename Tfloat>
class gpubuf_vec
{
    size_t   N{};          // full matrix dimension
    size_t   ngpus{};
    Tfloat** bufs{};       // host array holding device pointers
public:
    gpubuf_vec() = default;

    gpubuf_vec(size_t N_, size_t ngpus_) : N(N_), ngpus(ngpus_)
    {
        /* host memory for the pointer list */
        bufs = static_cast<Tfloat**>(std::malloc(sizeof(Tfloat*)*ngpus));
        if(!bufs) throw std::bad_alloc();

        const size_t buf_elems = N*N / ngpus;

        for(int i = 0; i < static_cast<int>(ngpus); ++i) {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMalloc(&bufs[i], sizeof(Tfloat)*buf_elems));
            HIP_CHECK(hipMemset(bufs[i], 0, sizeof(Tfloat)*buf_elems));
            HIP_CHECK(hipDeviceSynchronize());
        }
        HIP_CHECK(hipSetDevice(0));
    }

    ~gpubuf_vec()
    {
        for(int i = 0; i < static_cast<int>(ngpus); ++i) {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipFree(bufs[i]));
        }
        std::free(bufs);
    }

    Tfloat*       operator[](int i)       { return bufs[i]; }
    const Tfloat* operator[](int i) const { return bufs[i]; }
    Tfloat**      data()                  { return bufs;    }
    const Tfloat**data() const            { return bufs;    }

    size_t length() const { return N; }
    size_t size()   const { return N*N / ngpus; }
};

/* ------------------------------------------------------------------ */
/*  GPUTimer (unchanged)                                              */
/* ------------------------------------------------------------------ */
struct GPUTimer
{
    hipEvent_t start{}, stop{};
    GPUTimer()  { HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
    ~GPUTimer() { HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop)); }

    void  tick()    { HIP_CHECK(hipEventRecord(start,0)); }
    void  tock()    { HIP_CHECK(hipEventRecord(stop,0));  HIP_CHECK(hipEventSynchronize(stop)); }
    float elapsed() { float ms; HIP_CHECK(hipEventElapsedTime(&ms,start,stop)); return ms; }

    void sync_all(size_t ngpus)
    { for(size_t i=0;i<ngpus;++i){ HIP_CHECK(hipSetDevice(i)); HIP_CHECK(hipDeviceSynchronize()); } }
};

/* ------------------------------------------------------------------ */
/*  … everything else (kernels, host helpers, verify, etc.) is        */
/*  unchanged and can follow here.                                    */
/* ------------------------------------------------------------------ */
#include "helper_rest_of_file.inl"

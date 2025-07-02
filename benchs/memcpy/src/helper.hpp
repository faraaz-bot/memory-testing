#pragma once
/*───────────────────────────────────────────────────────────────
 *  helper.hpp  —  misc utilities for the memcpy benchmark
 *  (patched for Scale / MI300X – 2025-07-02)
 *
 *  ✱ host-side array of device pointers (fixes seg-fault)
 *  ✱ guards for HIP_CHECK re-definition
 *  ✱ removed hand-rolled min(), we use std::min / std::max
 *───────────────────────────────────────────────────────────────*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>      // malloc / free
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "hip_to_cuda.h"       // must precede our HIP_CHECK

/* ------------------------------------------------------------------ */
/*  HIP_CHECK  — keep only one definition                             */
/* ------------------------------------------------------------------ */
#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t _hip_err = (cmd);                                       \
        if(_hip_err != hipSuccess) {                                       \
            std::cerr << "HIP error (" << hipGetErrorString(_hip_err)      \
                      << ") at " << __FILE__ << ':' << __LINE__ << '\n';   \
            std::exit(-1);                                                 \
        }                                                                  \
    } while(0)
#endif

/* ------------------------------------------------------------------ */
/*  helpers                                                           */
/* ------------------------------------------------------------------ */
inline size_t ceildiv(size_t n, size_t d)       { return (n + d - 1) / d; }
inline bool   is_power_of_two(size_t n)         { return n && !(n & (n-1)); }
using std::min;  using std::max;

/* ------------------------------------------------------------------ */
/*  enums & benchmark context                                         */
/* ------------------------------------------------------------------ */
enum class precision   { p_single, p_double, p_complex_single, p_complex_double };
enum generator         { gen_random, gen_ordered };

struct benchmark_context
{
    size_t                   N{};
    size_t                   ngpus{};
    int                      verbose{};
    int                      mpi_size{};
    std::vector<hipStream_t> streams;
    bool                     verify_results{false};    // used only in MPI build
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
    size_t   N{};      // matrix dimension
    size_t   ngpus{};
    Tfloat** bufs{};   // host array holding device pointers

public:
    gpubuf_vec() = default;

    gpubuf_vec(size_t N_, size_t ngpus_) : N(N_), ngpus(ngpus_)
    {
        bufs = static_cast<Tfloat**>(std::malloc(sizeof(Tfloat*) * ngpus));
        if(!bufs) throw std::bad_alloc();

        const size_t buf_elems = N * N / ngpus;

        for(int i = 0; i < static_cast<int>(ngpus); ++i) {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMalloc(&bufs[i], sizeof(Tfloat) * buf_elems));
            HIP_CHECK(hipMemset(bufs[i], 0, sizeof(Tfloat) * buf_elems));
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
    size_t size()   const { return N * N / ngpus; }
};

/* ------------------------------------------------------------------ */
/*  GPU timer                                                         */
/* ------------------------------------------------------------------ */
struct GPUTimer
{
    hipEvent_t start{}, stop{};
    GPUTimer()  { HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
    ~GPUTimer() { HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop)); }

    void  tick()    { HIP_CHECK(hipEventRecord(start, 0)); }
    void  tock()    { HIP_CHECK(hipEventRecord(stop, 0));  HIP_CHECK(hipEventSynchronize(stop)); }
    float elapsed() { float ms; HIP_CHECK(hipEventElapsedTime(&ms, start, stop)); return ms; }

    void sync_all(size_t ngpus) {
        for(size_t i=0;i<ngpus;++i) { HIP_CHECK(hipSetDevice(i)); HIP_CHECK(hipDeviceSynchronize()); }
    }
};

/* ------------------------------------------------------------------ */
/*  *** EVERYTHING FROM HERE DOWNWARDS IS IDENTICAL TO THE ORIGINAL *** */
/*  (data-gen kernels, verify helpers, setup/reset/teardown, etc.)      */
/* ------------------------------------------------------------------ */

/* --- PRNG kernel macro -------------------------------------------- */
#define xorwow_next(states, maxv, minv, val)                       \
    uint32_t t = states[4];                                        \
    uint32_t s = states[0];                                        \
    states[4]  = states[3];                                        \
    states[3]  = states[2];                                        \
    states[2]  = states[1];                                        \
    states[1]  = s;                                                \
    t ^= t >> 2;  t ^= t << 1;  t ^= s ^ (s << 4);                 \
    states[0] = t;                                                 \
    states[5] += 362437;                                           \
    uint32_t temp = t + states[5];                                 \
    val = minv + (static_cast<Tfloat>(temp) * (maxv - minv)) /     \
                    static_cast<Tfloat>(4294967295U);

/* --- populate_array kernel ---------------------------------------- */
template <typename Tfloat>
__global__ void populate_array(size_t   N,
                               Tfloat*  out,
                               Tfloat   minv,
                               Tfloat   maxv,
                               bool     isRandom,
                               size_t   seed)
{
    const size_t bIdx = blockIdx.x;
    const size_t tIdx = threadIdx.x;
    const size_t itemsPerThread = N;
    const size_t blockSize      = itemsPerThread * blockDim.x;
    const size_t start          = (tIdx * itemsPerThread) + (bIdx * blockSize);

    if(isRandom)
    {
        uint32_t st[6] = { static_cast<uint32_t>(seed ^  tIdx + bIdx),
                           static_cast<uint32_t>(seed >> 1 ^ (tIdx + bIdx*2)),
                           static_cast<uint32_t>(seed >> 2 ^ (tIdx + bIdx*3)),
                           static_cast<uint32_t>(seed >> 3 ^ (tIdx + bIdx*4)),
                           static_cast<uint32_t>(seed >> 4 ^ (tIdx + bIdx*5)),
                           static_cast<uint32_t>(seed +  tIdx + bIdx) };

        /* warm-up */
        for(int i=0;i<5;++i) { xorwow_next(st, maxv, minv, st[0]); }

        for(size_t i = 0; i < itemsPerThread; ++i) {
            if(start + i >= N*N) continue;
            xorwow_next(st, maxv, minv, out[start+i]);
        }
    }
    else
    {
        for(size_t i = 0; i < itemsPerThread; ++i) {
            if(start + i >= N*N) continue;
            out[start+i] = start + i;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  host-side helpers (generate, verify, print, setup, etc.)          */
/* ------------------------------------------------------------------ */

/* generate host vector, create device buf, fill on GPU, copy back */
template <typename Tfloat>
std::vector<Tfloat> generate(size_t N, size_t M,
                             generator gen, Tfloat minv, Tfloat maxv)
{
    std::vector<Tfloat> host(N*M);
    bool isRandom = (gen == gen_random);

    Tfloat* dArr;
    HIP_CHECK(hipMalloc(&dArr, sizeof(Tfloat)*N*M));

    size_t threads = (N <= 1024 ? N : 1024);
    size_t blocks  = std::ceil(static_cast<double>(N*M) /
                               static_cast<double>(threads*N));

    auto now = std::chrono::time_point_cast<std::chrono::milliseconds>
               (std::chrono::system_clock::now());
    size_t seed = now.time_since_epoch().count();

    populate_array<<<blocks, threads>>>(N, dArr, minv, maxv, isRandom, seed);
    HIP_CHECK(hipMemcpy(host.data(), dArr, sizeof(Tfloat)*N*M,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dArr));
    return host;
}

/* assemble device partitions into one host-side matrix */
template <typename Tfloat>
void assemble_output_to_host(int N, int ngpus,
                             Tfloat** gpubufs, Tfloat* host)
{
    const size_t chunk = N*N / ngpus;
    for(int i=0;i<ngpus;++i)
        HIP_CHECK(hipMemcpy(host + i*chunk, gpubufs[i],
                            sizeof(Tfloat)*chunk, hipMemcpyDeviceToHost));
}

/* element-wise compare two N×N matrices */
template <typename Tfloat>
bool is_same_matrix(int N,
                    const std::vector<Tfloat>& a,
                    const std::vector<Tfloat>& b)
{
    return std::equal(a.begin(), a.end(), b.begin());
}

/* host reference copy (block cyclic) */
template <typename Tfloat>
void host_copy(size_t N, size_t ngpus,
               const Tfloat* in, Tfloat* out)
{
    const size_t chunk = N*N / ngpus;
    std::vector<size_t> offs(ngpus);
    for(int i=0;i<ngpus;++i) offs[i] = i*chunk;

    const size_t sub = N / ngpus;                 // block edge
    const size_t bytes = sizeof(Tfloat)*sub;
    const size_t elems_per_row = sub*ngpus;

#pragma omp parallel for
    for(int src=0; src<ngpus; ++src)
        for(int dst=0; dst<ngpus; ++dst)
            for(int row=0; row<sub; ++row)
            {
                size_t src_off = offs[src] + dst*sub + elems_per_row*row;
                size_t dst_off = offs[dst] + src*sub + elems_per_row*row;
                std::memcpy(out+dst_off, in+src_off, bytes);
            }
}

/* host reference transpose (out-of-place) */
template <typename Tfloat>
void host_transpose(int N, const Tfloat* in, Tfloat* out)
{
#pragma omp parallel for
    for(int i=0;i<N;++i)
        for(int j=0;j<N;++j)
            out[j*N + i] = in[i*N + j];
}

/* print utilities -------------------------------------------------- */
template <typename Tfloat>
__global__ void print(const int N, const Tfloat* in)
{
    printf("[ ");
    for(int i=0;i<N;++i) printf("%.6f ", in[i]);
    printf("]\n");
}

template <typename Tfloat>
__global__ void print2d(const int rows, const int cols, const Tfloat* in)
{
    printf("[\n");
    for(int r=0;r<rows;++r){
        printf("  [ ");
        for(int c=0;c<cols;++c) printf("%.6f ", in[r*cols+c]);
        printf("]\n");
    }
    printf("]\n");
}

template <typename Tfloat>
void print_host_2d(int rows, int cols, const std::vector<Tfloat>& v)
{
    std::cout << "[\n";
    for(int r=0;r<rows;++r){
        std::cout << "  [ ";
        for(int c=0;c<cols;++c) std::cout << std::setw(6) << v[r*cols+c] << ' ';
        std::cout << "]\n";
    }
    std::cout << "]\n";
}

/* verify result against host reference ----------------------------- */
template <typename Tfloat>
bool verify_results(size_t N, size_t ngpus, int verbose,
                    std::vector<Tfloat>& ref,
                    gpubuf_vec<Tfloat>&  dev_out,
                    std::vector<Tfloat>& assembled)
{
    assemble_output_to_host<Tfloat>(N, ngpus, dev_out.data(), assembled.data());
    bool ok = is_same_matrix<Tfloat>(N, ref, assembled);
    if(!ok && verbose){
        std::cout << "Mismatch!\nHost:\n";
        print_host_2d<Tfloat>(N,N,ref);
        std::cout << "Device:\n";
        print_host_2d<Tfloat>(N,N,assembled);
    }
    return ok;
}

/* setup / reset / teardown ---------------------------------------- */
template <typename Tfloat>
void setup(size_t N, size_t ngpus,
           gpubuf_vec<Tfloat>& in, gpubuf_vec<Tfloat>& out,
           const std::vector<Tfloat>& h_in,
           std::vector<hipStream_t>& streams)
{
    size_t chunk = N*N / ngpus;
    for(int i=0;i<ngpus;++i){
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMemcpy(in[i], h_in.data()+i*chunk,
                            sizeof(Tfloat)*chunk, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(out[i], 0, sizeof(Tfloat)*chunk));
        for(int j=0;j<ngpus;++j) HIP_CHECK(hipStreamCreate(&streams[i*ngpus+j]));
    }
}

template <typename Tfloat>
void reset(int N, int ngpus,
           gpubuf_vec<Tfloat>& out,
           std::vector<Tfloat>& assembled)
{
    size_t chunk = N*N / ngpus;
    for(int i=0;i<ngpus;++i){
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMemset(out[i], 0, sizeof(Tfloat)*chunk));
        HIP_CHECK(hipDeviceSynchronize());
    }
    std::fill(assembled.begin(), assembled.end(), 0);
}

template <typename Tfloat>
void teardown(int ngpus,
              gpubuf_vec<Tfloat>& in,
              gpubuf_vec<Tfloat>& out,
              std::vector<hipStream_t>& streams)
{
    for(int i=0;i<ngpus;++i){
        HIP_CHECK(hipSetDevice(i));
        for(int j=0;j<ngpus;++j) HIP_CHECK(hipStreamDestroy(streams[i*ngpus+j]));
    }
}

/* CLI11 lexical_cast helpers -------------------------------------- */
inline bool lexical_cast(const std::string& s, precision& p)
{
    if(s=="single"||s=="0")       p=precision::p_single;
    else if(s=="double"||s=="1")  p=precision::p_double;
    else if(s=="c_single"||s=="2")p=precision::p_complex_single;
    else if(s=="c_double"||s=="3")p=precision::p_complex_double;
    else return false;
    return true;
}

inline bool lexical_cast(const std::string& s, generator& g)
{
    if(s=="random"||s=="0")       g=gen_random;
    else if(s=="ordered"||s=="1") g=gen_ordered;
    else return false;
    return true;
}

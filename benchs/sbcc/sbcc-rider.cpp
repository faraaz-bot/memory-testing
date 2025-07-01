#include "hip_to_cuda.h"
#include <random>

#ifdef SBCC_RUNTIME_COMPILE
#include <hip/hiprtc.h>

#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Helper class that handles alignment of kernel arguments
class RTCKernelArgs
{
public:
    RTCKernelArgs() = default;
    void append_ptr(void* ptr)
    {
        append(&ptr, sizeof(void*));
    }
    void append_size_t(size_t s)
    {
        append(&s, sizeof(size_t));
    }
    void append_unsigned_int(unsigned int i)
    {
        append(&i, sizeof(unsigned int));
    }
    void append_int(int i)
    {
        append(&i, sizeof(int));
    }
    void append_double(double d)
    {
        append(&d, sizeof(double));
    }
    void append_float(float f)
    {
        append(&f, sizeof(float));
    }

    size_t size_bytes() const
    {
        return buf.size();
    }
    void* data()
    {
        return buf.data();
    }

private:
    void append(void* src, size_t nbytes)
    {
        // values need to be aligned to their width (i.e. 8-byte values
        // need 8-byte alignment, 4-byte needs 4-byte alignment)
        size_t oldsize = buf.size();
        size_t padding = oldsize % nbytes ? nbytes - (oldsize % nbytes) : 0;
        buf.resize(oldsize + padding + nbytes);
        std::copy_n(static_cast<const char*>(src), nbytes, buf.begin() + oldsize + padding);
    }
    std::vector<char> buf;
};

hipModule_t   module = nullptr;
hipFunction_t kernel = nullptr;
#else
#include "sbcc-kernel.h"
#endif
#include "sbrr-kernel.h"

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

template <typename T1, typename T2>
T1 ceildiv(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

// return a GPU buffer filled with random float2 data
gpubuf_t<float2> hipMalloc_random_float2s(unsigned int Nfloat2s)
{
    gpubuf_t<float2> ret;
    if(ret.alloc(sizeof(float2) * Nfloat2s) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");

    std::vector<float2> hostBuf(Nfloat2s);

    auto partitions     = std::max<size_t>(std::thread::hardware_concurrency(), 32);
    auto partition_size = ceildiv(Nfloat2s, partitions);

#pragma omp parallel for
    for(unsigned int partition = 0; partition < partitions; ++partition)
    {
        std::mt19937                           gen(partition);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        auto begin = partition * partition_size;
        if(begin >= Nfloat2s)
            continue;
        auto end = std::min(begin + partition_size, Nfloat2s);

        for(auto d = hostBuf.begin() + begin; d != hostBuf.begin() + end; ++d)
        {
            d->x = dis(gen);
            d->y = dis(gen);
        }
    }
    if(hipMemcpy(ret.data(), hostBuf.data(), sizeof(float2) * Nfloat2s, hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to memcpy");
    return ret;
}

// return a GPU buffer filled with size_t's copied from a host vector
gpubuf_t<size_t> host_sizes_to_dev(const std::vector<size_t>& h)
{
    gpubuf_t<size_t> ret;
    if(ret.alloc(sizeof(size_t) * h.size()) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemcpy(ret.data(), h.data(), sizeof(size_t) * h.size(), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to memcpy");
    return ret;
}

void launch_sbrr(const gpubuf_t<float2>& twiddles,
                 size_t                  dim,
                 const gpubuf_t<size_t>& lengths_d,
                 const gpubuf_t<size_t>& stride_d,
                 size_t                  nbatch,
                 gpubuf_t<float2>&       buf,
                 dim3                    gridDim,
                 dim3                    blockDim,
                 unsigned int            lds_bytes)
{
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(ip_forward_length125_SBRR<float2,
                                                  SB_UNIT,
                                                  EmbeddedType::NONE,
                                                  CallbackType::NONE,
                                                  DirectRegType::FORCE_OFF_OR_NOT_SUPPORT>),
        gridDim,
        blockDim,
        lds_bytes,
        nullptr,
        twiddles.data(),
        dim,
        lengths_d.data(),
        stride_d.data(),
        nbatch,
        0,
        nullptr,
        nullptr,
        0,
        nullptr,
        nullptr,
        buf.data());
}

void launch_sbcc(const gpubuf_t<float2>& twiddles,
                 size_t                  dim,
                 const gpubuf_t<size_t>& lengths_d,
                 const gpubuf_t<size_t>& stride_d,
                 size_t                  nbatch,
                 gpubuf_t<float2>&       buf,
                 dim3                    gridDim,
                 dim3                    blockDim,
                 unsigned int            lds_bytes)
{
#ifdef SBCC_RUNTIME_COMPILE
    RTCKernelArgs kargs;
    kargs.append_ptr(twiddles.data());
    kargs.append_ptr(nullptr);
    kargs.append_size_t(dim);
    kargs.append_ptr(lengths_d.data());
    kargs.append_ptr(stride_d.data());
    kargs.append_size_t(nbatch);
    kargs.append_unsigned_int(0);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_unsigned_int(0);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(buf.data());

    auto  size     = kargs.size_bytes();
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      kargs.data(),
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &size,
                      HIP_LAUNCH_PARAM_END};
    if(hipModuleLaunchKernel(kernel,
                             gridDim.x,
                             gridDim.y,
                             gridDim.z,
                             blockDim.x,
                             blockDim.y,
                             blockDim.z,
                             lds_bytes,
                             0,
                             nullptr,
                             config)
       != hipSuccess)
    {
        throw std::runtime_error("failed to launch");
    }
#else
    hipLaunchKernelGGL(fft_rtc_fwd_len125_sp_ip_CI_sbcc_dirReg,
                       gridDim,
                       blockDim,
                       lds_bytes,
                       0,
                       twiddles.data(),
                       nullptr,
                       dim,
                       lengths_d.data(),
                       stride_d.data(),
                       nbatch,
                       0,
                       nullptr,
                       nullptr,
                       0,
                       nullptr,
                       nullptr,
                       buf.data());
#endif
}

int main()
{
#ifdef SBCC_RUNTIME_COMPILE
    // load the kernel from the separate header file
    std::ifstream header_file("../sbcc-kernel.h");
    std::string   header_str;
    std::getline(header_file, header_str, static_cast<char>(0));
    hiprtcProgram prog;
    if(hiprtcCreateProgram(&prog, header_str.c_str(), "rtc.cu", 0, nullptr, nullptr)
       != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("unable to create program");
    }
    std::vector<const char*> options;
    options.push_back("-O3");

    auto compileResult = hiprtcCompileProgram(prog, options.size(), options.data());
    if(compileResult != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS)
                throw std::runtime_error(log.data());
        }
        throw std::runtime_error("compile failed without log");
    }

    size_t codeSize;
    if(hiprtcGetCodeSize(prog, &codeSize) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code size");

    std::vector<char> code(codeSize);
    if(hiprtcGetCode(prog, code.data()) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code");
    hiprtcDestroyProgram(&prog);

    if(hipModuleLoadData(&module, code.data()) != hipSuccess)
        throw std::runtime_error("failed to load module");

    if(hipModuleGetFunction(&kernel, module, "fft_rtc_fwd_len125_sp_ip_CI_sbcc_dirReg")
       != hipSuccess)
        throw std::runtime_error("failed to get function");
#endif

    // problem size: single-precision batch-10 length-125x125x125 FFT.
    const unsigned int length = 125;
    const unsigned int batch  = 10;

    // length-125 SBCC needs 120 normal twiddles
    const unsigned int twiddle_length = 120;

    // length-125 SBRR does 25 threads per transform, 10 transforms per block
    const unsigned int threads_per_transformRR = 25;
    const unsigned int transforms_per_blockRR  = 10;
    const unsigned int gridRR                  = ceildiv(125 * 125 * batch, transforms_per_blockRR);
    const unsigned int threadsRR               = threads_per_transformRR * transforms_per_blockRR;
    const bool         halfLdsRR               = false;
    const unsigned int lds_bytesRR
        = transforms_per_blockRR * length * sizeof(float2) / (halfLdsRR ? 2 : 1);

    // length-125 SBCC does 25 threads per transform, 16 transforms per block
    const unsigned int threads_per_transformCC = 25;
    const unsigned int transforms_per_blockCC  = 16;
    const unsigned int gridCC                  = ceildiv(125, transforms_per_blockCC) * 125 * batch;
    const unsigned int threadsCC               = threads_per_transformCC * transforms_per_blockCC;
    const bool         halfLdsCC               = false;
    const unsigned int lds_bytesCC
        = transforms_per_blockCC * length * sizeof(float2) / (halfLdsCC ? 2 : 1);

    const unsigned int nTrials = 20;

    auto lengths_RR_d = host_sizes_to_dev({125});
    auto stride_RR_d  = host_sizes_to_dev({1, 125});

    auto lengths_CC1_d = host_sizes_to_dev({125, 125});
    auto stride_CC1_d  = host_sizes_to_dev({125, 1, 15625});

    auto lengths_CC2_d = host_sizes_to_dev({125, 125, 125});
    auto stride_CC2_d  = host_sizes_to_dev({15625, 1, 125, 1953125});

    auto input    = hipMalloc_random_float2s(length * length * length * batch);
    auto twiddles = hipMalloc_random_float2s(twiddle_length);

    std::vector<float> samples(nTrials);

    hipEvent_t start, stop;
    if(hipEventCreate(&start) != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");
    if(hipEventCreate(&stop) != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");
    for(unsigned int i = 0; i < nTrials; ++i)
    {
        if(hipEventRecord(start) != hipSuccess)
            throw std::runtime_error("hipEventRecord failed");

        launch_sbrr(
            twiddles, 1, lengths_RR_d, stride_RR_d, 156250, input, gridRR, threadsRR, lds_bytesRR);
        launch_sbcc(
            twiddles, 2, lengths_CC1_d, stride_CC1_d, 1250, input, gridCC, threadsCC, lds_bytesCC);
        launch_sbcc(
            twiddles, 3, lengths_CC2_d, stride_CC2_d, 1250, input, gridCC, threadsCC, lds_bytesCC);

        if(hipEventRecord(stop) != hipSuccess)
            throw std::runtime_error("hipEventRecord failed");
        if(hipEventSynchronize(stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");
        if(hipEventElapsedTime(&samples[i], start, stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failed");
    }

    for(auto s : samples)
    {
        printf("%f ms\n", static_cast<double>(s));
    }
    std::sort(samples.begin(), samples.end());
    printf("median %f ms\n", static_cast<double>(samples[samples.size() / 2]));
    return 0;
}

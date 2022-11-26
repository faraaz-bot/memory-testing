#include <hip/hip_runtime.h>
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

// return a GPU buffer filled with random double2 data
gpubuf_t<double2> hipMalloc_random_double2s(unsigned int Ndouble2s)
{
    gpubuf_t<double2> ret;
    if(ret.alloc(sizeof(double2) * Ndouble2s) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");

    std::vector<double2> hostBuf(Ndouble2s);

    auto partitions     = std::max<size_t>(std::thread::hardware_concurrency(), 32);
    auto partition_size = ceildiv(Ndouble2s, partitions);

#pragma omp parallel for
    for(unsigned int partition = 0; partition < partitions; ++partition)
    {
        std::mt19937                           gen(partition);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        auto begin = partition * partition_size;
        if(begin >= Ndouble2s)
            continue;
        auto end = std::min(begin + partition_size, Ndouble2s);

        for(auto d = hostBuf.begin() + begin; d != hostBuf.begin() + end; ++d)
        {
            d->x = dis(gen);
            d->y = dis(gen);
        }
    }
    if(hipMemcpy(ret.data(), hostBuf.data(), sizeof(double2) * Ndouble2s, hipMemcpyHostToDevice)
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
    options.push_back("-Xclang");
    options.push_back("-fallow-half-arguments-and-returns");
    options.push_back("-D__HIP_HCC_COMPAT_MODE__=1");
    options.push_back("-mllvm");
    options.push_back("-amdgpu-early-inline-all=true");
    options.push_back("-mllvm");
    options.push_back("-amdgpu-function-calls=false");
    options.push_back("-D__HIP_PLATFORM_AMD__=1");
    options.push_back("-D__HIP_PLATFORM_HCC__=1");
    options.push_back("-O3");
    options.push_back("-DNDEBUG");
    options.push_back("-fPIC");
    options.push_back("-fvisibility=hidden");
    options.push_back("-fvisibility-inlines-hidden");
    options.push_back("-fno-gpu-rdc");
    options.push_back("-x");
    options.push_back("hip");
    options.push_back("-std=gnu++17");
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

    if(hipModuleGetFunction(
           &kernel, module, "fft_rtc_fwd_len168_dp_op_CI_CI_sbcc_twdbase8_2step_dirReg")
       != hipSuccess)
        throw std::runtime_error("failed to get function");
#endif

    // problem size: double-precision batch-5000 length-21504 FFT.
    // Length-21504 is decomposed into length-168 SBCC + length-128 SBRC.
    const unsigned int lengthCC = 168;
    const unsigned int lengthRC = 128;
    const unsigned int length   = lengthCC * lengthRC;
    const unsigned int batch    = 5000;

    // length-168 SBCC needs 161 normal twiddles, 512 large twd.
    const unsigned int twiddle_length       = 161;
    const unsigned int large_twiddle_length = 512;

    // length-168 SBCC does 42 threads per transform, 6 transforms per block
    const unsigned int threads_per_transform = 42;
    const unsigned int transforms_per_block  = 6;
    const unsigned int grid                  = ceildiv(lengthRC, transforms_per_block) * batch;
    const unsigned int threads               = threads_per_transform * transforms_per_block;
    const bool         halfLds               = true;
    const unsigned int lds_bytes
        = transforms_per_block * lengthCC * sizeof(double2) / (halfLds ? 2 : 1);

    const unsigned int nTrials = 20;

    std::vector<size_t> lengths_h;
    lengths_h.push_back(lengthCC);
    lengths_h.push_back(lengthRC);
    std::vector<size_t> stride_h;
    stride_h.push_back(lengthRC);
    stride_h.push_back(1);
    stride_h.push_back(length);
    auto lengths_d    = host_sizes_to_dev(lengths_h);
    auto stride_in_d  = host_sizes_to_dev(stride_h);
    auto stride_out_d = host_sizes_to_dev(stride_h);

    auto input          = hipMalloc_random_double2s(length * batch);
    auto output         = hipMalloc_random_double2s(length * batch);
    auto twiddles       = hipMalloc_random_double2s(twiddle_length);
    auto large_twiddles = hipMalloc_random_double2s(large_twiddle_length);

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
#ifdef SBCC_RUNTIME_COMPILE
        RTCKernelArgs kargs;
        kargs.append_ptr(twiddles.data());
        kargs.append_ptr(large_twiddles.data());
        kargs.append_size_t(2);
        kargs.append_ptr(lengths_d.data());
        kargs.append_ptr(stride_in_d.data());
        kargs.append_ptr(stride_out_d.data());
        kargs.append_size_t(batch);
        kargs.append_unsigned_int(0);
        kargs.append_ptr(nullptr);
        kargs.append_ptr(nullptr);
        kargs.append_unsigned_int(0);
        kargs.append_ptr(nullptr);
        kargs.append_ptr(nullptr);
        kargs.append_ptr(input.data());
        kargs.append_ptr(output.data());

        auto  size     = kargs.size_bytes();
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          kargs.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &size,
                          HIP_LAUNCH_PARAM_END};
        if(hipModuleLaunchKernel(kernel, grid, 1, 1, threads, 1, 1, lds_bytes, 0, nullptr, config)
           != hipSuccess)
        {
            throw std::runtime_error("failed to launch");
        }
#else
        hipLaunchKernelGGL(fft_rtc_fwd_len168_dp_op_CI_CI_sbcc_twdbase8_2step_dirReg,
                           grid,
                           threads,
                           lds_bytes,
                           0,
                           twiddles.data(),
                           large_twiddles.data(),
                           2,
                           lengths_d.data(),
                           stride_in_d.data(),
                           stride_out_d.data(),
                           batch,
                           0,
                           nullptr,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           input.data(),
                           output.data());
#endif
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

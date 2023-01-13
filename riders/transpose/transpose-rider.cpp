// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>

#include <boost/program_options.hpp>
#include <iostream>
namespace po = boost::program_options;

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifdef TRANSPOSE_RUNTIME_COMPILE
#include <hip/hiprtc.h>

#include <fstream>
#include <string>
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
#include "transpose-kernel.h"
#endif

template <typename T1, typename T2>
T1 ceildiv(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

// Enum for user-specified variable type.
enum var_type
{
    real_single,
    real_double,
    complex_single,
    complex_double
};

// Buffer initialization kernel
template <typename Tval>
__global__ void __launch_bounds__(1024, 1)
    init_buffer(Tval* __restrict__ idata, const int Nx, const int Ny)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < Nx && iy < Ny)
    {
        idata[iy * Nx + ix] = (Tval)cos(iy * Nx + ix);
        //idata[iy * Nx + ix] = (Tval)(iy * Nx + ix);
    }
}

class transpose_params
{
public:
    size_t blockSize = 32;
    int    padding   = 1;

    bool                complex          = true;
    bool                double_precision = true;
    std::vector<size_t> length;

    // Get the enum for the data type from the type parameters.
    var_type get_var_type() const
    {
        if(double_precision)
        {
            if(complex)
                return complex_double;
            else
                return real_double;
        }
        else
        {
            if(complex)
                return complex_single;
            else
                return real_single;
        }
    }

    // sizeof for the user-specified data type.
    size_t var_size() const
    {
        switch(get_var_type())
        {
        case real_single:
            return sizeof(float);
        case real_double:
            return sizeof(double);
        case complex_single:
            return sizeof(std::complex<float>);
        case complex_double:
            return sizeof(std::complex<double>);
        }
    }

    std::string print_var_type() const
    {
        switch(get_var_type())
        {
        case real_single:
            return "float";
        case real_double:
            return "double";
        case complex_single:
            return "std::complex<float>";
        case complex_double:
            return "std::complex<double>";
        }
    }

    transpose_params(){};
    ~transpose_params(){};

    // Device buffer size computation.
    size_t buffer_size()
    {
        return var_size() * length[0] * length[1];
    }

    // Launch bound computation.
    dim3 blocks()
    {
        return dim3(ceildiv(length[0], blockSize), ceildiv(length[1], blockSize));
    }
    dim3 threads()
    {
        return dim3(blockSize, blockSize);
    }

    // Initialize the device buffer with data.
    void compute_input(void* in)
    {
        switch(get_var_type())
        {
        case real_single:
            hipLaunchKernelGGL(init_buffer<float>,
                               blocks(),
                               threads(),
                               0,
                               0, // stream
                               (float*)in,
                               length[0],
                               length[1]);
            break;
        case complex_single:
            hipLaunchKernelGGL(init_buffer<float2>,
                               blocks(),
                               threads(),
                               0,
                               0, // stream
                               (float2*)in,
                               length[0],
                               length[1]);
            break;
        case real_double:
            hipLaunchKernelGGL(init_buffer<double>,
                               blocks(),
                               threads(),
                               0,
                               0, // stream
                               (double*)in,
                               length[0],
                               length[1]);
            break;
        case complex_double:
            hipLaunchKernelGGL(init_buffer<double2>,
                               blocks(),
                               threads(),
                               0,
                               0, // stream
                               (double2*)in,
                               length[0],
                               length[1]);
            break;
        }
    }

    // Compute occupancy from hip API.
    int occupancy()
    {
        int        max_blocks_per_sm{};
        hipError_t ret{};

#ifdef TRANSPOSE_RUNTIME_COMPILE
        ret = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kernel, threads().x * threads().y * threads().z, lds_bytes());
#else
        switch(get_var_type())
        {
        case real_single:
            ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                               transpose<float>,
                                                               threads().x * threads().y
                                                                   * threads().z,
                                                               lds_bytes());
            break;
        case complex_single:
            ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                               transpose<float2>,
                                                               threads().x * threads().y
                                                                   * threads().z,
                                                               lds_bytes());
            break;
        case real_double:
            ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                               transpose<double>,
                                                               threads().x * threads().y
                                                                   * threads().z,
                                                               lds_bytes());
            break;
        case complex_double:
            ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                               transpose<double2>,
                                                               threads().x * threads().y
                                                                   * threads().z,
                                                               lds_bytes());
            break;
        }
#endif
        if(ret != hipSuccess)
            throw std::runtime_error("hipOccupancyMaxActiveBlocksPerMultiprocessor failed");

        return max_blocks_per_sm;
    }

    // Compute shared memory requirement
    size_t lds_bytes()
    {
#if USE_LDS
        size_t lds_count = (blockSize + padding) * blockSize;
#else
        size_t lds_count = 0;
#endif
        return lds_count * var_size();
    }

    // hipExtLaunch execution path.
    void ext_execute(void* in, void* out, hipEvent_t start, hipEvent_t stop)
    {
        int ggl_flags = 0;

#ifdef TRANSPOSE_RUNTIME_COMPILE
        // just do normal execute for now
        execute(in, out);
#else
        switch(get_var_type())
        {
        case real_single:
            hipExtLaunchKernelGGL(transpose<float>,
                                  blocks(),
                                  threads(),
                                  lds_bytes(),
                                  0, // stream
                                  start,
                                  stop,
                                  ggl_flags,
                                  (float*)in,
                                  (float*)out,
                                  length[0],
                                  length[1],
                                  blockSize,
                                  padding);
            break;

        case complex_single:
            hipExtLaunchKernelGGL(transpose<float2>,
                                  blocks(),
                                  threads(),
                                  lds_bytes(),
                                  0, // stream
                                  start,
                                  stop,
                                  ggl_flags,
                                  (float2*)in,
                                  (float2*)out,
                                  length[0],
                                  length[1],
                                  blockSize,
                                  padding);
            break;
        case real_double:
            hipExtLaunchKernelGGL(transpose<double>,
                                  blocks(),
                                  threads(),
                                  lds_bytes(),
                                  0, // stream
                                  start,
                                  stop,
                                  ggl_flags,
                                  (double*)in,
                                  (double*)out,
                                  length[0],
                                  length[1],
                                  blockSize,
                                  padding);
            break;
        case complex_double:
            hipExtLaunchKernelGGL(transpose<double2>,
                                  blocks(),
                                  threads(),
                                  lds_bytes(),
                                  0, // stream
                                  start,
                                  stop,
                                  ggl_flags,
                                  (double2*)in,
                                  (double2*)out,
                                  length[0],
                                  length[1],
                                  blockSize,
                                  padding);
            break;
        }
#endif
    };

    // Normal execution path.
    void execute(void* in, void* out)
    {
#ifdef TRANSPOSE_RUNTIME_COMPILE
        RTCKernelArgs kargs;
        kargs.append_ptr(in);
        kargs.append_ptr(out);
        kargs.append_int(length[0]);
        kargs.append_int(length[1]);
        kargs.append_int(blockSize);
        kargs.append_int(padding);

        auto  size     = kargs.size_bytes();
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          kargs.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &size,
                          HIP_LAUNCH_PARAM_END};

        auto gridDim  = blocks();
        auto blockDim = threads();
        (void)hipModuleLaunchKernel(kernel,
                                    gridDim.x,
                                    gridDim.y,
                                    gridDim.z,
                                    blockDim.x,
                                    blockDim.y,
                                    blockDim.z,
                                    lds_bytes(),
                                    0,
                                    nullptr,
                                    config);
#else
        switch(get_var_type())
        {
        case real_single:
            hipLaunchKernelGGL(transpose<float>,
                               blocks(),
                               threads(),
                               lds_bytes(),
                               0, // stream
                               (float*)in,
                               (float*)out,
                               length[0],
                               length[1],
                               blockSize,
                               padding);
            break;
        case complex_single:
            hipLaunchKernelGGL(transpose<float2>,
                               blocks(),
                               threads(),
                               lds_bytes(),
                               0, // stream
                               (float2*)in,
                               (float2*)out,
                               length[0],
                               length[1],
                               blockSize,
                               padding);
            break;
        case real_double:
            hipLaunchKernelGGL(transpose<double>,
                               blocks(),
                               threads(),
                               lds_bytes(),
                               0, // stream
                               (double*)in,
                               (double*)out,
                               length[0],
                               length[1],
                               blockSize,
                               padding);
            break;
        case complex_double:
            hipLaunchKernelGGL(transpose<double2>,
                               blocks(),
                               threads(),
                               lds_bytes(),
                               0, // stream
                               (double2*)in,
                               (double2*)out,
                               length[0],
                               length[1],
                               blockSize,
                               padding);
            break;
        }
#endif
    };
};

#ifdef TRANSPOSE_RUNTIME_COMPILE
void rtc_transpose(const transpose_params& params)
{
    std::string src;
    switch(params.get_var_type())
    {
    case real_single:
        src += "typedef float Tval;\n";
        break;
    case complex_single:
        src += "typedef float2 Tval;\n";
        break;
    case real_double:
        src += "typedef double Tval;\n";
        break;
    case complex_double:
        src += "typedef double2 Tval;\n";
        break;
    }
    std::ifstream header_file("../transpose-kernel.h");
    std::string   header_str;
    std::getline(header_file, header_str, static_cast<char>(0));
    src += header_str;

    hiprtcProgram prog;
    if(hiprtcCreateProgram(&prog, src.c_str(), "rtc.cu", 0, nullptr, nullptr) != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("unable to create program");
    }
    std::vector<const char*> options;
#ifdef USE_LDS
    options.push_back("-DUSE_LDS=1");
#endif
    options.push_back("-DTRANSPOSE_RUNTIME_COMPILE=1");

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

    if(hipModuleGetFunction(&kernel, module, "transpose") != hipSuccess)
        throw std::runtime_error("failed to get function");
}
#endif

template <typename Tval, typename Tvlength>
int Tcheck_result(const void* vin, const void* vout, const Tvlength& length, const int verbose)
{
    const auto pin  = (Tval*)vin;
    const auto pout = (Tval*)vout;

    int nbad = 0;

    for(int ix = 0; ix < length[0]; ++ix)
    {
        for(int iy = 0; iy < length[1]; ++iy)
        {
            if(pin[ix * length[1] + iy] != pout[iy * length[0] + ix])
            {
                nbad++;
                if(verbose)
                    std::cout << "incorrect result at (" << ix << "," << iy << ")\n";
            }
        }
    }
    return nbad;
}

// After having copied the data to the host, check that it's a transpose.
int check_result(const void*             vin,
                 const void*             vout,
                 const transpose_params& params,
                 const int               verbose)
{
    switch(params.get_var_type())
    {
    case real_single:
        return Tcheck_result<float>(vin, vout, params.length, verbose);
        break;
    case complex_single:
        return Tcheck_result<std::complex<float>>(vin, vout, params.length, verbose);
        break;
    case real_double:
        return Tcheck_result<double>(vin, vout, params.length, verbose);
        break;
    case complex_double:
        return Tcheck_result<std::complex<double>>(vin, vout, params.length, verbose);
        break;
    default:
        throw std::runtime_error("invalid data type");
    }
    return 0;
}

template <typename Tval>
void Tprint_buffer(const Tval* buf, const size_t Nx, const size_t Ny)
{
    for(size_t i = 0; i < Nx; ++i)
    {
        for(size_t j = 0; j < Ny; ++j)
        {
            std::cout << buf[i * Ny + j];
            if(j != 0)
                std::cout << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

// Print a host buffer.
void print_buffer(const transpose_params&  params,
                  const std::vector<char>& buf,
                  const size_t             nx,
                  const size_t             ny)
{
    switch(params.get_var_type())
    {
    case real_single:
        Tprint_buffer<float>((float*)buf.data(), nx, ny);
        break;
    case complex_single:
        Tprint_buffer<std::complex<float>>((std::complex<float>*)buf.data(), nx, ny);
        break;
    case real_double:
        Tprint_buffer<double>((double*)buf.data(), nx, ny);
        break;
    case complex_double:
        Tprint_buffer<std::complex<double>>((std::complex<double>*)buf.data(), nx, ny);
        break;
    default:
        throw std::runtime_error("invalid data type");
    }
}

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // Control output verbosity:
    int verbose{};

    // hip Device number for running tests:
    int deviceId{};

    // Number of performance trial samples
    int ntrial{};

    // Use hipExtLaunch function for kernel launch and timing:
    bool extLaunch = false;

    // Paramter structure for doing a transpose.
    transpose_params params;

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("transpose rider command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")
        ("length",  po::value<std::vector<size_t>>(&params.length)->multitoken(), "Lengths.")
        ("ext,e", "Use hipExtLaunchKernelGGL for launch and time kernels.")
        ("double", "Double precision transpose (default: single).");
    //clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

#if USE_LDS
    std::cout << "Using LDS.\n";
#else
    std::cout << "Not using LDS.\n";
#endif
    std::cout << "LDS bytes: " << params.lds_bytes() << std::endl;

#ifdef ROW_MAJOR
    std::cout << "Row major.\n";
#else
    std::cout << "Column major.\n";
#endif

    
    if(vm.count("ntrial"))
    {
        std::cout << "Running profile with " << ntrial << " samples\n";
    }

    if(!vm.count("length"))
    {
        std::cout << "Please specify transform length!" << std::endl;
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

    params.double_precision = vm.count("double");

    if(vm.count("length"))
    {
        std::cout << "length:";
        for(auto& i : params.length)
            std::cout << " " << i;
        std::cout << "\n";
        if(params.length.size() != 2) {
            std::cout << "You must provide exactly two lengths; exiting.\n";
        }
    }
    std::cout << "variable type: " << params.print_var_type() << std::endl;
    
    if(vm.count("ext"))
    {
        std::cout << "using ext kernel launcher\n";
        extLaunch = true;
    }

#ifdef TRANSPOSE_RUNTIME_COMPILE
    rtc_transpose(params);
#endif
    
    std::cout << "occupancy: " << params.occupancy() << std::endl;
    
    std::cout << std::endl;

    auto hip_ret = hipSuccess;

    // GPU input and output buffers:
    void* in;
    void* out;
    hip_ret = hipMalloc(&in, params.buffer_size());
    if(hip_ret != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
    hip_ret = hipMalloc(&out, params.buffer_size());
    if(hip_ret != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    // Compute input data:
    params.compute_input(in);

    hipEvent_t start, stop;
    if(hipEventCreate(&start) != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");
    if(hipEventCreate(&stop) != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");

    // Run once as a warm-up and test.
    params.ext_execute(in, out, start, stop);
    if(hipEventSynchronize(stop) != hipSuccess)
        throw std::runtime_error("hipEventSynchronize failed");

    // Host buffers:
    auto gpu_input = std::vector<char>(params.var_size() * params.length[0] * params.length[1]);
    auto gpu_output = std::vector<char>(params.var_size() * params.length[0] * params.length[1]);
    if( hipMemcpy(gpu_input.data(),
                  in,
                  gpu_input.size(),
                  hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
        
    if( hipMemcpy(gpu_output.data(),
                  out,
                  gpu_output.size(),
                  hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
        
    if(verbose > 1)
    {
        std::cout << "GPU input:\n";
        print_buffer(params, gpu_input, params.length[0], params.length[1]);
        std::cout << "GPU output:\n";
        print_buffer(params, gpu_output, params.length[1], params.length[0]);
    }

    const int nbad = check_result(gpu_input.data(), gpu_output.data(), params, verbose);
    if(verbose)
        std::cout << "number of incorrect values: " << nbad << std::endl;
    if(nbad > 0)
        std::cerr << "TRANSPOSE FAILED" << std::endl;
        
    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    for(int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
        params.compute_input(in);
        
        if(extLaunch) {
            params.ext_execute(in, out, start, stop);
        } else {
            if(hipEventRecord(start) != hipSuccess)
                throw std::runtime_error("hipEventRecord failed");
            params.execute(in, out);
            if(hipEventRecord(stop) != hipSuccess)
                throw std::runtime_error("hipEventRecord failed");
        }

        if(hipEventSynchronize(stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");

        float time = -1.0;
        if(hipEventElapsedTime(&time, start, stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failed");

        gpu_time[itrial] = time;

    }

    std::cout << "\nExecution gpu time:";
    for(const auto& i : gpu_time)
    {
        std::cout << " " << i;
    }
    std::cout << " ms" << std::endl;
    
    hip_ret = hipFree(in);
    if(hip_ret != hipSuccess)
        throw std::runtime_error("hipFree failed");
    hip_ret = hipFree(out);
    if(hip_ret != hipSuccess)
        throw std::runtime_error("hipFree failed");

#ifdef TRANSPOSE_RUNTIME_COMPILE
    kernel = nullptr;
    (void)hipModuleUnload(module);
    module = nullptr;
#endif
    
    return 0;
} 

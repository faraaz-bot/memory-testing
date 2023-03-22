#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#define XSTR(x) STR(x)
#define STR(x) #x

static constexpr auto kernel{
    R"(
#define XSTR(x) STR(x)
#define STR(x) #x
#pragma message "__clang_version__: " XSTR(__clang_version__)
extern "C"
__global__
void cosine_kernel(float* x, size_t n)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
       x[tid] *= cosf(x[tid]);
    }
}
)"};

std::string hiprtcResult_str(const hiprtcResult ret)
{
  switch(ret)
    {
    case HIPRTC_SUCCESS:
      return "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
      return "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
      return "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
      return "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
      return "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
      return "HIPRTC_ERROR_INTERNAL_ERROR";
    case HIPRTC_ERROR_LINKING:
      return "HIPRTC_ERROR_LINKING";
    }
}

struct cosine_kernel_args
{
    hipDeviceptr_t a_;
    size_t b_;
};

    
int main(int argc, char* argv[])
{
    std::cout << "RTC test code" << std::endl;

    size_t n = 1<<20;


    size_t nthread = 1;
    size_t nrepeat = 1<<14;
    int deviceId = 0;
    bool segfault = false;
    
    po::options_description opdesc("rtc sample command line options");
    opdesc.add_options()("help,h", "produces this help message")
      ("n", po::value<size_t>(&n)->default_value(1<<20), "data laneght")
      ("nthread", po::value<size_t>(&nthread)->default_value(1), "Number of omp threads")
      ("nrepeat", po::value<size_t>(&nrepeat)->default_value(1), "Number of omp threads")
      ("d", po::value<int>(&deviceId)->default_value(0), "HIP device ID.")
      ("segfault", "Call hipSetDevice in between module load and execution.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);
    
    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

    
    if(vm.count("segfault"))
    {
      segfault = true;
    }
    
    hiprtcProgram prog;
  
    int num_headers = 0;
  
    std::vector<const char*> header_names;
    std::vector<const char*> header_sources;
  
    hiprtcCreateProgram(&prog,                 // hiprtc program
                        kernel,                // kernel string
                        "gpu_kernel.cu",       // Name of the file
                        num_headers,           // Number of headers
                        &header_sources[0],    // Header sources
                        &header_names[0]);     // Name of header files

    const char* options[] = {};

    auto rtc_ret = HIPRTC_SUCCESS;
  
    rtc_ret = hiprtcCompileProgram(prog,  
                                   0,        
                                   options);
  
    if(rtc_ret != HIPRTC_SUCCESS) {
      std::stringstream ss;
      ss << "hiprtcCompileProgram failed with code ";
      ss << hiprtcResult_str(rtc_ret);
      throw std::runtime_error(ss.str());
    }
    {
      std::cout << "Host clang version: " <<  __clang_version__ << "\n";
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::cout << "compilation log:\n";
        
        if (logSize) {
            std::string log(logSize, '\0');
            hiprtcGetProgramLog(prog, &log[0]);
            std::cout << log << std::endl;

        }
    }

    size_t codeSize;
    rtc_ret = hiprtcGetCodeSize(prog, &codeSize);
    if(rtc_ret == HIPRTC_SUCCESS) {
        std::cout << "codeSize: " << codeSize << std::endl;
    } else {
        throw std::runtime_error("hiprtcGetCodeSize");
    }
  
    hipModule_t module;
    hipFunction_t kernel;

    std::vector<char> kernel_binary(codeSize);
    rtc_ret = hiprtcGetCode(prog, kernel_binary.data());
    if(rtc_ret != HIPRTC_SUCCESS) {
        throw std::runtime_error("hiprtcGetCode");
    }

    rtc_ret = hiprtcDestroyProgram(&prog);
    if(rtc_ret != HIPRTC_SUCCESS) {
        throw std::runtime_error("hiprtcDestroyProgram");
    }

    if(!segfault) {
      // Calling here is OK.
      if(hipSetDevice(deviceId)  != hipSuccess) {
        throw std::runtime_error("hipSetDevice");
      }
    }
  
    
    auto hip_ret = hipModuleLoadData(&module, kernel_binary.data());
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleLoadData");
    }
  
    hip_ret = hipModuleGetFunction(&kernel, module, "cosine_kernel");
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleGetFunction");
    }

    if(segfault) {
      // Calling here produces a segfault.
      if(hipSetDevice(deviceId)  != hipSuccess) {
        throw std::runtime_error("hipSetDevice");
      }
    }   

    // Number of data points that we will output:
    const size_t nshow = std::min(n, (size_t)8);
    
    std::vector<float> hX(n);
    for(size_t i = 0; i < hX.size(); ++i) {
        hX[i] = (float)i; 
    }

    std::cout << "input:";
    for(size_t i = 0; i < nshow; ++i) {
        std::cout << " " << hX[i];
    }
    std::cout << std::endl;
    
    const size_t bufferSize = hX.size() * sizeof(float);
    auto size = sizeof(cosine_kernel_args);
    
    std::vector<hipDeviceptr_t> dX(nthread);
    std::vector<cosine_kernel_args> args(nthread);

    for(size_t ithread = 0; ithread < nthread; ++ithread) {
        hip_ret = hipMalloc((void **)&dX[ithread], bufferSize);
        if(hip_ret != hipSuccess) {
            throw std::runtime_error("hipMalloc");
        }

        hip_ret = hipMemcpyHtoD(dX[ithread], hX.data(), bufferSize);
        if(hip_ret != hipSuccess) {
            throw std::runtime_error("hipMemcpyHtoD");
        }    

        args[ithread].a_ = dX[ithread];
        args[ithread].b_ = n;
    }

#pragma omp parallel for num_threads(nthread)
    for(size_t ithread = 0; ithread < nthread; ++ithread) {
        int tid = omp_get_thread_num();
        std::cout << "thread " << tid << std::endl;
        
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args[ithread],
            HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
            HIP_LAUNCH_PARAM_END};

        size_t nthreads = 1024;
        size_t nblocks = (n + nthreads) / nthreads;

        for(size_t irepeat = 0; irepeat < nrepeat; ++irepeat) {
            hip_ret = hipModuleLaunchKernel(kernel, nblocks, 1, 1, nthreads, 1, 1,
                                            0, nullptr, nullptr, config);
            if(hip_ret != hipSuccess) {
                throw std::runtime_error("hipModuleLaunchKernel");
            }
        }
    }
    
    for(size_t ithread = 0; ithread < nthread; ++ithread) {
        hip_ret = hipMemcpyDtoH(hX.data(), dX[ithread], bufferSize);
        if(hip_ret != hipSuccess) {
            throw std::runtime_error("hipMemcpyDtoH");
        }    

        std::cout << "output " << ithread << ":";
        for(size_t i = 0; i < nshow; ++i) {
            std::cout << " " << hX[i];
        }
        std::cout << std::endl;
    }
    
    // Clean up
    hip_ret = hipModuleUnload(module);
    if(hip_ret != hipSuccess) {
      throw std::runtime_error("hipModuleUnload");
    }
    
    for(size_t ithread = 0; ithread < nthread; ++ithread) {
      hip_ret = hipFree((void *)dX[ithread]);
      if(hip_ret != hipSuccess) {
	throw std::runtime_error("hipFree");
      }
    }
  
    return 0;
}

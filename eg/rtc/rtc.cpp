#include <iostream>
#include <vector>
#include <algorithm>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>


static constexpr auto kernel{
    R"(
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

int main()
{
    std::cout << "RTC test code" << std::endl;

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
        std::cout << "compile failed" << std::endl;
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);

        if (logSize) {
            std::string log(logSize, '\0');
            hiprtcGetProgramLog(prog, &log[0]);
            // Corrective action with logs
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

    auto hip_ret = hipModuleLoadData(&module, kernel_binary.data());
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleLoadData");
    }
  
    hip_ret = hipModuleGetFunction(&kernel, module, "cosine_kernel");
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleGetFunction");
    }

    const size_t n = 1<<8;

    // Number of data points that we will output:
    const size_t nshow = std::min(n, (size_t)16);
    
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
    
    hipDeviceptr_t dX;
    hip_ret = hipMalloc((void **)&dX, bufferSize);
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipMalloc");
    }

    hip_ret = hipMemcpyHtoD(dX, hX.data(), bufferSize);
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipMemcpyHtoD");
    }    

    struct {
        hipDeviceptr_t a_;
        size_t b_;
    } args{dX, n};

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    size_t nthreads = 1024;
    size_t nblocks = (n + nthreads) / nthreads;
    
    hip_ret = hipModuleLaunchKernel(kernel, nblocks, 1, 1, nthreads, 1, 1,
                                    0, nullptr, nullptr, config);
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleLaunchKernel");
    }

    hip_ret = hipMemcpyDtoH(hX.data(), dX, bufferSize);
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipMemcpyDtoH");
    }    

    std::cout << "output:";
    for(size_t i = 0; i < nshow; ++i) {
        std::cout << " " << hX[i];
    }
    std::cout << std::endl;
        
    // Clean up
    hipFree((void *)dX);
    hipModuleUnload(module);
  
    return 0;
}

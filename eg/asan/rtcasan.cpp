#include<iostream>
#include<vector>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

static constexpr auto kernelstr{
    R"(
#define XSTR(x) STR(x)
#define STR(x) #x
#pragma message "__clang_version__: " XSTR(__clang_version__)
extern "C"
__global__
void set1(int* p)
{
     const int i = blockDim.x*blockIdx.x + threadIdx.x;
     p[i] = 1;
}
)"};

struct set1_kernel_args
{
    hipDeviceptr_t a_;
};

int main(int argc, char **argv)
{    
    std::cout << "rtc address sanitizer example\n";
    
    // Number of ints allocated on device:
    int m = 32;//std::atoi(argv[1]);

    // grid dim:
    int n1 = 2; //std::atoi(argv[2]);

    // blocksize: 
    int n2 = 16; //std::atoi(argv[3]);

    // Number of ints allocated on host:
    int c = 32;//std::atoi(argv[4]);

    std::cout << "device size m: " << m << "\n";
    std::cout << "host size c:   " << c << "\n";
    std::cout << "running " << n1 << " blocks of " << n2 << " threads\n";

    auto rtc_ret = HIPRTC_SUCCESS;
    
    hiprtcProgram prog;
    int num_headers = 0;
    std::vector<const char*> header_names;
    std::vector<const char*> header_sources;
    rtc_ret = hiprtcCreateProgram(&prog,                 // hiprtc program
                                  kernelstr,             // kernel string
                                  "gpu_kernel.cu",       // Name of the file
                                  num_headers,           // Number of headers
                                  header_sources.data(), // Header sources
                                  header_names.data());  // Name of header files
    if(rtc_ret != HIPRTC_SUCCESS) {
        std::cout << "create failed" << std::endl;
        throw std::runtime_error("hiprtcCreateProgram");
    }
    
    const char* options[] = {};
    rtc_ret = hiprtcCompileProgram(prog,
                                   0,
                                   options);
    if(rtc_ret != HIPRTC_SUCCESS) {
      throw std::runtime_error("compile failed");
    }
    
    size_t codeSize = 0;
    rtc_ret = hiprtcGetCodeSize(prog, &codeSize);
    if(rtc_ret == HIPRTC_SUCCESS) {
        std::cout << "codeSize: " << codeSize << std::endl;
    } else {
        throw std::runtime_error("hiprtcGetCodeSize");
    }
  
    std::vector<char> kernel_binary(codeSize);
    rtc_ret = hiprtcGetCode(prog, kernel_binary.data());
    if(rtc_ret != HIPRTC_SUCCESS) {
        throw std::runtime_error("hiprtcGetCode");
    }
    
    rtc_ret = hiprtcDestroyProgram(&prog);
    if(rtc_ret != HIPRTC_SUCCESS) {
        throw std::runtime_error("hiprtcDestroyProgram");
    }

    hipModule_t kernel_module;
    auto hip_ret = hipModuleLoadData(&kernel_module, kernel_binary.data());
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleLoadData");
    }
      
    hipFunction_t kernel_function;
    hip_ret = hipModuleGetFunction(&kernel_function, kernel_module, "set1");
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleGetFunction");
    }
   
    // Device pointers
    int *dp = nullptr;

    if(hipMalloc(&dp, m*sizeof(int)) != hipSuccess)
    {
        throw std::runtime_error("hipMalloc failed");
    }

    //hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);

    set1_kernel_args args;
    args.a_ = dp;
    auto argsize = sizeof(set1_kernel_args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsize,
        HIP_LAUNCH_PARAM_END};
    
    hip_ret = hipModuleLaunchKernel(kernel_function,
                                    n1, 1, 1, // grid dims
                                    n2, 1, 1, // block dims
                                    0, // shared mem
                                    nullptr, // stream
                                    nullptr, // kernelParams (not yet implemented)
                                    config); // extra
    if(hip_ret != hipSuccess) {
        throw std::runtime_error("hipModuleLaunchKernel");
    }

    std::vector<int> hp(c);
    if( hipMemcpy(hp.data(), dp, m*sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    {
        throw std::runtime_error("hipMemcpy failed");
    }
    for(const auto & i : hp) {
        std::cout << i << " ";
    }
    std::cout << "\n";
        
    
    if(hipDeviceSynchronize() != hipSuccess)
    {
        throw std::runtime_error("hipDeviceSynchronize failed");
    }
    if(hipFree(dp) != hipSuccess)
    {
        throw std::runtime_error("hipFree failed");
    }

    std::puts("Done.");
    return 0;
}

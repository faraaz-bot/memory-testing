#include<iostream>
#include<vector>
#include<sstream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

static constexpr auto kernelstr{
    R"(
#define XSTR(x) STR(x)
#define STR(x) #x
//#pragma message "__clang_version__: " XSTR(__clang_version__)
#warning ("hip arch: " XSTR(__HIP_ARCH__))
//#message ("__HIP_DEVICE_COMPILE__: " XSTR(__HIP_DEVICE_COMPILE__))
#pragma message ( STR(__amdgcn_target_id__))
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

    int m{};
    int n1{};
    int n2{};
    int c{};
    
    po::options_description opdesc("asan sample command line options");
        opdesc.add_options()("help,h", "produces this help message")
        ("m", po::value<int>(&m)->default_value(32), "device buffer allocation length")
        ("n1", po::value<int>(&n1)->default_value(2), "grid dim")
        ("n2", po::value<int>(&n2)->default_value(16), "thread block dim")
        ("c", po::value<int>(&c)->default_value(32), "host buffer allocation length");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }
    
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
    
    std::vector<const char*> options;
    //options.push_back("-O3");
    //options.push_back("-g");
    //options.push_back("-std=c++14");

    // NB: "gfx90a:xnack-" gives 
    //std::string gpu_arch = "gfx90a:xnack-";
    std::string gpu_arch = "gfx90a:sramecc+:xnack+";
    std::string gpu_arch_arg = "--gpu-architecture=" + gpu_arch;
    options.push_back(gpu_arch_arg.c_str());

    options.push_back("-fsanitize=address");
    
    rtc_ret = hiprtcCompileProgram(prog,
                                   options.size(),
                                   options.data());
    size_t logSize = 0;
    hiprtcGetProgramLogSize(prog, &logSize);
    std::cout << "compilation log:\n";
    if (logSize) {
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        std::cout << log << std::endl;
    }
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
        std::stringstream ss;
        ss << "hipModuleLoadData error: " << hip_ret << " " << hipGetErrorString(hip_ret);
        throw std::runtime_error( ss.str().c_str() );
    }
      
    hipFunction_t kernel_function;
    hip_ret = hipModuleGetFunction(&kernel_function, kernel_module, "set1");
    if(hip_ret != hipSuccess) {
        std::stringstream ss;
        ss << "hipModuleGetFunction error: " << hip_ret << " " << hipGetErrorString(hip_ret);
        throw std::runtime_error(ss.str().c_str());
    }
   
    // Device pointers
    int *dp = nullptr;

    if(hipMalloc(&dp, m * sizeof(int)) != hipSuccess)
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
    if( hipMemcpy(hp.data(), dp, m * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
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

// hipcc rtc.cpp -fopenmp  && time ./a.out


#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

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

struct cosine_kernel_args
{
    hipDeviceptr_t a_;
    size_t b_;
};

    
int main()
{
    std::cout << "RTC test code" << std::endl;

    const size_t n = 1<<20;
    const size_t nthread = 16;
    const size_t nrepeat = 1<<14;
    
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
            std::cout << log << std::endl;
            throw std::runtime_error("hiprtcCompileProgram");
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
    hipModuleUnload(module);
    for(size_t ithread = 0; ithread < nthread; ++ithread) {
        hipFree((void *)dX[ithread]);
    }
  
    return 0;
}

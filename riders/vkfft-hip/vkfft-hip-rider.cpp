#include <iostream>


#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

#include "vkFFT.h"
#include "benchmark_scripts/vkFFT_scripts/include/utils_VkFFT.h"

#include "fft_params.h"

int main()
{
    fft_params params;

    params.length = {8};
    params.nbatch = 1;

    std::cout << params.token() << std::endl;

    if(hipInit(0) != hipSuccess)
    {
        throw std::runtime_error("hipInit failed");
    }

    VkGPU vkGPU = {};
    vkGPU.device = 0;
    
    if(hipSetDevice((int)vkGPU.device_id) != hipSuccess)
    {
        throw std::runtime_error("hipSetDevice failed");
    }
    if( hipDeviceGet(&vkGPU.device, (int)vkGPU.device_id) != hipSuccess)
    {
        throw std::runtime_error("hipGetDevice failed");
    }
    if(hipCtxCreate(&vkGPU.context, 0, (int)vkGPU.device) != hipSuccess)
    {
        throw std::runtime_error("hipCtxCreate failed");
    }
    
    // TODO: set all params data.

    VkFFTConfiguration configuration = {};
    
    // So, looks like vkFFT ignores the fft dim, and actually looks at all of the 3 dims, so set
    // them to one by default.
    configuration.size[0] = 1;
    configuration.size[1] = 1;
    configuration.size[2] = 1;
    
    configuration.FFTdim = params.length.size();
    for (int i = 0; i< params.length.size(); ++i) {
        configuration.size[i] = params.length[i];
    }
    configuration.numberBatches = params.nbatch;

    // No discrete cosine transform.
    configuration.performDCT = false;
    
    configuration.performR2C = (params.transform_type == fft_transform_type_real_forward ||
                                params.transform_type == fft_transform_type_real_inverse);
    
    configuration.doublePrecision = params.precision == fft_precision_single ? 0 : 1;

    configuration.disableReorderFourStep = 0;
    configuration.registerBoost = 0;
    //configuration.isCompilerInitialized = 0;
    
    configuration.device = &vkGPU.device;

    const size_t storageComplexSize = params.precision == fft_precision_double ? sizeof(std::complex<double>) : sizeof(std::complex<float>);
    
    // Allocate buffer for the input data.
    // Assumed contiguous for now.
    uint64_t bufferSize = 0;
    if (params.transform_type == fft_transform_type_real_forward || params.transform_type == fft_transform_type_real_inverse) {
        bufferSize = (uint64_t)(storageComplexSize / 2) * (configuration.size[0] + 2)
            * configuration.size[1] * configuration.size[2] * configuration.numberBatches;
    }
    else {
        bufferSize = (uint64_t)storageComplexSize
            * configuration.size[0] * configuration.size[1] * configuration.size[2] * configuration.numberBatches;
    }
             
    hipDoubleComplex* buffer = 0;
    if( hipMalloc((void**)&buffer, bufferSize) != hipSuccess)
    {
        throw std::runtime_error("hipMalloc failed");
    }
    configuration.buffer = (void**)&buffer;
                        
    configuration.bufferSize = &bufferSize;

    VkFFTLaunchParams launchParams = {};
    
    std::vector<std::complex<double>> idata(std::accumulate(params.length.begin(),
                                                            params.length.end(),
                                                            static_cast<size_t>(1),
                                                            std::multiplies<size_t>()));
    // Intitilize the 1D data for now.
    for(int i = 0; i < params.length[0]; ++i)
    {
        idata[i] = std::complex<double>(i, i);
    }
    std::cout << "input:";
    for(const auto& val : idata)
    {
        std::cout << " " << val;
    }
    std::cout << std::endl;
    if(hipMemcpy(buffer, idata.data(), idata.size() * sizeof(decltype(idata)::value_type), hipMemcpyHostToDevice) !=  hipSuccess)
    {
        throw std::runtime_error("hipMemcpy failed");
    }

    VkFFTApplication app = {};
    if(initializeVkFFT(&app, configuration) !=  VKFFT_SUCCESS)
    {
        throw std::runtime_error("initializeVkFFT failed");
    }
    
    // -1 for forward.
    if(VkFFTAppend(&app, -1, &launchParams) !=  VKFFT_SUCCESS)
    {
        throw std::runtime_error("VkFFTAppend failed");
    }


    std::vector<std::complex<double>> odata(std::accumulate(params.length.begin(),
                                                            params.length.end(),
                                                            static_cast<size_t>(1),
                                                            std::multiplies<size_t>()));
    if(hipMemcpy(odata.data(), buffer, odata.size() * sizeof(decltype(odata)::value_type), hipMemcpyDeviceToHost) !=  hipSuccess)
    {
        throw std::runtime_error("hipMemcpy failed");
    }

    std::cout << "output:";
    for(const auto& val : odata)
    {
        std::cout << " " << val;
    }
    std::cout << std::endl;
    
    return 0;
}

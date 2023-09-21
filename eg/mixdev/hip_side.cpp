#include <iostream>
#include <vector>
#include <complex>
#include <hip/hip_runtime_api.h>
#include "hipfft/hipfftXt.h"

std::vector<std::vector<std::complex<double>>> hip_data(const int Nx,
                                                        const int Ny,
                                                        std::vector<int> &gpus,
                                                        std::vector<std::complex<double>> & input)
{
    std::cout << "cufftXt version\n";    


    hipLibXtDesc* desc; // input descriptor

    
    hipfftHandle plan      = nullptr;

    auto hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");

    // Define which GPUs are to be used
     hipfft_rt = hipfftXtSetGPUs(plan, gpus.size(), gpus.data());
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtSetGPUs failed.");

    // Create the plan
    size_t workSize[gpus.size()];
    hipfft_rt = hipfftMakePlan2d(plan, Nx, Ny, HIPFFT_Z2Z, workSize);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftMakePlan2d failed.");
    
    // Copy input data to GPUs
    hipfftXtSubFormat_t format = HIPFFT_XT_FORMAT_INPLACE;
    hipfft_rt                  = hipfftXtMalloc(plan, &desc, format);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMalloc failed.");
    
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(desc),
                               reinterpret_cast<void*>(input.data()),
                               HIPFFT_COPY_HOST_TO_DEVICE);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtMemcpy H2D failed.");

    // Execute the plan
    hipfft_rt = hipfftXtExecDescriptor(plan, desc, desc, -1);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftXtExecDescriptor failed.");


    // Put the gathered data in the last vector.  Yeah, it's a hack.
    std::vector<std::vector<std::complex<double>>> outs(gpus.size() + 1);
    for(int idx = 0; idx < gpus.size(); ++idx) {
        const size_t bufsize = desc->descriptor->size[idx];
        std::cout << "idx: " << idx << " size: " << bufsize << "\n";
        const size_t bufcount = bufsize / sizeof(std::complex<double>);
        outs[idx].resize(bufcount);
        
        auto hipret = hipMemcpy(outs[idx].data(),
                                desc->descriptor->data[idx],
                                bufsize,
                                hipMemcpyDeviceToHost);
        if(hipret != hipSuccess) {
            std::stringstream ss;
            ss << "hipMemcpy D2H failed with code ";
            ss << hipret;
            throw std::runtime_error(ss.str());
        }
    }
    

    outs[outs.size() - 1].resize(Nx * Ny);
    hipfft_rt = hipfftXtMemcpy(plan,
                               reinterpret_cast<void*>(outs[outs.size() - 1].data()),
                               reinterpret_cast<void*>(desc),
                               HIPFFT_COPY_DEVICE_TO_HOST);
    
    // FIXME: free things
    
    return outs;
}

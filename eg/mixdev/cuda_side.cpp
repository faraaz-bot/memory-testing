#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <complex>
#include<type_traits>

std::vector<std::vector<std::complex<double>>> cuda_data(const int Nx,
                                                         const int Ny,
                                                         std::vector<int> &gpus,
                                                         std::vector<std::complex<double>> & input)
{
    std::cout << "cufftXt version\n";


    cudaLibXtDesc* desc; // input descriptor

    
    cufftHandle plan;

    auto cufft_rt = cufftCreate(&plan);
    if(cufft_rt != CUFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");

    // Define which GPUs are to be used
     cufft_rt = cufftXtSetGPUs(plan, gpus.size(), gpus.data());
    if(cufft_rt != CUFFT_SUCCESS)
        throw std::runtime_error("cufftXtSetGPUs failed.");

    // Create the plan
    size_t workSize[gpus.size()];
    cufft_rt = cufftMakePlan2d(plan, Nx, Ny, CUFFT_Z2Z, workSize);
    if(cufft_rt != CUFFT_SUCCESS)
        throw std::runtime_error("cufftMakePlan2d failed.");
    
    // Copy input data to GPUs
    cufftXtSubFormat_t format = CUFFT_XT_FORMAT_INPLACE;
    cufft_rt                  = cufftXtMalloc(plan, &desc, format);
    if(cufft_rt != CUFFT_SUCCESS)
        throw std::runtime_error("cufftXtMalloc failed.");
    
    cufft_rt = cufftXtMemcpy(plan,
                               reinterpret_cast<void*>(desc),
                               reinterpret_cast<void*>(input.data()),
                               CUFFT_COPY_HOST_TO_DEVICE);
    if(cufft_rt != CUFFT_SUCCESS)
        throw std::runtime_error("cufftXtMemcpy H2D failed.");

    // Execute the plan
    cufft_rt = cufftXtExecDescriptor(plan, desc, desc, -1);
    if(cufft_rt != CUFFT_SUCCESS)
        throw std::runtime_error("cufftXtExecDescriptor failed.");


    // Put the gathered data in the last vector.  Yeah, it's a hack.
    std::vector<std::vector<std::complex<double>>> outs(gpus.size() + 1);
    for(int idx = 0; idx < gpus.size(); ++idx) {
        const size_t bufsize = desc->descriptor->size[idx];
        std::cout << "idx: " << idx << " size: " << bufsize << "\n";
        const size_t bufcount = bufsize / sizeof(std::complex<double>);
        outs[idx].resize(bufcount);
        
        auto cudaret = cudaMemcpy(outs[idx].data(),
                                  desc->descriptor->data[idx],
                                  bufsize,
                                  cudaMemcpyDeviceToHost);

        if(cudaret != cudaSuccess) {
            std::stringstream ss;
            ss << "cudaMemcpy D2H failed with code ";
            ss << cudaret;
            throw std::runtime_error(ss.str());
        }
    }
    
    outs[outs.size() - 1].resize(Nx * Ny);
    cufft_rt = cufftXtMemcpy(plan,
                               reinterpret_cast<void*>(outs[outs.size() - 1].data()),
                             reinterpret_cast<void*>(desc),
                             CUFFT_COPY_DEVICE_TO_HOST);
    
    // FIXME: free things
    
    return outs;
}

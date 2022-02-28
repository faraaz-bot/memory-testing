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

#include "vkfft_params.h"

#include "gpubuf.h"

int verbose = 3;

int main()
{
    vkfft_params params;

    params.length = {8};
    params.nbatch = 1;

    params.validate();
    
    std::cout << params.token() << std::endl;

    if(hipInit(0) != hipSuccess)
    {
        throw std::runtime_error("hipInit failed");
    }

    if(params.create_plan() != fft_status_success)
    {
        throw std::runtime_error("plan creation failed");
    }
    
    // Input data:
    auto gpu_input = allocate_host_buffer(params.precision, params.itype, params.isize);
    compute_input(params, gpu_input);
    if(verbose > 1)
    {
        std::cout << "GPU input:\n";
        params.print_ibuffer(gpu_input);
    }
    
    // GPU input and output buffers:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        if( ibuffer[i].alloc(ibuffer_sizes[i]) != hipSuccess)
        {
            throw std::runtime_error("ibuffer alloc failed");
        }
        pibuffer[i] = ibuffer[i].data();
    }
    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    if(params.placement == fft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            if(obuffer_data[i].alloc(obuffer_sizes[i]) != hipSuccess)
            {
                throw std::runtime_error("obuffer alloc failed");
            }
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }
    
    // Warm up once:
    for(int idx = 0; idx < gpu_input.size(); ++idx)
    {
        if(hipMemcpy(pibuffer[idx],
                     gpu_input[idx].data(),
                     gpu_input[idx].size(),
                     hipMemcpyHostToDevice)
           != hipSuccess)
        {
                throw std::runtime_error("obuffer alloc failed");
        }
    }
    
    if(params.execute(pibuffer.data(), pobuffer.data()) != fft_status_success)
    {
        throw std::runtime_error("exec failed");
    }
    
    if(verbose > 2)
    {
        auto output = allocate_host_buffer(params.precision, params.otype, params.osize);
        for(int idx = 0; idx < output.size(); ++idx)
        {
            if( hipMemcpy(output[idx].data(),
                          pobuffer[idx],
                          output[idx].size(),
                          hipMemcpyDeviceToHost)
                != hipSuccess)
            {
                throw std::runtime_error("obuffer hipMemcpy failed");
            }
        }
        std::cout << "GPU output:\n";
        params.print_obuffer(output);
    }

    
    return 0;
}

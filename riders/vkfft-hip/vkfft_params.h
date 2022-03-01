#ifndef VKFFT_PARAMS_H
#define VKFFT_PARAMS_H

#include "fft_params.h"
#include "vkFFT.h"

class vkfft_params : public fft_params
{
private:
    uint64_t bufferSize = 0;
    
public:

    explicit vkfft_params(){};

    explicit vkfft_params(const fft_params& p)
        : fft_params(p){};
    
    vkfft_params(const vkfft_params&) = delete;
    vkfft_params& operator=(const vkfft_params&) = delete;
    
    VkFFTConfiguration configuration = {};
    VkGPU vkGPU = {};
    VkFFTApplication app = {};
    

    fft_status create_plan() override
    {    
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
        

        

	configuration.FFTdim = length.size();
            
	// So, looks like vkFFT ignores the fft dim, and actually looks at all of the 3 dims, so set
	// them to one by default.
	configuration.size[0] = 1;
	configuration.size[1] = 1;
	configuration.size[2] = 1;
	for (int i = 0; i < length.size(); ++i) {
	    configuration.size[i] = length[i];
	}
        
	configuration.isInputFormatted = 1;
	configuration.inputBufferStride[0] = 1;
	configuration.inputBufferStride[1] = 1;
	configuration.inputBufferStride[2] = 1;
	for (int i = 0; i < istride.size(); ++i) {
	    configuration.inputBufferStride[i] = istride[i];
	}

	configuration.isOutputFormatted = 1;
	configuration.outputBufferStride[0] = 1;
	configuration.outputBufferStride[1] = 1;
	configuration.outputBufferStride[2] = 1;
	for (int i = 0; i < istride.size(); ++i) {
	    configuration.outputBufferStride[i] = ostride[i];
	}
        
	configuration.numberBatches = nbatch;

	// No discrete cosine transform.
	configuration.performDCT = false;
    
	configuration.performR2C = (transform_type == fft_transform_type_real_forward ||
				    transform_type == fft_transform_type_real_inverse);
    
	configuration.doublePrecision = precision == fft_precision_single ? 0 : 1;

	configuration.disableReorderFourStep = 0;
	configuration.registerBoost = 0;
	//configuration.isCompilerInitialized = 0;
    
	configuration.device = &vkGPU.device;


	const size_t storageComplexSize = precision == fft_precision_double
	    ? sizeof(std::complex<double>) : sizeof(std::complex<float>);
    
	// Allocate buffer for the input data.
	// Assumed contiguous for now.
	uint64_t bufferSize = 0;
	if (transform_type == fft_transform_type_real_forward
	    || transform_type == fft_transform_type_real_inverse) {
	    bufferSize = (uint64_t)(storageComplexSize / 2) * (configuration.size[0] + 2)
		* configuration.size[1] * configuration.size[2] * configuration.numberBatches;
	}
	else {
	    bufferSize = (uint64_t)storageComplexSize
		* configuration.size[0] * configuration.size[1] * configuration.size[2] * configuration.numberBatches;
	}
             

	configuration.bufferSize = &bufferSize;
        
	if(initializeVkFFT(&app, configuration) !=  VKFFT_SUCCESS)
	{
	    throw std::runtime_error("initializeVkFFT failed");
	}
	return fft_status_success;
    }
    
    virtual fft_status execute(void** in, void** out) override
    {
	return execute(in[0], out[0]);
    };

    fft_status execute(void* ibuffer, void* obuffer)
    {
	VkFFTLaunchParams launchParams = {};
	launchParams.buffer = (void**)&ibuffer;
	if(placement == fft_placement_notinplace)
	{
	    launchParams.outputBuffer = (void**)&obuffer;
	}
                
	const int direction = (transform_type == fft_transform_type_complex_forward ||
			       transform_type == fft_transform_type_real_forward ) ? -1 : 1;
        
	if(VkFFTAppend(&app, direction, &launchParams) ==  VKFFT_SUCCESS)
	    return fft_status_success;
        
	return fft_status_failure;
    }
};

#endif 

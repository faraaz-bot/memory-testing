// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include <boost/program_options.hpp>
namespace po = boost::program_options;


//#include "library/include/rocfft.h"
#include "clients/fft_params.h"
#include "shared/gpubuf.h"

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>


template<class Tcomplex>
__global__ void trivial_transform_c2c(const Tcomplex* in, Tcomplex* out, const size_t N)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
      out[idx].x = in[idx].x * N;
      out[idx].y = in[idx].y * N;
    }
}

__global__ void trivial_transform_r2c_float(const float* in, float2* out, const size_t halfN)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
      out[idx].x = in[2 * idx] * halfN;
      out[idx].y = in[2 * idx + 1] * halfN;
    }
}

__global__ void trivial_transform_r2c_double(const double* in, double2* out, const size_t halfN)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
      out[idx].x = in[2 * idx] * halfN; 
      out[idx].y = in[2 * idx + 1] * halfN;
    }
}

template<class Tfloat, class Tcomplex>
__global__ void trivial_transform_r2c_double(const Tfloat* in, Tcomplex* out, const size_t halfN)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
      out[2 * idx] = out[idx].x * halfN; 
      out[2 * idx + 1] = out[idx].y * halfN; 
    }
}

 __global__ void trivial_transform_c2r_float(const float2* in, float* out, const size_t halfN)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
      out[2 * idx] = in[idx].x * halfN; 
      out[2 * idx + 1] = in[idx].y * halfN; 
    }
}

 __global__ void trivial_transform_c2r_double(const double2* in, double* out, const size_t halfN)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
      out[2 * idx] = in[idx].x * halfN; 
      out[2 * idx + 1] = in[idx].y * halfN; 
    }
}

class hipbw_params : public fft_params
{
public:
    hipbw_params(){};

    hipbw_params(const fft_params& p)
        : fft_params(p){};

    ~hipbw_params(){};
 
    size_t vram_footprint() override
    {
      return 0;
    }
  
  fft_status create_plan() override
  {
    return fft_status_success;
  }
  
  virtual fft_status execute(void** in, void** out) override
  {
    using Tlength = decltype(length)::value_type;
    const size_t N = std::accumulate(length.begin(), length.end(),  (Tlength)nbatch,
				     std::multiplies<Tlength>());
    size_t blockSize = 256;
    size_t blocks    = (N + blockSize - 1) / blockSize;

    switch(precision) {
      case fft_precision_single:
	switch(transform_type) {
	  case fft_transform_type_complex_forward:
	  case fft_transform_type_complex_inverse:
	    hipLaunchKernelGGL(trivial_transform_c2c<float2>,
			       dim3(blocks),
			       dim3(blockSize),
			       0, // sharedMemBytes
			       0, // stream
			       (float2*)in[0],
			       (float2*)out[0],
			       N);
	    break;
	  case fft_transform_type_real_forward:
	    hipLaunchKernelGGL(trivial_transform_r2c_float,
	    		       dim3(blocks),
	    		       dim3(blockSize),
	    		       0, // sharedMemBytes
	    		       0, // stream
	    		       (float*)in[0],
	    		       (float2*)out[0],
	    		       N / 2);
	    break;
	  case fft_transform_type_real_inverse:
	    hipLaunchKernelGGL(trivial_transform_c2r_float,
	    		       dim3(blocks),
	    		       dim3(blockSize),
	    		       0, // sharedMemBytes
	    		       0, // stream
	    		       (float2*)in[0],
	    		       (float*)out[0],
	    		       N / 2);
	    break;
	}
	break;
      case fft_precision_double:
	switch(transform_type)  {
	  case fft_transform_type_complex_forward:
	  case fft_transform_type_complex_inverse:
	      hipLaunchKernelGGL(trivial_transform_c2c<double2>,
			       dim3(blocks),
			       dim3(blockSize),
			       0, // sharedMemBytes
			       0, // stream
			       (double2*)in[0],
			       (double2*)out[0],
			       N);

	      break;
	  case fft_transform_type_real_forward:
	    hipLaunchKernelGGL(trivial_transform_r2c_double,
			       dim3(blocks),
			       dim3(blockSize),
			       0, // sharedMemBytes
			       0, // stream
			       (double*)in[0],
			       (double2*)out[0],
			       N / 2);
	    break;
	  case fft_transform_type_real_inverse:
	    hipLaunchKernelGGL(trivial_transform_c2r_double,
	    		       dim3(blocks),
	    		       dim3(blockSize),
	    		       0, // sharedMemBytes
	    		       0, // stream
	    		       (double2*)in[0],
	    		       (double*)out[0],
	    		       N / 2);
	    break;
	}
	break;
    }

    // FIXME: implement
    return fft_status_success;
  };
  
};

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // Control output verbosity:
    int verbose{};

    // hip Device number for running tests:
    int deviceId{};

    // Number of performance trial samples
    int ntrial{};

    // test parameters:
    hipbw_params params;

    // Token string to fully specify fft params.
    std::string token;
    // Declare the supported options.

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("hipbw rider command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the hipbw library")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (default: single)")
        ("transformType,t", po::value<fft_transform_type>(&params.transform_type)
         ->default_value(fft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "batchSize,b", po::value<size_t>(&params.nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", po::value<fft_array_type>(&params.itype)
          ->default_value(fft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<fft_array_type>(&params.otype)
          ->default_value(fft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("length",  po::value<std::vector<size_t>>(&params.length)->multitoken(), "Lengths.")
        ("istride", po::value<std::vector<size_t>>(&params.istride)->multitoken(), "Input strides.")
        ("ostride", po::value<std::vector<size_t>>(&params.ostride)->multitoken(), "Output strides.")
        ("idist", po::value<size_t>(&params.idist)->default_value(0),
         "Logical distance between input batches.")
        ("odist", po::value<size_t>(&params.odist)->default_value(0),
         "Logical distance between output batches.")
        ("isize", po::value<std::vector<size_t>>(&params.isize)->multitoken(),
         "Logical size of input buffer.")
        ("osize", po::value<std::vector<size_t>>(&params.osize)->multitoken(),
         "Logical size of output buffer.")
        ("ioffset", po::value<std::vector<size_t>>(&params.ioffset)->multitoken(), "Input offsets.")
        ("ooffset", po::value<std::vector<size_t>>(&params.ooffset)->multitoken(), "Output offsets.")
        ("token", po::value<std::string>(&token));
    //clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return EXIT_SUCCESS;
    }

    if(vm.count("ntrial"))
    {
        std::cout << "Running profile with " << ntrial << " samples\n";
    }

    if(token != "")
    {
        std::cout << "Reading fft params from token:\n" << token << std::endl;

        try
        {
            params.from_token(token);
        }
        catch(...)
        {
            std::cout << "Unable to parse token." << std::endl;
            return 1;
        }
    }
    else
    {
        if(!vm.count("length"))
        {
            std::cout << "Please specify transform length!" << std::endl;
            std::cout << opdesc << std::endl;
            return EXIT_SUCCESS;
        }

        params.placement
            = vm.count("notInPlace") ? fft_placement_notinplace : fft_placement_inplace;
        params.precision = vm.count("double") ? fft_precision_double : fft_precision_single;

        if(vm.count("notInPlace"))
        {
            std::cout << "out-of-place\n";
        }
        else
        {
            std::cout << "in-place\n";
        }

        if(vm.count("length"))
        {
            std::cout << "length:";
            for(auto& i : params.length)
                std::cout << " " << i;
            std::cout << "\n";
        }

        if(vm.count("istride"))
        {
            std::cout << "istride:";
            for(auto& i : params.istride)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(vm.count("ostride"))
        {
            std::cout << "ostride:";
            for(auto& i : params.ostride)
                std::cout << " " << i;
            std::cout << "\n";
        }

        if(params.idist > 0)
        {
            std::cout << "idist: " << params.idist << "\n";
        }
        if(params.odist > 0)
        {
            std::cout << "odist: " << params.odist << "\n";
        }

        if(vm.count("ioffset"))
        {
            std::cout << "ioffset:";
            for(auto& i : params.ioffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
        if(vm.count("ooffset"))
        {
            std::cout << "ooffset:";
            for(auto& i : params.ooffset)
                std::cout << " " << i;
            std::cout << "\n";
        }
    }

    std::cout << std::flush;

    // Fixme: set the device id properly after the IDs are synced
    // bewteen hip runtime and rocm-smi.
    // HIP_V_THROW(hipSetDevice(deviceId), "set device failed!");

    params.validate();

    if(!params.valid(verbose))
    {
        throw std::runtime_error("Invalid parameters, add --verbose=1 for detail");
    }


    std::cout << "Token: " << params.token() << std::endl;
    if(verbose)
    {
        std::cout << params.str(" ") << std::endl;
    }
    
    if(verbose)
    {
        std::cout << params.str() << std::endl;
    }

    // Input data:
    auto gpu_input = allocate_host_buffer(params.precision, params.itype, params.isize);
    compute_input(params, gpu_input);

auto hip_ret = hipSuccess;

    // GPU input and output buffers:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
      auto ret = ibuffer[i].alloc(ibuffer_sizes[i]);
      if(ret != hipSuccess)
	throw std::runtime_error("alloc failed");
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
	  hip_ret = obuffer_data[i].alloc(obuffer_sizes[i]);
	  if(hip_ret != hipSuccess)
	    throw std::runtime_error("alloc failed");
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }


// Warm up once:
for(int idx = 0; idx < gpu_input.size(); ++idx) {
  hip_ret = hipMemcpy(
    pibuffer[idx], gpu_input[idx].data(), gpu_input[idx].size(), hipMemcpyHostToDevice);
  if(hip_ret != hipSuccess)
    throw std::runtime_error("hipMemcpy failed");

 }



    params.execute(pibuffer.data(), pobuffer.data());

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    hip_ret = hipEventCreate(&start);
if(hip_ret != hipSuccess)
    throw std::runtime_error("hipEventCreate failed");
    hip_ret = hipEventCreate(&stop);
if(hip_ret != hipSuccess)
    throw std::runtime_error("hipEventCreate failed");

for(int itrial = 0; itrial < gpu_time.size(); ++itrial)
	  {
	    // Copy the input data to the GPU:
	    for(int idx = 0; idx < gpu_input.size(); ++idx)
	      {
		hip_ret = hipMemcpy(pibuffer[idx],
			  gpu_input[idx].data(),
			  gpu_input[idx].size(),
			  hipMemcpyHostToDevice);
		if(hip_ret != hipSuccess)
		  throw std::runtime_error("hipMemcpy failed");

	      }

	    hip_ret = hipEventRecord(start);
	    if(hip_ret != hipSuccess)
	      throw std::runtime_error("hipEventRecord failed");

	    params.execute(pibuffer.data(), pobuffer.data());

	    hip_ret = hipEventRecord(stop);
	    if(hip_ret != hipSuccess)
	      throw std::runtime_error("hipEventRecord failed");
	    
	    hip_ret = hipEventSynchronize(stop);
	    if(hip_ret != hipSuccess)
	      throw std::runtime_error("hipEventSynchronize failed");

	    
	    float time;
	    hip_ret = hipEventElapsedTime(&time, start, stop);
	    if(hip_ret != hipSuccess)
	      throw std::runtime_error("hipEventElapsedTime failed");

	    gpu_time[itrial] = time;

	  }

    std::cout << "\nExecution gpu time:";
    for(const auto& i : gpu_time)
    {
        std::cout << " " << i;
    }
    std::cout << " ms" << std::endl;
    
  return 0;
} 

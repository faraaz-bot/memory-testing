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
#include<hip/hip_ext.h>

//#define USE_LDS 0

template<typename T1, typename T2>
T1 ceildiv(T1 a, T2 b)
{
    return (a + b - 1) / b;
}


// Tiled transpose through LDS
template <typename Tval>
__global__ void
__launch_bounds__(1024,1)
transpose(const Tval* __restrict__ idata,
                          Tval* __restrict__ odata,
                          const int Nx,
                          const int Ny,
                          const int tileDim,
                          const int padding)
{
# if USE_LDS

    extern __shared__ __align__(sizeof(Tval)) unsigned char shmem_ptr[];
    Tval* lds = reinterpret_cast<Tval*>(shmem_ptr);

    // Input indices: straight copy.
    const int gipos = (blockIdx.x * blockDim.x + threadIdx.x) * Ny + (blockIdx.y * blockDim.y + threadIdx.y);
    const int lipos = threadIdx.x * (tileDim + padding)            + threadIdx.y;
       
    // Contiguous read
    if((blockIdx.x * blockDim.x + threadIdx.x) < Nx && (blockIdx.y * blockDim.y + threadIdx.y) < Ny)
    {
        lds[lipos] = idata[gipos];
    }

    __syncthreads();

    // Output indices
    const int lopos = threadIdx.y * (tileDim + padding)            + threadIdx.x;
    const int gopos = (blockIdx.y * blockDim.y + threadIdx.x) * Nx + (blockIdx.x * blockDim.x + threadIdx.y);

    // Contiguous write
    if( (blockIdx.y * blockDim.y + threadIdx.x) < Ny && (blockIdx.x * blockDim.x + threadIdx.y) < Nx)
    {
        odata[gopos] = lds[lopos];
    }
#else
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < Nx && iy < Ny) {
        odata[ix * Ny + iy] = idata[iy * Nx + ix];
    }
#endif
    
}


class transpose_params : public fft_params
{
public:
    size_t blockSize = 32;
    int padding = 1;
    transpose_params(){};

    transpose_params(const fft_params& p)
        : fft_params(p){};

    ~transpose_params(){};

    virtual void compute_osize() override
        {
            auto   ol  = olength_cm(); // transposed output
            size_t val = compute_ptrdiff(ol, ostride, nbatch, odist);
            osize.resize(nobuffer());
            for(unsigned int i = 0; i < osize.size(); ++i)
            {
                osize[i] = val + ooffset[i];
            }
        }

    virtual std::vector<size_t> olength() const override
        {
            return length_cm();
        }
    
    virtual std::vector<size_t> obuffer_sizes() const override
        {
            return std::vector<size_t> {::var_size<size_t>(precision, otype) * length[0] * length[1]};
        }
    size_t vram_footprint() override
        {
            return 0;
        }
  
    fft_status create_plan() override
        {
            return fft_status_success;
        }

    dim3 blocks() {
        return dim3(ceildiv(length[0], blockSize), ceildiv(length[1], blockSize));
    }
    
    dim3 threads() {
        return dim3(blockSize, blockSize);
    }
    
    int occupancy() {
        int         max_blocks_per_sm{};
        hipError_t  ret{};

        switch(precision) {
        case fft_precision_single:
            switch(transform_type) {
            case fft_transform_type_complex_forward:
            case fft_transform_type_complex_inverse:
                ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                                   transpose<float2>,
                                                                   threads().x * threads().y * threads().z,         
                                                                   lds_bytes());
                break;
            case fft_transform_type_real_forward:
            case fft_transform_type_real_inverse:
                ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                                   transpose<float>,
                                                                   threads().x * threads().y * threads().z,         
                                                                   lds_bytes());
                break;
            }
            break;
        case fft_precision_double:
            switch(transform_type)  {
            case fft_transform_type_complex_forward:
            case fft_transform_type_complex_inverse:
                ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                                   transpose<double2>,             
                                                                   threads().x * threads().y * threads().z,         
                                                                   lds_bytes());
                break;
            case fft_transform_type_real_forward:
            case fft_transform_type_real_inverse:
                ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                                   transpose<double>,             
                                                                   threads().x * threads().y * threads().z,         
                                                                   lds_bytes());
                break;
            }
            break;
        }
        return max_blocks_per_sm;
    }

    size_t var_size()
        {

        switch(precision) {
        case fft_precision_single:
            switch(transform_type) {
            case fft_transform_type_complex_forward:
            case fft_transform_type_complex_inverse:
                return sizeof(float2);
                break;
            case fft_transform_type_real_forward:
            case fft_transform_type_real_inverse:
                return sizeof(float);
                break;
            }
            break;
        case fft_precision_double:
            switch(transform_type)  {
            case fft_transform_type_complex_forward:
            case fft_transform_type_complex_inverse:
                return sizeof(double2);
                break;
            case fft_transform_type_real_forward:
            case fft_transform_type_real_inverse:
                return sizeof(double);
                break;
            }
            break;
        }
        return 0;
        }
    
    size_t lds_bytes() {
#if USE_LDS
        size_t lds_count = (blockSize + padding) * blockSize;
#else
        size_t lds_count = 0;
#endif
        return lds_count * var_size();
    }
    
    virtual fft_status ext_execute(void** in, void** out, hipEvent_t start, hipEvent_t stop) 
        {
            int ggl_flags = 0;

            switch(precision) {
            case fft_precision_single:
                switch(transform_type) {
                case fft_transform_type_complex_forward:
                case fft_transform_type_complex_inverse:
                    hipExtLaunchKernelGGL(transpose<float2>,
                                          blocks(),
                                          threads(),
                                          lds_bytes(),
                                          0, // stream
                                          start,
                                          stop,
                                          ggl_flags,
                                          (float2*)in[0],
                                          (float2*)out[0],
                                          length[0],
                                          length[1],
                                          blockSize,
                                          padding);
                    break;
                case fft_transform_type_real_forward:
                case fft_transform_type_real_inverse:
                    hipExtLaunchKernelGGL(transpose<float>,
                                          blocks(),
                                          threads(),
                                          lds_bytes(),
                                          0, // stream
                                          start,
                                          stop,
                                          ggl_flags,
                                          (float*)in[0],
                                          (float*)out[0],
                                          length[0],
                                          length[1],
                                          blockSize,
                                          padding);
                    break;
                }
                break;
            case fft_precision_double:
                switch(transform_type)  {
                case fft_transform_type_complex_forward:
                case fft_transform_type_complex_inverse:
                    hipExtLaunchKernelGGL(transpose<double2>,
                                          blocks(),
                                          threads(),
                                          lds_bytes(),
                                          0, // stream
                                          start,
                                          stop,
                                          ggl_flags,
                                          (double2*)in[0],
                                          (double2*)out[0],
                                          length[0],
                                          length[1],
                                          blockSize,
                                          padding);

                    break;
                case fft_transform_type_real_forward:
                case fft_transform_type_real_inverse:
                    hipExtLaunchKernelGGL(transpose<double>,
                                          blocks(),
                                          threads(),
                                          lds_bytes(),
                                          0, // stream
                                          start,
                                          stop,
                                          ggl_flags,
                                          (double*)in[0],
                                          (double*)out[0],
                                          length[0],
                                          length[1],
                                          blockSize,
                                          padding);
                    break;
                }
                break;
            }

            return fft_status_success;
        };

    virtual fft_status execute(void** in, void** out) override
        {
            switch(precision) {
            case fft_precision_single:
                switch(transform_type) {
                case fft_transform_type_complex_forward:
                case fft_transform_type_complex_inverse:
                    hipLaunchKernelGGL(transpose<float2>,
                                       blocks(),
                                       threads(),
                                       lds_bytes(),
                                       0, // stream
                                       (float2*)in[0],
                                       (float2*)out[0],
                                       length[0],
                                       length[1],
                                       blockSize,
                                       padding);
                    break;
                case fft_transform_type_real_forward:
                case fft_transform_type_real_inverse:
                    hipLaunchKernelGGL(transpose<float>,
                                       blocks(),
                                       threads(),
                                       lds_bytes(),
                                       0, // stream
                                       (float*)in[0],
                                       (float*)out[0],
                                       length[0],
                                       length[1],
                                       blockSize,
                                       padding);
                    break;
                }
                break;
            case fft_precision_double:
                switch(transform_type)  {
                case fft_transform_type_complex_forward:
                case fft_transform_type_complex_inverse:
                    hipLaunchKernelGGL(transpose<double2>,
                                       blocks(),
                                       threads(),
                                       lds_bytes(),
                                       0, // stream
                                       (double2*)in[0],
                                       (double2*)out[0],
                                       length[0],
                                       length[1],
                                       blockSize,
                                       padding);

                    break;
                case fft_transform_type_real_forward:
                case fft_transform_type_real_inverse:
                    hipLaunchKernelGGL(transpose<double>,
                                       blocks(),
                                       threads(),
                                       lds_bytes(),
                                       0, // stream
                                       (double*)in[0],
                                       (double*)out[0],
                                       length[0],
                                       length[1],
                                       blockSize,
                                       padding);
                    break;
                }
                break;
            }
            return fft_status_success;
        };
};



template<typename Tval, typename Tvlength>
int Tcheck_result(const void* vin, const void* vout, const Tvlength& length, const int verbose)
{
    const auto pin = (Tval*)vin;
    const auto pout = (Tval*)vout;

    int nbad = 0;
    
    for(int ix = 0; ix <  length[0]; ++ix) {
        for(int iy = 0; iy <  length[1]; ++iy) {
            if(pin[ix * length[1] + iy] != pout[iy * length[0] + ix]) {
                nbad++;
                if(verbose)
                    std::cout << "incorrect result at (" << ix << "," << iy << ")\n";
            }
        }
    }
    return nbad;
}

int check_result(const void* vin, const void* vout, const fft_params& params, const int verbose)
{
    switch(params.precision)
    {
    case fft_precision_single:
    {
                
        switch(params.transform_type) {
        case fft_transform_type_complex_forward:
        case fft_transform_type_complex_inverse:
            return Tcheck_result<std::complex<float>>(vin, vout, params.length, verbose);
            break;
                    
        case fft_transform_type_real_forward:
        case fft_transform_type_real_inverse:
            return Tcheck_result<float>(vin, vout, params.length, verbose);
            break;
        default:
            throw std::runtime_error("invalid transform type");
        }
        break;
    }
    case fft_precision_double:
        switch(params.transform_type) {
        case fft_transform_type_complex_forward:
        case fft_transform_type_complex_inverse:
            return Tcheck_result<std::complex<double>>(vin, vout, params.length, verbose);
            break;
                    
        case fft_transform_type_real_forward:
        case fft_transform_type_real_inverse:
            return Tcheck_result<double>(vin, vout, params.length, verbose);
            break;
        default:
            throw std::runtime_error("invalid transform type");
        }
        break;
    default:
        throw std::runtime_error("invalid precision");                
    }
    return 0;
}
    
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
    transpose_params params;

    // Token string to fully specify fft params.
    std::string token;
    // Declare the supported options.

    bool extLaunch = false;

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("transpose rider command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the transpose library")
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
        ("token", po::value<std::string>(&token))
        ("ext,e", "Not in-place FFT transform (default: in-place)");
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


    if(vm.count("ext"))
    {
        std::cout << "using ext kernel launcher\n";
        extLaunch = true;
    }
            
    std::cout << std::flush;

    // Fixme: set the device id properly after the IDs are synced
    // bewteen hip runtime and rocm-smi.
    // HIP_V_THROW(hipSetDevice(deviceId), "set device failed!");

    params.validate();

    if(params.placement == fft_placement_inplace)
    {
        std::cout << "in-place transpose not implemented; please call with -o\n";
        exit(1);
    }
    
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

    std::cout << "occupancy: " << params.occupancy() << std::endl;

    auto hip_ret = hipSuccess;

    // GPU input and output buffers:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        auto hip_ret = ibuffer[i].alloc(ibuffer_sizes[i]);
        if(hip_ret != hipSuccess)
            throw std::runtime_error("alloc failed");
        pibuffer[i] = ibuffer[i].data();
    }
    
    // Input data:
    compute_input(params, ibuffer);

    if(verbose > 1)
    {
        // Copy input to CPU
        auto cpu_input = allocate_host_buffer(params.precision, params.itype, params.isize);
        for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
        {
            auto hip_ret = hipMemcpy(cpu_input.at(idx).data(),
                                     ibuffer[idx].data(),
                                     ibuffer_sizes[idx],
                                     hipMemcpyDeviceToHost);
            if(hip_ret != hipSuccess)
                throw std::runtime_error("hipMemcpy failed");
            
        }

        std::cout << "GPU input:\n";
        params.print_ibuffer(cpu_input);
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

    hipEvent_t start, stop;
    if(hipEventCreate(&start) != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");
    if(hipEventCreate(&stop) != hipSuccess)
        throw std::runtime_error("hipEventCreate failed");
    
    params.ext_execute(pibuffer.data(), pobuffer.data(), start, stop);
    if(hipEventSynchronize(stop) != hipSuccess)
        throw std::runtime_error("hipEventSynchronize failed");

    auto gpu_input = std::vector<char>(var_size<size_t>(params.precision, params.otype)
                                        * params.length[0] * params.length[1]);
    auto gpu_output = std::vector<char>(var_size<size_t>(params.precision, params.otype)
                                        * params.length[0] * params.length[1]);
    if( hipMemcpy(gpu_input.data(),
                  pibuffer[0],
                  gpu_input.size(),
                  hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
        
    if( hipMemcpy(gpu_output.data(),
                  pobuffer[0],
                  gpu_output.size(),
                  hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
        
    if(verbose > 3)
    {
        std::cout << "GPU output:\n";
        std::vector<std::vector<char>> vgpu_output = {gpu_output};
        params.print_obuffer(vgpu_output);
    }

    const int nbad = check_result(gpu_input.data(), gpu_output.data(), params, verbose);
    if(verbose)
        std::cout << "number of incorrect values: " << nbad << std::endl;
    if(nbad > 0)
        std::cerr << "TRANSPOSE FAILED" << std::endl;
        
        
    // Run the transform several times and record the execution time:1
    std::vector<double> gpu_time(ntrial);

    for(int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
        compute_input(params, ibuffer);
        
        if(extLaunch) {
            params.ext_execute(pibuffer.data(), pobuffer.data(), start, stop);
        } else {
            if(hipEventRecord(start) != hipSuccess)
                throw std::runtime_error("hipEventRecord failed");
            params.execute(pibuffer.data(), pobuffer.data());
            if(hipEventRecord(stop) != hipSuccess)
                throw std::runtime_error("hipEventRecord failed");
        }

        if(hipEventSynchronize(stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");

        float time = -1.0;
        if(hipEventElapsedTime(&time, start, stop) != hipSuccess)
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

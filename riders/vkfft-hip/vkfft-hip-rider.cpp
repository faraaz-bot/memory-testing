#include <iostream>

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include "benchmark_scripts/vkFFT_scripts/include/utils_VkFFT.h"
#include "vkFFT.h"

#include "vkfft_params.h"

#include "gpubuf.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

inline void
    hip_V_Throw(hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != hipSuccess)
    {
        std::stringstream tmp;
        tmp << "HIP_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
}
#define HIP_V_THROW(_status, _message) hip_V_Throw(_status, _message, __LINE__, __FILE__)

int main(int argc, char* argv[])
{
    vkfft_params params;

    int verbose = 3;

    // hip Device number for running tests:
    int deviceId{};

    // Number of performance trial samples
    int ntrial{};

    // Token string to fully specify fft params.
    std::string token;

    po::options_description opdesc("rocfft rider command line options");
    opdesc.add_options()("help,h", "produces this help message")(
        "device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")(
        "verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")(
        "ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")(
        "notInPlace,o", "Not in-place FFT transform (default: in-place)")(
        "double", "Double precision transform (default: single)")(
        "precision",
        po::value<fft_precision>(&params.precision),
        "Transform precision: single (default), double, half")(
        "transformType,t",
        po::value<fft_transform_type>(&params.transform_type)
            ->default_value(fft_transform_type_complex_forward),
        "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
        "forward\n3) real inverse")("batchSize,b",
                                    po::value<size_t>(&params.nbatch)->default_value(1),
                                    "If this value is greater than one, arrays will be used ")(
        "itype",
        po::value<fft_array_type>(&params.itype)->default_value(fft_array_type_unset),
        "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
        "hermitian interleaved\n4) hermitian planar")(
        "otype",
        po::value<fft_array_type>(&params.otype)->default_value(fft_array_type_unset),
        "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
        "hermitian interleaved\n4) hermitian planar")(
        "length", po::value<std::vector<size_t>>(&params.length)->multitoken(), "Lengths.")(
        "istride", po::value<std::vector<size_t>>(&params.istride)->multitoken(), "Input strides.")(
        "ostride",
        po::value<std::vector<size_t>>(&params.ostride)->multitoken(),
        "Output strides.")("idist",
                           po::value<size_t>(&params.idist)->default_value(0),
                           "Logical distance between input batches.")(
        "odist",
        po::value<size_t>(&params.odist)->default_value(0),
        "Logical distance between output batches.")(
        "isize",
        po::value<std::vector<size_t>>(&params.isize)->multitoken(),
        "Logical size of input buffer.")(
        "osize",
        po::value<std::vector<size_t>>(&params.osize)->multitoken(),
        "Logical size of output buffer.")(
        "ioffset", po::value<std::vector<size_t>>(&params.ioffset)->multitoken(), "Input offsets.")(
        "ooffset",
        po::value<std::vector<size_t>>(&params.ooffset)->multitoken(),
        "Output offsets.")("token", po::value<std::string>(&token));

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return 0;
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
            exit(1);
        }
    }
    else
    {
        if(!vm.count("length"))
        {
            std::cout << "Please specify transform length!" << std::endl;
            std::cout << opdesc << std::endl;
            return 0;
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

    params.validate();
    if(!params.valid(verbose))
    {
        throw std::runtime_error("Invalid parameters, add --verbose=1 for detail");
    }

    std::cout << "Token: " << params.token() << std::endl;

    if(hipInit(0) != hipSuccess)
    {
        throw std::runtime_error("hipInit failed");
    }

    if(params.create_plan() != fft_status_success)
    {
        throw std::runtime_error("plan creation failed");
    }

    // GPU input buffer:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        HIP_V_THROW(ibuffer[i].alloc(ibuffer_sizes[i]), "Creating input Buffer failed");
        pibuffer[i] = ibuffer[i].data();
    }

    // Input data:
    params.compute_input(ibuffer);

    if(verbose > 1)
    {
        // Copy input to CPU
        auto cpu_input = allocate_host_buffer(params.precision, params.itype, params.isize);
        for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
        {
            HIP_V_THROW(hipMemcpy(cpu_input.at(idx).data(),
                                  ibuffer[idx].data(),
                                  ibuffer_sizes[idx],
                                  hipMemcpyDeviceToHost),
                        "hipMemcpy failed");
        }

        std::cout << "GPU input:\n";
        params.print_ibuffer(cpu_input);
    }

    // GPU output buffer:
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
            HIP_V_THROW(obuffer_data[i].alloc(obuffer_sizes[i]), "Creating output Buffer failed");
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    if(params.execute(pibuffer.data(), pobuffer.data()) != fft_status_success)
    {
        throw std::runtime_error("FFT plan execution failed!");
    }

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    if(hipEventCreate(&start) != hipSuccess)
    {
        throw std::runtime_error("hipEventCreate failed");
    }
    if(hipEventCreate(&stop) != hipSuccess)
    {
        throw std::runtime_error("hipEventCreate failed");
    }

    for(int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {
        params.compute_input(ibuffer);

        HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

        params.execute(pibuffer.data(), pobuffer.data());

        HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
        HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

        float time;
        if(hipEventElapsedTime(&time, start, stop) != hipSuccess)
        {
            throw std::runtime_error("hipEventElapsedTime failed");
        }
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

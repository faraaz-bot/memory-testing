#include<iostream>

#include "hip_to_cuda.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

__global__ void
set1(int *p)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    p[i] = 1;
}

int
main(int argc, char **argv)
{
    std::cout << "Address sanitizer example\n";
    
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

    
    // Device pointers
    int *dp = nullptr;

    if(hipMalloc(&dp, m * sizeof(int)) != hipSuccess)
    {
        throw std::runtime_error("hipMalloc failed");
    }
    
    hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);


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

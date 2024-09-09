#include<iostream>

#include <hip/hip_runtime.h>


__global__ void
set1(int *p)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    p[i] = 1;
}

int
main(int argc, char **argv)
{
    std::cout << "address sanitizer example\n";
    
    // Number of ints allocated on device:
    int m = 32;//std::atoi(argv[1]);

    // grid dim:
    int n1 = 2; //std::atoi(argv[2]);

    // blocksize: 
    int n2 = 16; //std::atoi(argv[3]);

    // Number of ints allocated on host:
    int c = 32;//std::atoi(argv[4]);

    std::cout << "device size m: " << m << "\n";
    std::cout << "host size c:   " << c << "\n";
    std::cout << "running " << n1 << " blocks of " << n2 << " threads\n";

    
    // Device pointers
    int *dp = nullptr;

    if(hipMalloc(&dp, m*sizeof(int)) != hipSuccess)
    {
        throw std::runtime_error("hipMalloc failed");
    }
    
    hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);
    int *hp = (int*)malloc(c * sizeof(int));
    
    if( hipMemcpy(hp, dp, m*sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    {
        throw std::runtime_error("hipMemcpy failed");
    }
    
    if(hipDeviceSynchronize() != hipSuccess)
    {
        throw std::runtime_error("hipDeviceSynchronize failed");
    }
    if(hipFree(dp) != hipSuccess)
    {
        throw std::runtime_error("hipFree failed");
    }
    free(hp);
    std::puts("Done.");
    return 0;
}

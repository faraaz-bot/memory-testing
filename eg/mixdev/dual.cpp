#include <iostream>

//#include <cuda_runtime.h>
#include <hip/hip_runtime_api.h>


#include <vector>
#include <complex>

#include "dual.hpp"

int main()
{
    std::cout << "y'all\n";

    // NB: Nx and Ny must be at least 32 for cuFFTXt.
    int Nx = 32;
    int Ny = 32;
    std::vector<int> gpus = {0, 0};
    std::cout << "Nx: " << Nx << "\n";
    std::cout << "Ny: " << Ny << "\n";
    std::cout << "GPUs:";
    for(auto gpu : gpus)
        std::cout << " " << gpu;
    std::cout << "\n";
    

    std::vector<std::complex<double>> input(Nx * Ny);

    for(int i = 0; i < input.size(); ++i) {
        input[i] = 1.0 / (1.0 + i);
    }

    std::cout << "hip_out:\n";
    auto hip_out = hip_data(Nx, Ny, gpus, input);
    for(int idx = 0; idx < hip_out.size(); ++idx) {
        for(const auto&val : hip_out[idx]) {
	  //std::cout << " " << val;
        }
        std::cout << "\n";
    }

    std::cout << "cuda_out:\n";
    auto cuda_out = cuda_data(Nx, Ny, gpus, input);
    for(int idx = 0; idx < cuda_out.size(); ++idx) {
        for(const auto&val : cuda_out[idx]) {
	  //std::cout << " " << val;
        }
        std::cout << "\n";
    }
    

    double maxdiff = 0.0;
    for(int idx = 0; idx < gpus.size(); ++idx) {
        std::cout << "buffer " << idx << "\n";
        std::cout << "index\trocfft\t\t\tcufft\t\t\tdifference\n";
        for(int jdx = 0; jdx < hip_out[idx].size(); ++jdx) {
            double diff = std::norm(hip_out[idx][jdx] - cuda_out[idx][jdx]);
            if(diff > maxdiff)
                maxdiff = diff;
            //if(jdx < 16)
                std::cout << jdx << "\t"
                          << hip_out[idx][jdx] << "\t"
                          << cuda_out[idx][jdx] << "\t"
                          << diff << "\n";
        }
    }

    std::cout << "maxdiff over separate buffers: " << maxdiff << "\n";

    double wholemaxdiff = 0.0;
    const int lastbufidx = hip_out.size() - 1;
    for(int jdx = 0; jdx < Nx * Ny; ++ jdx) { 
        double diff = std::norm(hip_out[lastbufidx][jdx] - cuda_out[lastbufidx][jdx]);
        if(diff > wholemaxdiff)
            wholemaxdiff = diff;
    }
    std::cout << "maxdiff over XT-coped buffer: " << wholemaxdiff << "\n";
    
    return 0;
}

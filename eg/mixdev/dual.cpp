#include <iostream>

//#include <cuda_runtime.h>
#include <hip/hip_runtime_api.h>


#include <vector>
#include <complex>

#include "dual.hpp"

int main()
{
    std::cout << "y'all\n";

    int Nx = 32;
    int Ny = 32;
    
    std::vector<std::complex<double>> input(Nx * Ny);

    for(int i = 0; i < input.size(); ++i) {
        input[i] = 1.0 / (1.0 + i);
    }

    auto hip_out = hip_data(Nx, Ny, input);
    std::cout << "hip_out:";
    for(int idx = 0; idx < hip_out.size(); ++idx) {
        for(const auto&val : hip_out[idx]) {
            std::cout << " " << val;
        }
        std::cout << "\n";
    }
    
    auto cuda_out = cuda_data(Nx, Ny, input);
    std::cout << "cuda_out:";
    for(int idx = 0; idx < cuda_out.size(); ++idx) {
        for(const auto&val : cuda_out[idx]) {
            std::cout << " " << val;
        }
        std::cout << "\n";
    }
    

    double maxdiff = 0.0;
    for(int idx = 0; idx < hip_out.size(); ++idx) {
        std::cout << "buffer " << idx << "\n";
        for(int jdx = 0; jdx < hip_out[idx].size(); ++jdx) {
            double diff = std::norm(hip_out[idx][jdx] - cuda_out[idx][jdx]);
            if(diff > maxdiff)
                maxdiff = diff;
            std::cout << jdx << "\t"
                      << hip_out[idx][jdx] << "\t"
                      << cuda_out[idx][jdx] << "\t"
                      << diff << "\n";
        }
    }

    std::cout << "maxdiff: " << maxdiff << "\n";
    
    return 0;
}

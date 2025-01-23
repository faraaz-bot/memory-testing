#include <stdio.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    int mpi_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int mpi_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    size_t N = 16;

    const size_t localsize = (N / mpi_size) + (mpi_rank < (N % mpi_size) ? 1 : 0);
    const size_t localstart = mpi_rank * (N / mpi_size)
        + std::min(mpi_rank, static_cast<decltype(mpi_rank)>(N % mpi_size));
    
    for(int irank = 0; irank < mpi_size; ++irank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(irank == mpi_rank) {
            std::cout << "rank " << mpi_rank 
                      << " local size: " << localsize
                      << " local start: " << localstart
                      << "\n" << std::flush;
        }
    }

    std::vector<float> localdata(localsize);
    for(size_t idx = 0; idx < localsize; ++idx) {
        //localdata[idx] = idx + localstart;
        localdata[idx] = idx + 10 * mpi_rank;;
    }
    
    for(int irank = 0; irank < mpi_size; ++irank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(irank == mpi_rank) {
            std::cout << "rank " << mpi_rank 
                      << " input:";
            for(const auto val : localdata)
                std::cout << " " << val;
            std::cout << "\n" << std::flush;
                    
        }
    }

    const size_t localbytes = sizeof(float) * localsize;
    
    float* devdata = nullptr;
    if(hipMalloc(&devdata, localbytes ) != hipSuccess)
    {
        throw std::runtime_error("hipMalloc failed");
    }
    
    if(hipMemcpy(devdata, localdata.data(), localbytes, hipMemcpyHostToDevice) != hipSuccess)
    {
        throw std::runtime_error("hipMemcpy failed");
    }

    MPI_Request req_ata;
    
    const int sendsize = localsize / mpi_size;
    MPI_Ialltoall(devdata, sendsize, MPI_FLOAT,
                 devdata, sendsize, MPI_FLOAT,
                  MPI_COMM_WORLD, &req_ata);
    
    // Verify the result:
    std::vector<float> localoutput(localsize);
    if(hipMemcpy(localoutput.data(), devdata, localbytes, hipMemcpyDeviceToHost) != hipSuccess)
    {
        throw std::runtime_error("hipMemcpy failed");
    }
    for(int irank = 0; irank < mpi_size; ++irank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(irank == mpi_rank) {
            std::cout << "rank " << mpi_rank 
                      << " output:";
            for(const auto val : localoutput)
                std::cout << " " << val;
            std::cout << "\n" << std::flush;
                    
        }
    }

    
        
    if(hipFree(devdata) != hipSuccess)
    {
        throw std::runtime_error("hipFree failed");
    }

        
    MPI_Finalize();
    
    return 0;
}

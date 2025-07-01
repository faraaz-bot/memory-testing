#include <stdio.h>
#include "hip_to_cuda.h"
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
    
    float* devin = nullptr;
    if(hipMalloc(&devin, localbytes ) != hipSuccess)
    {
        throw std::runtime_error("hipMalloc failed");
    }
    if(hipMemcpy(devin, localdata.data(), localbytes, hipMemcpyHostToDevice) != hipSuccess)
    {
        throw std::runtime_error("hipMemcpy failed");
    }

    float* devout = nullptr;
    if(hipMalloc(&devout, localbytes ) != hipSuccess)
    {
        throw std::runtime_error("hipMalloc failed");
    }
    
    MPI_Request req_ata;
    MPI_Status status_ata;
    
    const int sendsize = localsize / mpi_size;

    std::vector<int> sendcounts(mpi_size);
    std::fill(sendcounts.begin(), sendcounts.end(), sendsize);
    std::vector<int> sendoffset(mpi_size);
    sendoffset[0] = 0;
    for(int idx = 1; idx < sendoffset.size(); ++idx) {
        sendoffset[idx] = sendoffset[idx - 1] + sendcounts[idx - 1];
    }
    
    
    std::vector<int> recvcounts(mpi_size);
    std::fill(recvcounts.begin(), recvcounts.end(), sendsize);
    std::vector<int> recvoffset(mpi_size);
    recvoffset[0] = 0;
    for(int idx = 1; idx < recvoffset.size(); ++idx) {
        recvoffset[idx] = recvoffset[idx - 1] + recvcounts[idx - 1];
    }

    for(int irank = 0; irank < mpi_size; ++irank) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(irank == mpi_rank) {
            std::cout << "rank " << mpi_rank 
                      << "\n\tsendcount:";
            for(const auto val : sendcounts)
                std::cout << " " << val;
            std::cout << "\n\tsendoffset:";
            for(const auto val : sendoffset)
                std::cout << " " << val;
            std::cout << "\n\trecvcount:";
            for(const auto val : recvcounts)
                std::cout << " " << val;
            std::cout << "\n\trecvoffset:";
            for(const auto val : recvoffset)
                std::cout << " " << val;
            std::cout << "\n" << std::flush;
        }
    }

    MPI_Ialltoallv(devin, sendcounts.data(), sendoffset.data(), MPI_FLOAT,
                   devout, recvcounts.data(), recvoffset.data(),  MPI_FLOAT,
                   MPI_COMM_WORLD, &req_ata);
    std::vector<MPI_Request> vreq;
    std::vector<MPI_Status> vstatus;

    if(false) {
        vreq.push_back(req_ata);
        vstatus.push_back(status_ata);
        if(MPI_Waitall(vreq.size(), vreq.data(), vstatus.data()) != MPI_SUCCESS)
        {
            throw std::runtime_error("waitall failed");
        }
    } else {
        MPI_Wait(&req_ata, MPI_STATUS_IGNORE);
    }
    
    
    // Verify the result:
    std::vector<float> localoutput(localsize);
    if(hipMemcpy(localoutput.data(), devout, localbytes, hipMemcpyDeviceToHost) != hipSuccess)
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
            
    if(hipFree(devout) != hipSuccess)
    {
        throw std::runtime_error("hipFree failed");
    }

    if(hipFree(devin) != hipSuccess)
    {
        throw std::runtime_error("hipFree failed");
    }

        
    MPI_Finalize();
    
    return 0;
}

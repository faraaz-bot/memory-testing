// OMPI_CC=hipcc mpicc -o ./gpu-aware ./gpu-aware.cpp

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <vector>
#include <iostream>


__global__ void multby2(int * data, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) 
        data[idx] *= 2;
}

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

int main(int argc, char **argv) {
    MPI_Status status;
    MPI_Request request;

    auto hipret = hipSuccess;
    
    hipStream_t stream;
    hipret = hipStreamCreate (&stream);
    if(hipret != hipSuccess)
        throw std::runtime_error("hipStreamCreate failed");

    hipEvent_t mpi_event, h2d_event;

    
    
    hipret = hipEventCreateWithFlags(&mpi_event, hipEventDisableTiming);
    if(hipret != hipSuccess)
        throw std::runtime_error("hipEventCreateWithFlags failed");
    hipret = hipEventCreateWithFlags(&h2d_event, hipEventDisableTiming);
    if(hipret != hipSuccess)
        throw std::runtime_error("hipEventCreateWithFlags failed");
    
    //hipEventRecord(mpi_event, stream);
    
    const int N = 100;

    MPI_Init(&argc,&argv);
    int mpi_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int mpi_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    //allocate buffers
    std::vector<int> h_buf(N);

    const size_t buf_size = h_buf.size() * sizeof(decltype(h_buf)::value_type);
    int *d_buf = nullptr;
    if(mpi_rank == 0 || mpi_rank == 1) {
        hipret = hipMalloc(&d_buf, buf_size);
        if(hipret != hipSuccess)
            throw std::runtime_error("hipMalloc failed");
    }
    
    //initialize buffers

    switch(mpi_rank) {
    case 0:
        for(int i = 0; i < N; i++)
            h_buf[i] = i;
        break;
    case 1:
        for(int i = 0; i < N; i++)
            h_buf[i] = -1;
        break;
    default:
        break;
    }

    if(mpi_rank == 0 || mpi_rank == 1) {
        hipret = hipMemcpyAsync(d_buf, h_buf.data(), buf_size, hipMemcpyHostToDevice, stream);
        if(hipret != hipSuccess)
            throw std::runtime_error("hipMemcpyAsync failed");
        
    }
    
    hipret = hipEventRecord(h2d_event, stream);
    if(hipret != hipSuccess)
        throw std::runtime_error("hipEventRecord failed");
    
    hipret = hipEventSynchronize(h2d_event);
    if(hipret != hipSuccess)
        throw std::runtime_error("hipEventSynchronize failed");

    
    //communication
    switch(mpi_rank) {
    case 0:
        MPI_Isend(d_buf, N, MPI_INT, 1, 123, MPI_COMM_WORLD, &request);
        break;
    case 1:
        MPI_Irecv(d_buf, N, MPI_INT, 0, 123, MPI_COMM_WORLD, &request);
        break;
    default:
        break;
    }

    //validate results
    if(mpi_rank == 1) {

        std::vector<MPI_Status> vstatus;
        std::vector<MPI_Request> vrequest;

        vstatus.push_back(status);
        vrequest.push_back(request);
    
        MPI_Waitall(1, vrequest.data(), vstatus.data());
        
        int blockSize = 512;
        const int gridSize    = ceildiv(N, blockSize);
        multby2<<<dim3(gridSize), dim3(blockSize), 0, stream>>>(d_buf, N);
        
        hipret = hipMemcpyAsync(h_buf.data(), d_buf, buf_size, hipMemcpyDeviceToHost, stream);
        if(hipret != hipSuccess)
            throw std::runtime_error("hipMemcpyAsync failed");
        int nerror = 0;
        for(int i = 0; i < N; ++i) {
            if(h_buf[i] != 2 * i) {
                printf("Error: buffer[%d]=%d but expected %d\n", i, h_buf[i], 2 * i);
                ++nerror;
            }
        }
        fflush(stdout);
        if(nerror == 0)
            printf("all good!\n");
        else
            std::cout << "things didn't work!\n";
    }

    //free buffers
    if(mpi_rank == 0 || mpi_rank == 1) {
        hipret = hipFree(d_buf);
        if(hipret != hipSuccess)
            throw std::runtime_error("hipFree failed");
    }
    hipret = hipStreamSynchronize(stream);
    if(hipret != hipSuccess)
        throw std::runtime_error("hipStreamSynchronize failed");
    hipret = hipStreamDestroy(stream);
    if(hipret != hipSuccess)
        throw std::runtime_error("hipStreamDestroy failed");
        
    MPI_Finalize();
}

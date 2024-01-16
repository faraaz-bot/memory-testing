// OMPI_CC=hipcc mpicc -o ./gpu-aware ./gpu-aware.cpp

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <vector>


int main(int argc, char **argv) {

    int *d_buf = nullptr;
    MPI_Status status;

    const int N = 100;

    MPI_Init(&argc,&argv);
    int mpi_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int mpi_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    //allocate buffers
    std::vector<int> h_buf(N);

    const size_t buf_size = h_buf.size() * sizeof(decltype(h_buf)::value_type);
    hipMalloc(&d_buf, buf_size);
    
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
        hipMemcpy(d_buf, h_buf.data(), buf_size, hipMemcpyHostToDevice);
    }
        
    //communication
    switch(mpi_rank) {
    case 0:
        MPI_Send(d_buf, N, MPI_INT, 1, 123, MPI_COMM_WORLD);
        break;
    case 1:
        MPI_Recv(d_buf, N, MPI_INT, 0, 123, MPI_COMM_WORLD, &status);
        break;
    default:
        break;
    }

    //validate results
    if(mpi_rank == 1) {
        hipMemcpy(h_buf.data(), d_buf, buf_size, hipMemcpyDeviceToHost);
        for(int i = 0; i < N; ++i) {
            // if(h_buf[i] != i)
            //     printf("Error: buffer[%d]=%d but expected %dn", i, h_buf[i], i);
        }
        //fflush(stdout);
        //printf("all good!\n");
    }

    //free buffers
    hipFree(d_buf);
	
    MPI_Finalize();
}

#include <iostream>
#include <mpi.h>
#include <vector>


int main(int argc, char **argv) {
    MPI_Status status;
    MPI_Request request;

    const int N = 100;

    MPI_Init(&argc,&argv);
    int mpi_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int mpi_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    //allocate buffers
    std::vector<int> h_buf(N);

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
   
    //communication
    switch(mpi_rank) {
    case 0:
        MPI_Isend(h_buf.data(), N, MPI_INT, 1, 123, MPI_COMM_WORLD, &request);
        break;
    case 1:
        MPI_Irecv(h_buf.data(), N, MPI_INT, 0, 123, MPI_COMM_WORLD, &request);
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
        
        int nerror = 0;
        for(int i = 0; i < N; ++i) {
            if(h_buf[i] != i) {
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
        
    MPI_Finalize();
}

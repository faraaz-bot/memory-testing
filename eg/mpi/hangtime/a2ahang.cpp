#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    std::cout << "asdf\n";

    // Set up a window on each rank, initialized to zero:
    MPI_Win window;
    int window_buffer = 0;
    MPI_Win_create(&window_buffer,
                   sizeof(int),
                   sizeof(int),
                   MPI_INFO_NULL,
                   MPI_COMM_WORLD,
                   &window);
    std::cout << "rank " << mpi_rank << " window: " << window_buffer << "\n";

    // We need this to happen in plan creation:
    MPI_Win_fence(0, window);

    
    
    int N = 8;
    std::vector<int> buf(N);
    switch(mpi_rank) {
    case 0:
        for(int i = 0; i < N; i++)
            buf[i] = i;
        break;
    case 1:
        for(int i = 0; i < N; i++)
            buf[i] = -1;
        break;
    default:
        break;
    }

    // Communication
    MPI_Status status;
    MPI_Request request;

    switch(mpi_rank) {
    case 0:
    {
        int fail = 1;
        if(!fail)
        {
            // All good on rank-0, send the data!
            MPI_Isend(buf.data(), N, MPI_INT, 1, 123, MPI_COMM_WORLD, &request);
        }
        else
        {
            // Well, rank-0 failed, so send an error status:
            int errcode = 1;
            auto ret = MPI_Put(&errcode, // origin address
                               1,        // origin count
                               MPI_INT,  // origin data type
                               1,        // target rank
                               0,        // target displacement
                               1,        // target count
                               MPI_INT,  // target data type
                               window);  // window
        }
        break;
    }
    case 1:
    {
        MPI_Irecv(buf.data(), N, MPI_INT, 0, 123, MPI_COMM_WORLD, &request);
        break;
    }
    default:
        break;
    }

    if(mpi_rank == 1) {
        int ready = 0; // not ready
        while(!ready) {
            MPI_Test(&request, &ready, MPI_STATUS_IGNORE);
            //std::cout << window_buffer << "\n";
            if(window_buffer != 0)
                break;
        }

        if(window_buffer == 0) 
        {
            // no error code
            for(const auto val : buf)
                std::cout << val << " ";
            std::cout << "\n";
        }
        else
        {
            std::cout << "rank 0 told rank 1 to give up\n";
            if(MPI_Cancel(&request) != MPI_SUCCESS)
                std::cout << "Cancel failed for some reason...\n";
        }
    }
    
    
    MPI_Finalize();
    return 0;
}

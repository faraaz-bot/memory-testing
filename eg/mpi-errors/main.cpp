#include "../argv/CLI11.hpp"
#include "helper.hpp"
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <numeric>

// Program to test the error handling of MPI implementations
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    // MPI_Comm_set_errhandler(comm, MPI_ERRORS_ARE_FATAL);
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    int mpi_rank  = -1;
    int num_ranks = -1;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &num_ranks);

    // Setup input
    const size_t elems_per_rank = 16;
    const size_t send_size      = elems_per_rank / num_ranks;
    gpubuf       d_input(elems_per_rank);
    gpubuf       d_out(elems_per_rank);

    std::vector<float> h_input(num_ranks * elems_per_rank);
    for(int i = 0; i < num_ranks * elems_per_rank; i++)
        h_input[i] = i;

    HIP_CHECK(hipMemcpy(d_input.data(),
                        h_input.data() + mpi_rank * elems_per_rank,
                        sizeof(float) * elems_per_rank,
                        hipMemcpyHostToDevice));

    // Print before
    // print1d<<<1, 1>>>(d_input.data(), elems_per_rank, mpi_rank);

    // Run some collective calls, check error per rank. Try different ways of triggering errors...
    // int ret = MPI_Alltoall(
    //     d_input.data(), send_size, MPI_FLOAT, d_out.data(), send_size, MPI_FLOAT, comm);
    int ret = -1;
    if(mpi_rank == 0)
        ret = MPI_Alltoall(nullptr, send_size, MPI_FLOAT, d_out.data(), send_size, MPI_FLOAT, comm);
    else
        ret = MPI_Alltoall(
            d_input.data(), send_size, MPI_FLOAT, d_out.data(), send_size, MPI_FLOAT, comm);

    // Print after
    // print1d<<<1, 1>>>(d_out.data(), elems_per_rank, mpi_rank);
    std::cout << "Return: MPI Rank " << mpi_rank << " has return code = " << ret << std::endl;
}

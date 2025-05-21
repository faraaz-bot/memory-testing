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

    int         ret = -1;
    MPI_Status  mpi_status;
    MPI_Request mpi_req;

    std::vector<MPI_Status>  vstatus;
    std::vector<MPI_Request> vreq;

    // Pass in a host side buffer. Not sure if MPI will do some device <-> host magic here though?
    if(mpi_rank == 0)
    {
        std::vector<float> dummy_host_buf(N);
        ret = MPI_Ialltoall(dummy_host_buf.data(),
                            send_size,
                            MPI_FLOAT,
                            d_out.data(),
                            send_size,
                            MPI_FLOAT,
                            comm,
                            &mpi_req);
    }
    else
        ret = MPI_Ialltoall(d_input.data(),
                            send_size,
                            MPI_FLOAT,
                            d_out.data(),
                            send_size,
                            MPI_FLOAT,
                            comm,
                            &mpi_req);

    if(ret != MPI_SUCCESS)
    {
        char errmsg[MPI_MAX_ERROR_STRING];
        int  errlen = -1;
        MPI_Error_string(ret, errmsg, &errlen);
        std::cout << "Return: MPI Rank " << mpi_rank << " has return = " << errmsg << std::endl;
    }

    vstatus.push_back(mpi_status);
    vreq.push_back(mpi_req);

    int ret2 = MPI_Waitall(vreq.size(), vreq.data(), vstatus.data());

    if(ret2 != MPI_SUCCESS)
    {
        char errmsg2[MPI_MAX_ERROR_STRING];
        int  errlen2 = -1;
        MPI_Error_string(ret2, errmsg2, &errlen2);
        std::cout << "MPI_Waitall returned " << errmsg2 << std::endl;
    }

    MPI_Finalize();
}

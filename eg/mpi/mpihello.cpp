#include <mpi.h>
#include <iostream>

int main(int argc, char** argv)
{
  int mpi_rank, mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if(mpi_rank == 0)
    std::cout << "mpi_size: " << mpi_size << "\n";
  MPI_Finalize();

  return 0;
}

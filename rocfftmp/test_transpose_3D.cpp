/*
* Example test rocfft_mp
* Computes parallel 3-D FFTs using 1-D (slabs) decomposition.
* See online documentation for details.
* Compilation (replace appropriately where needed):
    mpic++ test_transpose_3D.cpp -o test_transpose_3D 
// Execution:    
    mpirun -np <N_procs> ./test_transpose_3D <Nx> <Ny> <Nz>
    mpirun -np 2 ./test_transpose_3D 4 4 4
    mpirun -np 16 ./test_transpose_3D 128 256 192
*/

#include "transpose.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int nprocs, my_rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &my_rank);

    // Array size
    std::vector<size_t> N = {std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3])};

    // Splitting input amongst MPI processes and return local sizes
    // ROCFFT_SLABS_SPLIT_Z means slabs are obtaining partitioning along the Z-axis
    int axis_split_dimension = ROCFFT_SLABS_SPLIT_Z;

    auto   local_dims    = geometry_splitting(N, axis_split_dimension, comm);
    size_t local_fftsize = local_dims[0] * local_dims[1] * local_dims[2];

    // Input data on host
    std::vector<std::complex<double>> input(local_fftsize);
    std::vector<std::complex<double>> output(local_fftsize);

    // Random initialization
    std::minstd_rand                       park_miller(1234);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for(auto& e : input)
        e = static_cast<double>(unif(park_miller));

    // Print input
    if(my_rank == 0)
        rocfft_mp_print_array("Input data", my_rank, input, N[0]);

    // Processor grid shapes before and after global transposition

    MPI_Datatype  my_type          = MPI_DOUBLE_COMPLEX;
    MPI_Datatype* subarrays_input  = new MPI_Datatype[nprocs];
    MPI_Datatype* subarrays_output = new MPI_Datatype[nprocs];

    // Plan transpose from in_axis to out_axis
    int in_axis  = 2;
    int out_axis = 0;

    plan_transpose(N, in_axis, out_axis, my_type, subarrays_input, subarrays_output, comm);

    // 3-D transposition using advanced all-to-all
    std::vector<int> counts(nprocs, 1);
    std::vector<int> displacements(nprocs, 0);

    MPI_Alltoallw(input.data(),
                  counts.data(),
                  displacements.data(),
                  subarrays_input,
                  output.data(),
                  counts.data(),
                  displacements.data(),
                  subarrays_output,
                  comm);

    // // Print output
    if(my_rank == 0)
        rocfft_mp_print_array("output data", my_rank, output, N[0]);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
/*
* Example test rocfft_mp
* Computes parallel 3-D FFTs using 1-D (slabs) decomposition.
* See online documentation for details.
* Compilation (replace appropriately where needed):
    hipcc test_rocfft_mp_3D.cpp -lfftw3 -I/home/ayala/gits/rocFFT-internal/build/rocfft/include \
     -L/home/ayala/gits/rocFFT-internal/build/library/src/ -lrocfft -I/usr/lib/x86_64-linux-gnu/openmpi/include \ 
     -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -o test_rocfft_mp_3D 
// Execution:    
    mpirun -np <N_procs> ./test_rocfft_mp_3D <Nx> <Ny> <Nz>
    mpirun -np 16 ./test_rocfft_mp_3D 128 256 192
*/

#include "rocfft_mp.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int nprocs, my_rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &my_rank);

    // FFT input size 
    std::vector<size_t> N = { std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3])};

    // Splitting input amongst MPI processes and return local sizes
    // ROCFFT_SLABS_SPLIT_Z means slabs are obtaining partitioning along the Z-axis
    auto local_dims = geometry_splitting(N, ROCFFT_SLABS_SPLIT_Z, comm);

    size_t local_fftsize = local_dims[0]*local_dims[1]*local_dims[2];

    // Input data on host
    std::vector<std::complex<double>> input(local_fftsize);
    std::vector<std::complex<double>> input_copy(local_fftsize);
    std::vector<std::complex<double>> output(local_fftsize);

    // Random initialization
    std::minstd_rand park_miller(1234);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for(auto &e : input)
        e = static_cast<double>(unif(park_miller));

    std::copy(input.begin(), input.end(), input_copy.begin());

    // Print input
    // rocfft_mp_print_array("Input data", my_rank, input, N[0]);

    // Processor grid shapes before and after global transposition
    std::vector<int> in_transpose_shape;
    std::vector<int> out_transpose_shape;

    rocfft_mp_plan_slabs options(nprocs);

    MPI_Datatype my_type = MPI_DOUBLE_COMPLEX;
    int axis_split_dimension = ROCFFT_SLABS_SPLIT_Z;

    // Create rocfft multi-process 3-D plan
    auto plan_xyz = rocfft_mp_plan_3D(N, local_dims, axis_split_dimension, my_type,
                                      in_transpose_shape, out_transpose_shape, comm, options);

    // Execute rocfft multi-process 
    rocfft_mp_execute(plan_xyz, input, output, options);

    // Destroy rocfft MPI plan
    rocfft_mpi_plan_destroy(plan_xyz);
    rocfft_cleanup();

    // ---------------------------------------------------------------
    // Code from here is only for error validation using FFTW
    // Given our input, X, we take the result from rocfft_mp, FFT(X)
    // and apply a backward FFT using FFTW, i.e. we find IFFT(FFT(X))
    // The error is calculated as e = || X - IFFT(FFT(X)) ||
    // ---------------------------------------------------------------

    int root_rank = 0;
    if(my_rank == root_rank)
    {
        std::cout << "-----------------" << std::endl;
        std::cout << "3-D rocfftmp test" << std::endl;
        std::cout << "-----------------" << std::endl;
        
        
        std::cout << "FFT size  : " << N[0] << "x" << N[1] << "x" << N[2]<< std::endl; 

        std::vector<int> proc_grid = {1,1,1};
        std::vector<int> proc_grid_intermediate = {1,1,1};
        proc_grid[options.axis] = nprocs;
        proc_grid_intermediate[options.dim_fast] = nprocs;

        std::cout << "Proc grids: " << proc_grid[0] << "x" << proc_grid[1] << "x" << proc_grid[2] << "  ";
        proc_grid[options.axis] = 1;
        proc_grid[options.dim_fast] = nprocs;
        std:: cout << proc_grid_intermediate[0] << "x" << proc_grid_intermediate[1] << "x" << proc_grid_intermediate[2] << "  "; 
        std:: cout << proc_grid[0] << "x" << proc_grid[1] << "x" << proc_grid[2] << std::endl; 


        size_t global_size = N[0]*N[1]*N[2];
        std::vector<std::complex<double>> global_input(global_size);
        std::vector<std::complex<double>> rocfft_output(global_size);
        std::vector<std::complex<double>> rocfft_output_2(global_size);

        std::vector<int> counts(nprocs);
        
        MPI_Gather(&local_fftsize, 1, MPI_INT, counts.data(), 1, MPI_INT, root_rank, comm);

        std::vector<int> displacements = {0};
        for(unsigned int i = 1; i < nprocs; ++i)
            displacements.push_back(counts[i - 1] + displacements[i - 1]);

        // Gathering input parallel data into a master process, X = global_input.data()
        MPI_Gatherv(input_copy.data(), local_fftsize, MPI_DOUBLE_COMPLEX, global_input.data(), counts.data(), displacements.data(), MPI_DOUBLE_COMPLEX, root_rank, comm);

        // Gathering parallel output of 3-D rocfftmp calculation into a master process, FFT(X) = input.data()
        MPI_Gatherv(input.data(), local_fftsize, MPI_DOUBLE_COMPLEX, rocfft_output.data(), counts.data(), displacements.data(), MPI_DOUBLE_COMPLEX, root_rank, comm);

        // rocfft_mp_print_array("X: GLOBAL INPUT", my_rank, global_input, N[0]);

        // Inverse transform of rocfft output, i.e. we calculate D = IFFT(FFT(X))
        fftw_plan plan_validate;
        plan_validate = fftw_plan_dft_3d(N[2], N[1], N[0], (fftw_complex *) rocfft_output.data(), (fftw_complex *) rocfft_output.data(),
                                        FFTW_BACKWARD, FFTW_ESTIMATE);

        fftw_execute(plan_validate);
        // rocfft_mp_print_array("IFFT_{fftw}(FFT_{rocfft}(X)) ", my_rank, rocfft_output, N[0]);

        // We calculate the error:
        double err = 0.0;
        for(size_t i=0; i<global_size; i++)
        {
            err = std::max(err, std::abs( (1.0/global_size)*rocfft_output[i] - global_input[i]));
        }

        std::cout << "|| X - IFFT_{fftw}(FFT_{rocfft}(X)) ||_{max} = " << err << std::endl;

        fftw_destroy_plan(plan_validate); 
    }

    else
    {
        MPI_Gather(&local_fftsize, 1, MPI_INT, NULL, 0, MPI_INT, root_rank, comm);
        MPI_Gatherv(input_copy.data(), local_fftsize, MPI_DOUBLE_COMPLEX, NULL, NULL, NULL, MPI_DOUBLE_COMPLEX, root_rank, comm);
        MPI_Gatherv(input.data(), local_fftsize, MPI_DOUBLE_COMPLEX, NULL, NULL, NULL, MPI_DOUBLE_COMPLEX, root_rank, comm);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
while IFS= read -r token; do
    echo "mpirun --np 2 rocfft_mpi_worker $token --benchmark"
done < np_2.txt
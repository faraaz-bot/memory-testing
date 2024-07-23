#!/bin/bash

usage() { echo "Usage: $0 [-p <path_to_staging>] [-n <number of MPI procs>] [-l <length>]" 1>&2; exit 1; }

while getopts ":p:n:l:" o; do
    case "${o}" in
        p)
            p=${OPTARG}
            ;;    
        n)
            n=${OPTARG}
            ;;
        l)
            l=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

# Get MPI tokens from rocfft-test
# echo ./${p}/rocfft-test --gtest_filter=multi_gpu* --gtest_list_tests --mp_lib mpi --mp_ranks $np --mp_launch \"/usr/bin/mpirun --np ${n} ${p}/rocfft_mpi_worker\" > mpi_tokens

./${p}/rocfft-test --gtest_filter=multi_gpu* --gtest_list_tests --mp_lib mpi --mp_ranks $np --mp_launch \"/usr/bin/mpirun --np ${n} ${p}/rocfft_mpi_worker\" > mpi_tokens


# Filtering
grep fftw mpi_tokens | awk {'print $1'} | cut -d/ -f2- > tokens
# grep fftw test_mpi.txt | awk {'print $1'} | cut -d/ -f2- > tokens

# Run benchmark using rocfft_mpi_worker
while IFS= read -r token; do
    echo "mpirun --np 2 rocfft_mpi_worker $token --benchmark"
    # mpirun --np 2 rocfft_mpi_worker $token --benchmark
done < tokens

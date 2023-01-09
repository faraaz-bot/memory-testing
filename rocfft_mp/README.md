# rocFFT-mp

rocFFT-mp is a prototype implementation to compute 3-D FFTs using
either [rocFFT] or [FFTW3] as backends and advanced MPI for tensor transposition.

[rocFFT]: https://github.com/ROCmSoftwarePlatform/rocFFT
[FFTW3]: https://www.fftw.org


## Dependencies
rocFFT-mp requires [rocFFT] or [FFTW3] libraries to be installed in the system, and an MPI distribution such as [OpenMPI] or [MVAPICH].

[OpenMPI]: https://www.open-mpi.org
[MVAPICH]: https://mvapich.cse.ohio-state.edu


## Building from source

### Library build dependencies

To build the rocFFT-mp library:
* rocFFT-mp depends on [rocFFT] on AMD platforms;
* rocFFT-mp depends on [FFTW3] on other platforms.

## Building from source

The initial release of rocFFT-mp is provided as a header file
[rocfft_mp.h].

[rocfft_mp.h]: https://github.com/ROCmSoftwarePlatform/rocFFT-misc/blob/master/rocfft_mp/rocfft_mp.h


Tests are compiled using hipcc:

```
hipcc test_rocfft_mp_3D.cpp -I<PATH_TO_ROCFFT>/include
-L <PATH_TO_ROCFFT>/library -lrocfft -I<PATH_TO_FFTW3>/include
-L <PATH_TO_FFTW3>/library -lfftw3   -I<PATH_TO_MPI>/include
-L <PATH_TO_MPI>/library -lmpi -o test_rocfft_mp_3D
```

## Current Features

The following functionality is currently available for rocFFT-mp:
1. 3-D Complex-to-Complex FFT computation using slabs decomposition.
2. Precision: single and double.
3. Tensor transposition can be used as an independent kernel.

Current tests:

| Test          | Dependencies                  | Description                            |
|-----------------|-------------------------------|------------------------------------------|
| 3-D C2C FFT   | `test_rocfft_mp_3D.cpp`    | Parallel FFT via slab decomposition

## Contribution Rules

### Source code formatting

* C++ source code must be formatted with clang-format file herein.
* Python source code must be formatted with yapf --style pep8.

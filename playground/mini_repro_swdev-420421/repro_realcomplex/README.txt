Build with:

  hipcc -fopenmp rocfft_kernel_harness_0.cpp -o rocfft_kernel_harness_0

And run the resulting program with no arguments.

This program does 100 trials of 3 kernels each.  Each trial after the
first compares the results with the previous trial and aborts if the
result looks different.  The trials are all supposed to have the same
results because they have the same inputs and run the same kernels.

Kernel code is all in the fft_*.h header files.  The main file
(rocfft_kernel_harness_0.cpp) reads those files from the current
directory and uses hipRTC to build them into executable code.  rocFFT
would also build and launch kernels this way.

Note that this test program has different inputs from rocfft-test and
produces different outputs.

# Performance

## rocFFT vs cuFFT performance

### Overview

This tool:
- builds rocFFT (when on AMD platforms)
- builds hipFFT
- builds the Python hipFFT wrapper
- runs a bunch of FFTs and records their GPU execution time

### Building

The builds are done locally and installed into the 'build' directory.
To build:

    perf.py build --rocfft <commit> --hipfft <commit> [--cuda]

This uses git to clone the (public) rocFFT and hipFFT and does the
builds using appropriate flags.  Special care is taken to ensure
library paths are set correctly.

When building on a CUDA platform:

    export CUDA_PATH=/usr/local/cuda
    export PATH=$CUDA_PATH/bin:$PATH
    perf build --hipfft <commit> --cuda

### Running performance tests

To run the tests:

    perf.py run [--ntrials 10] [--verify] [--suite all]

Results are stored in a bunch of DAT files.  You would do the build
and run steps on both ROCM and CUDA platforms.

### Plotting the results

This step is a bit clunky...

1. Copy the ROCM results (CSV files) and put them into the `rocfft` directory.
2. Copy the CUDA results and put them into the `cufft` directory.
3. Run

    html_report.py rocfft cufft report

Open `report/figs.html` in your favourite browser.

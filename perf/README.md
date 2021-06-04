# Performance

## rocFFT vs cuFFT performance

### Overview

This tool:
- builds rocFFT (when on AMD platforms)
- builds hipFFT (optionally)
- builds the Python hipFFT wrapper (optionally)
- runs a bunch of FFTs and records their GPU execution time

There are two main modes:
- Running a single performance suite using the hipFFT wrapper.  This
  would be useful for comparing rocFFT to cuFFT via hipFFT.
- Running a comparison between two rocFFT commits.

In either case, the general workflow is:
- Build one or more versions.  Each build is typicall installed into
  its own directory.
- Run the performance suites using the builds.

### Building

The builds are done locally and installed into a build directory.
To build rocFFT:

    perf.py build --rocfft <commit> --destination <install-dir>
    
To build hipFFT (optionally):

    perf.py build --hipfft <commit> --destination <install-dir> [--cuda]

This uses git to clone the (public) repos and does the builds using
appropriate flags.  Special care is taken to ensure library paths are
set correctly.

When building on a CUDA platform:

    export CUDA_PATH=/usr/local/cuda
    export PATH=$CUDA_PATH/bin:$PATH
    perf build --hipfft <commit> --destination <install-dir> --cuda

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

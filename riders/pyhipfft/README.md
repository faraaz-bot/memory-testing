# Python wrapper for hipFFT

## Install

Run:

    python3 setup.py build
    python3 setup.py install --user

## Basic usage

Basic transforms:

    import hipfft
    import numpy as np
    import numpy.random as nr

    n = 1024
    y = np.asarray(nr.rand(n) + 1j * nr.rand(n))
    z = hipfft.forward(y)


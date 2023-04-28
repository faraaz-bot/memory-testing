#!/usr/bin/python3

import numpy as np
import random

# Populates an nx by ny matrix with randomized real values
def rinit_2d(nx, ny, seed=10):
    random.seed(seed)
    f = np.empty([nx, ny])
    for i in range(nx):
        for j in range(ny):
            f[i, j] = random.random()
    return f

def impose_expected_format_2d(tdata, nx, ny):
    sdata = np.empty([nx, ny // 2 + 1], dtype=complex)

    for j in range(ny // 2 + 1):
        # fill up to nx // 2 + 1 with copied values
        for i in range(nx // 2 + 1):
            sdata[i][j] = tdata[i][j]
        
        # fill remaining with conjugate values
        for i in range(nx // 2 + 1, nx):
            sdata[i][j] = tdata[nx - i][(ny - j) % ny].conj()

    return sdata

def test(nx, ny, seed):
    # initialize data
    x = rinit_2d(nx, ny, seed)

    # transpose direction so that the even direction is contiguous
    x_transpose = np.transpose(x)

    # execute real-complex transform
    X_transpose = np.fft.rfft2(x_transpose)

    # transpose back
    X = np.transpose(X_transpose)

    # impose expected format
    X_format = impose_expected_format_2d(X, nx, ny)

    # compute oracle
    X_oracle = np.fft.rfft2(x)

    # compare result to oracle
    if (not np.allclose(X_oracle, X_format)):
        print("Failed", nx, ny, seed)
        exit(0)

# try different sizes/values
for nx in range(1, 101):
    for ny in range(1, 101):
        for seed in range(3):
            test(nx, ny, seed)

#!/usr/bin/python3

import numpy as np
import random

np.set_printoptions(precision=3)

def rinit_2d(nx, ny, seed=10):
    random.seed(seed)
    f = np.empty([nx, ny])
    for i in range(nx):
        for j in range(ny):
            f[i, j] = random.random()
    return f

def rinit_3d(nx, ny, nz, seed=10):
    random.seed(seed)
    f = np.empty([nx, ny, nz])
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                f[i, j, k] = random.random()
    return f 

def impose_expected_format_2d(tdata, nx, ny):
    sdata = np.empty([nx, ny // 2 + 1], dtype=complex)

    for j in range(ny // 2 + 1):
        # fill up to nx // 2 + 1 with copied values
        for i in range(nx // 2 + 1):
            sdata[i][j] = tdata[i][j]
        
        # fill remaining with conjugate values
        for i in range(nx // 2 + 1, nx):
            sdata[i][j] = tdata[nx - i][0 if j == 0 else ny - j].conj()

    return sdata

def impose_expected_format_3d(tdata, nx, ny, nz):
    sdata = np.zeros([nx, ny, nz // 2 + 1], dtype=complex)

    for k in range(nz // 2 + 1):
        for j in range(ny):
            # fill up to nx // 2 + 1 with copied values
            for i in range(nx // 2 + 1):
                sdata[i][j][k] = tdata[i][j][k]

        # fill remaining with conjugate values
            for i in range(nx // 2 + 1, nx):
                sdata[i][j][k] = tdata[nx - i][0 if j == 0 else ny - j][0 if k == 0 else nz - k].conj()

    return sdata

def test_2d(nx, ny, seed):
    # initialize data
    x = rinit_2d(nx, ny, seed)

    # transpose direction
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

def test_3d(nx, ny, nz, seed):
    # initialize data
    x = rinit_3d(nx, ny, nz, seed)

    # transpose direction
    x_transpose = np.transpose(x)

    # execute real-complex transform
    X_transpose = np.fft.rfftn(x_transpose)

    # transpose back
    X = np.transpose(X_transpose)

    # impose expected format
    X_format = impose_expected_format_3d(X, nx, ny, nz)

    # compute oracle
    X_oracle = np.fft.rfftn(x)

    if (not np.allclose(X_oracle, X_format)):
        print("Failed", nx, ny, nz, seed)
        exit(0)

# run 2d tests
# for nx in range(1, 100 + 1):
#     for ny in range(1, 100 + 1):
#         for seed in range(3):
#             test_2d(nx, ny, seed)

# run 3d tests
# for nx in range(1, 25 + 1):
#     for ny in range(1, 25 + 1):
#         for nz in range (1, 25 + 1):
#             for seed in range(3):
#                 test_3d(nx, ny, nz, seed)

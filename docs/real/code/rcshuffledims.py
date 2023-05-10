#!/usr/bin/python3

import numpy as np
import random

np.set_printoptions(precision=3)

def array_random_strides(a):
    axes = []
    for i in range(len(a)):
        axes.append(i)
    
    axes = np.random.permutation(axes)

    dims = []
    for axis in axes:
        dims.append(a[axis])
    f = np.empty(dims)

    return np.transpose(f, np.argsort(axes))

def rinit_2d(nx, ny, seed=10):
    random.seed(seed)
    f = array_random_strides([nx, ny])
    for i in range(nx):
        for j in range(ny):
            f[i, j] = random.random()
    return f

def rinit_3d(nx, ny, nz, seed=10):
    random.seed(seed)
    f = array_random_strides([nx, ny, nz])
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                f[i, j, k] = random.random()
    return f 

def impose_expected_format_2d(tdata, nx, ny):
    sdata = np.empty([nx, ny // 2 + 1], dtype=complex)

    for j in range(ny // 2 + 1):
        # fill section with copied values
        for i in range(nx // 2 + 1):
            sdata[i][j] = tdata[i][j]
        
        # fill remaining with conjugate values
        for i in range(nx // 2 + 1, nx):
            sdata[i][j] = tdata[nx - i][0 if j == 0 else ny - j].conj()

    return sdata

def impose_expected_format_3d(tdata, nx, ny, nz, axes):
    sdata = np.empty([nx, ny, nz // 2 + 1], dtype=complex)

    for k in range(nz // 2 + 1):
        for j in range(ny // 2 + 1):
            # fill section with copied values
            for i in range(nx // 2 + 1):
                sdata[i][j][k] = tdata[i][j][k]
            
            # fill section with copied or conjugate values depending on transpose axes
            for i in range(nx // 2 + 1, nx):
                if axes[2] == 0:
                    sdata[i][j][k] = tdata[nx - i][0 if j == 0 else ny - j][0 if k == 0 else nz - k].conj()
                else:
                    sdata[i][j][k] = tdata[i][j][k]
    
        for j in range(ny // 2 + 1, ny):
            # fill section with copied or conjugate values depending on transpose axes
            for i in range(nx // 2 + 1):
                if axes[2] == 1:
                    sdata[i][j][k] = tdata[0 if i ==  0 else nx - i][ny - j][0 if k == 0 else nz - k].conj()
                else:
                    sdata[i][j][k] = tdata[i][j][k]
            
            # fill section with copied or conjugate values depending on transpose axes
            for i in range(nx // 2 + 1, nx):
                if axes[2] == 2:
                    sdata[i][j][k] = tdata[i][j][k]
                else:
                    sdata[i][j][k] = tdata[0 if i ==  0 else nx - i][ny - j][0 if k == 0 else nz - k].conj()

    return sdata

def test_2d(nx, ny, seed):
    # initialize data
    x = rinit_2d(nx, ny, seed)

    # no need to transpose
    if (not nx % 2 == 0 and ny % 2 == 0
        or x.strides[0] > x.strides[1] and (not nx % 2 == 0 or ny % 2 == 0)):
        return
    
    # transpose directions
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
    if not np.allclose(X_oracle, X_format):
        print("Failed", nx, ny, seed)
        exit(0)

def test_3d(nx, ny, nz, seed):
    # initialize data
    x = rinit_3d(nx, ny, nz, seed)

    dims = [nx, ny, nz]
    axes = [0, 1, 2]

    # sort strides in descending order
    if x.strides[axes[1]] < x.strides[axes[2]]:
        axes[1], axes[2] = axes[2], axes[1]
    if x.strides[axes[0]] < x.strides[axes[2]]:
        axes[0], axes[2] = axes[2], axes[0]
    if x.strides[axes[0]] < x.strides[axes[1]]:
        axes[0], axes[1] = axes[1], axes[0]


    # maximize contiguity for even dimensions
    if (dims[axes[0]] % 2 != 0):
        if (dims[axes[1]] % 2 == 0) and (dims[axes[2]] % 2 != 0):
            axes[1], axes[2] = axes[2], axes[1]
    else:
        if (dims[axes[1]] % 2 == 0):
            if (dims[axes[2]] % 2 != 0):
                axes = np.roll(axes, 1)
        else:
            if (dims[axes[2]] % 2 == 0):
                axes[0], axes[1] = axes[1], axes[0]
            else:
                axes = np.roll(axes, -1)

    # no need to transpose
    if list(axes) == [0, 1, 2]:
        return
 
    # transpose direction
    x_transpose = np.transpose(x, axes)
    
    # execute real-complex transform
    X_transpose = np.fft.rfftn(x_transpose)

    # transpose back
    X = np.transpose(X_transpose, np.argsort(axes))

    # impose expected format
    X_format = impose_expected_format_3d(X, nx, ny, nz, axes)

    # compute oracle
    X_oracle = np.fft.rfftn(x)

    if not np.allclose(X_oracle, X_format):
        print("Failed", nx, ny, nz, seed)
        exit(0)

# run 2d tests
for nx in range(1, 100 + 1):
    for ny in range(1, 100 + 1):
        for seed in range(3):
            test_2d(nx, ny, seed)

# run 3d tests
for nx in range(1, 25 + 1):
    for ny in range(1, 25 + 1):
        for nz in range (1, 25 + 1):
            for seed in range(3):
                test_3d(nx, ny, nz, seed)
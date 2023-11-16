#!/usr/bin/python3

import numpy as np
import random

np.set_printoptions(precision=3)

# makes an empty NumPy array with a random stride order
def array_random_strides(a, seed=10):
    axes = []
    for i in range(len(a)):
        axes.append(i)
    axes = np.random.RandomState(seed=seed).permutation(axes)
    dims = []
    for axis in axes:
        dims.append(a[axis])
    f = np.zeros(dims)
    return np.transpose(f, np.argsort(axes))

def rinit_2d(nx, ny, seed=10):
    random.seed(seed)
    f = array_random_strides([nx, ny], seed)
    for i in range(nx):
        for j in range(ny):
            f[i, j] = random.random()
    return f

def rinit_3d(nx, ny, nz, seed=10):
    random.seed(seed)
    f = array_random_strides([nx, ny, nz], seed)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                f[i, j, k] = random.random()
    return f 

def sort_strides(strides, axes):
    result = []
    for axis in axes:
        result.append(strides[axis])
    return tuple(result)

def complex_conjugate(x):
    return x.real - 1j * (x.imag)

def reverse_axes_3d(x):
    result = [None, None, None]
    for i in range(len(x)):
        result[x[i]] = i
    return result

def impose_expected_format_2d(tdata, tstrides, nx, ny):
    sdata = np.zeros([nx * (ny // 2 + 1)], dtype=complex)
    sstrides = [len(sdata) // nx, 
                len(sdata) // (nx * (ny // 2 + 1))]

    for j in range(ny // 2 + 1):
        # fill section with copied values
        for i in range(nx // 2 + 1):
            sdata_idx = i * sstrides[0] + j * sstrides[1]
            tdata_idx = i * tstrides[0] + j * tstrides[1]
            sdata[sdata_idx] = tdata[tdata_idx]
        
        # fill remaining with conjugate values
        for i in range(nx // 2 + 1, nx):
            sdata_idx = i * sstrides[0] + j * sstrides[1]
            tdata_idx = ((nx - i) * tstrides[0] + 
                         (0 if j == 0 else ny - j) * tstrides[1])
            sdata[sdata_idx] = complex_conjugate(tdata[tdata_idx])

    return sdata, sstrides

def impose_expected_format_3d(tdata, tstrides, nx, ny, nz, axes):
    sdata = np.zeros([nx * ny * (nz // 2 + 1)], dtype=complex)
    sstrides = [len(sdata) // nx, 
                len(sdata) // (nx * ny), 
                len(sdata) // (nx * ny * (nz // 2 + 1))]
    
    for k in range(nz // 2 + 1):
        for j in range(ny // 2 + 1):
            # fill section with copied values
            for i in range(nx // 2 + 1):
                sdata_idx = i * sstrides[0] + j * sstrides[1] + k * sstrides[2]
                tdata_idx = i * tstrides[0] + j * tstrides[1] + k * tstrides[2]
                sdata[sdata_idx] = tdata[tdata_idx]
            
            # fill section with copied or conjugate values depending on transpose axes
            for i in range(nx // 2 + 1, nx):
                sdata_idx = i * sstrides[0] + j * sstrides[1] + k * sstrides[2]
                if axes[2] == 0:
                    tdata_idx = ((nx - i) * tstrides[0] +
                                 (0 if j == 0 else ny - j) * tstrides[1] +
                                 (0 if k == 0 else nz - k) * tstrides[2])
                    sdata[sdata_idx] = complex_conjugate(tdata[tdata_idx])
                else:
                    tdata_idx = (i * tstrides[0] + 
                                 j * tstrides[1] + 
                                 k * tstrides[2])
                    sdata[sdata_idx] = tdata[tdata_idx]
    
        for j in range(ny // 2 + 1, ny):
            # fill section with copied or conjugate values depending on transpose axes
            for i in range(nx // 2 + 1):
                sdata_idx = i * sstrides[0] + j * sstrides[1] + k * sstrides[2]
                if axes[2] == 1:
                    tdata_idx = ((0 if i == 0 else nx - i) * tstrides[0] +
                                 (ny - j) * tstrides[1] +
                                 (0 if k == 0 else nz - k) * tstrides[2])
                    sdata[sdata_idx] = complex_conjugate(tdata[tdata_idx])
                else:
                    tdata_idx = (i * tstrides[0] + 
                                 j * tstrides[1] + 
                                 k * tstrides[2])
                    sdata[sdata_idx] = tdata[tdata_idx]
            
            # fill section with copied or conjugate values depending on transpose axes
            for i in range(nx // 2 + 1, nx):
                sdata_idx = i * sstrides[0] + j * sstrides[1] + k * sstrides[2]
                if axes[2] == 2:
                    tdata_idx = (i * tstrides[0] + 
                                 j * tstrides[1] + 
                                 k * tstrides[2])
                    sdata[sdata_idx] = tdata[tdata_idx]
                else:
                    tdata_idx = ((0 if i == 0 else nx - i) * tstrides[0] +
                                 (ny - j) * tstrides[1] +
                                 (0 if k == 0 else nz - k) * tstrides[2])
                    sdata[sdata_idx] = complex_conjugate(tdata[tdata_idx])

    return sdata, sstrides

def test_2d(nx, ny, seed):
    # initialize data
    x = rinit_2d(nx, ny, seed)

    # flatten into 1d array and strides
    istrides = [s // x.itemsize for s in x.strides]
    x_flat = x.flatten('K')

    # no need to transpose
    if (not nx % 2 == 0 and ny % 2 == 0
        or istrides[0] > istrides[1] and (not nx % 2 == 0 or ny % 2 == 0)):
        return
    
    # transpose directions
    istrides = sort_strides(istrides, [1, 0])

    # reshape into 2d array to execute transform
    x_transpose = np.lib.stride_tricks.as_strided(
        x_flat, shape=[ny, nx], strides=[s * x_flat.itemsize for s in istrides])

    # execute real-complex transform
    X_transpose = np.fft.rfft2(x_transpose)

    # flatten into 1d array and strides
    ostrides = [s // X_transpose.itemsize for s in X_transpose.strides]
    X_transpose_flat = X_transpose.flatten('K')

    # transpose back
    ostrides = sort_strides(ostrides, [1, 0])

    # impose expected format
    X_flat_format, ostrides = impose_expected_format_2d(
        X_transpose_flat, ostrides, nx, ny)

    # reshape into 2d array for comparison
    X_format = np.lib.stride_tricks.as_strided(
        X_flat_format, 
        shape=[nx, ny // 2 + 1], 
        strides=[s * X_flat_format.itemsize for s in ostrides])

    # compute oracle
    X_oracle = np.fft.rfft2(x)

    # compare result to oracle
    if not np.allclose(X_oracle, X_format):
        print("Failed", nx, ny, seed)
        exit(0)

def test_3d(nx, ny, nz, seed):
    # initialize data
    x = rinit_3d(nx, ny, nz, seed)

    # flatten into 1d array and strides
    istrides = [s // x.itemsize for s in x.strides]
    x_flat = x.flatten('K')
    
    dims = [nx, ny, nz]
    axes = [0, 1, 2]

    # sort strides in descending order
    if istrides[axes[1]] < istrides[axes[2]]:
        axes[1], axes[2] = axes[2], axes[1]
    if istrides[axes[0]] < istrides[axes[2]]:
        axes[0], axes[2] = axes[2], axes[0]
    if istrides[axes[0]] < istrides[axes[1]]:
        axes[0], axes[1] = axes[1], axes[0]

    # maximize contiguity for even dimensions
    if (dims[axes[0]] % 2 != 0):
        if (dims[axes[1]] % 2 == 0) and (dims[axes[2]] % 2 != 0):
            axes[1], axes[2] = axes[2], axes[1]
    else:
        if (dims[axes[1]] % 2 == 0):
            if (dims[axes[2]] % 2 != 0):
                axes[0], axes[1], axes[2] = axes[2], axes[0], axes[1]
        else:
            if (dims[axes[2]] % 2 == 0):
                axes[0], axes[1] = axes[1], axes[0]
            else:
                axes[0], axes[1], axes[2] = axes[1], axes[2], axes[0]

    # no need to transpose
    if list(axes) == [0, 1, 2]:
        return
 
    # transpose direction
    istrides = sort_strides(istrides, axes)

    # reshape into 3d array to execute transform
    x_transpose = np.lib.stride_tricks.as_strided(
        x_flat, 
        shape=[dims[a] for a in axes], 
        strides=[s * x_flat.itemsize for s in istrides])
    
    # execute real-complex transform
    X_transpose = np.fft.rfftn(x_transpose)

    # flatten into 1d array and strides
    ostrides = [s // X_transpose.itemsize for s in X_transpose.strides]
    X_transpose_flat = X_transpose.flatten('K')

    # transpose back
    ostrides = sort_strides(ostrides, reverse_axes_3d(axes))

    # impose expected format
    X_flat_format, ostrides = impose_expected_format_3d(
        X_transpose_flat, ostrides, nx, ny, nz, axes)
    
    # reshape into 3d array for comparison
    X_format = np.lib.stride_tricks.as_strided(
        X_flat_format, 
        shape=[nx, ny, nz // 2 + 1],
        strides=[s * X_flat_format.itemsize for s in ostrides])
    
    # compute oracle
    X_oracle = np.fft.rfftn(x)

    # compare result to oracle
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

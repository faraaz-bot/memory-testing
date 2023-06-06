#!/usr/bin/env python3

import numpy as np
import cmath

np.set_printoptions(suppress=True,linewidth=np.nan)

import sys
sys.path.insert(1, '../../real/code')
import rckernels

# Batch of 1D transforms.
length = [4]
nbatch = 4

# Complex length:
hlength = [length[-1] // 2 + 1]

# Allocate and initialize the data:
x = np.zeros(shape=np.append(nbatch, hlength), dtype=complex)
init = lambda ibatch, ix : ix
for ibatch in range(len(x)):
    for ix in range(len(x[ibatch])):
        x[ibatch, ix] = init(ibatch, ix)
print(x)

# The readop and writeop lambdas would be used like callbacks in
# rocFFT; the goal is to avoid a global memory round-trip.  In this
# sample python code, we're doing extra read/writes, but this isn't
# representative of the 

# Move each batch one index over:
readop = lambda readval, ibatch, ixd: readval * cmath.exp(1j * 2.0 * np.pi * ( ixd[0] * ibatch) / length[0] )
#readop = None

# Perform a function on the output that depends frequency and batch:
writeop = lambda writeval, ibatch, idx: writeval * np.exp(1 / (ibatch + idx[0] + 1))
#writeop = None

# Perform the pointwise read op, perform the batched 1D
xout = rckernels.crfft_even(x, length, nbatch, readop, writeop)

# The output:
print(xout)

#!/usr/bin/env python3

import numpy as np
import cmath

np.set_printoptions(suppress=True,linewidth=np.nan)

# Batch of 1D transforms.
length = 4
nbatch = 4


# Complex length:
hlength = length // 2 + 1

# Allocate and initialize the data:
x = np.zeros(shape=(nbatch, hlength), dtype=complex)
init = lambda ibatch, ix : ix
for ibatch in range(len(x)):
    for ix in range(len(x[ibatch])):
        x[ibatch, ix] = init(ibatch, ix)
print(x)

# The readop and writeop lambdas would be used like callbacks in
# rocFFT; the goal is to avoid a global memory round-trip.  In this
# sample python code, we're doing extra read/writes, but this isn't
# representative of the 

# If the length is even, move the Fourier origin to length/2:
# readop = lambda readval, ibatch, ix: -readval if (ix % 2 != 0 and ibatch > 0) else readval

# Move each batch one index over:
readop = lambda readval, ibatch, ix: readval * cmath.exp(1j * 2 * np.pi * (ix * ibatch) / length)

# Perform a function on the output that depends frequency and batch:
writeop = lambda writeval, ibatch, ix: writeval * np.exp(1/(ibatch + ix + 1))

# Allocate the output buffer:
xout = np.zeros(shape=(nbatch, length), dtype=float)

# Perform the pointwise read op, perform the batched 1D
# real-to-complex transform, and perform the write op.
for ibatch in range(len(x)):
    for ix in range(len(x[ibatch])):
        x[ibatch, ix] = readop(x[ibatch, ix], ibatch, ix)
    xout[ibatch] = np.fft.irfft(x[ibatch], length)
    for ix in range(len(xout[ibatch])):
        xout[ibatch, ix] = writeop(xout[ibatch, ix], ibatch, ix)

# The output:
print(xout)

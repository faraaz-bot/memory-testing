#!/usr/bin/env python3

import numpy as np
import cmath

#np.set_printoptions(suppress=True,linewidth=np.nan)

import sys
sys.path.insert(1, '../../real/code')
import rckernels

import operator

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d

import itertools
import functools

# Physical domain size:
domain = [1, 1]

# Batch of 1D transforms:
N = 15
length = [N, N]
nbatch = 1


# Allocate and initialize the data:

init = lambda ibatch, idx : (length[0]/ 2 - idx[0]) * (length[0]/ 2 - idx[0])
if len(length) == 2:
    init = lambda ibatch, idx : (length[0]/ 2 - idx[0]) * (length[0]/ 2 - idx[0]) +  (length[1]/ 2 - idx[1]) * (length[1]/ 2 - idx[1])
    
x = np.zeros(shape=np.append(nbatch, length), dtype=float)

for ibatch in range(len(x)):
    for idx in (list(itertools.product(*[range(l) for l in length]))):
        x[ibatch][idx] = init(ibatch, idx)
#print(x[0])

if len(length) == 1:
    plt.plot(x[0])
elif len(length) == 2:
    plt.imshow(x[0])
plt.show()

# The readop and writeop lambdas would be used like callbacks in
# rocFFT; the goal is to avoid a global memory round-trip.  In this
# sample python code, we're doing extra read/writes, but this isn't
# representative of the 

# Move each batch one index over:
#readop = lambda readval, ibatch, ixd: readval * cmath.exp(1j * 2.0 * np.pi * ( ixd[0] * ibatch) / length[0] )
readop = None

k0 = np.empty(len(length))
for dim in range(len(k0)):
    k0[dim] = 2.0 * np.pi * domain[dim] / length[dim]
print(k0)
k0squared = np.empty(len(k0))

midop = None
if len(length) == 1:
    midop = lambda readval, ibatch, idx: 0 if idx[0] == 0 else readval * ( 1.0 - 1.0 / (idx[0] * idx[0] * k0squared[0]) )
elif len(length) == 2:
    midop = lambda readval, ibatch, idx: 0 if (idx[0] == 0 and idx[1] == 0) else norm * readval * ( 1.0 - 1.0 / (idx[0] * idx[0] * k0squared[0] + idx[1] * idx[1] * k0squared[1]) )

# Perform a function on the output that depends frequency and batch:
norm = 1.0 / functools.reduce(operator.mul, length, 1)
#writeop = lambda writeval, ibatch, idx: writeval * norm
writeop = None

# Perform the pointwise read op, perform the batched 1D
xout = rckernels.rfft_round(x, length, nbatch, readop, midop, writeop)

# The output:
#print(xout[0])


if len(length) == 1:
    plt.plot(xout[0])
elif len(length) == 2:
    plt.imshow(xout[0])
plt.show()

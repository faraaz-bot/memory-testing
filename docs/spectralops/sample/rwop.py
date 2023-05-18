#!/usr/bin/env python3

import numpy as np
np.set_printoptions(suppress=True,linewidth=np.nan)

import cmath

import sys
sys.path.insert(1, '../../real/code')
import rckernels

import itertools

# Batch of 1D transforms.
length = np.array([3, 4], dtype=int)
nbatch = 1

# Allocate and initialize the data:
init = lambda ibatch, idx : np.exp(1.0 / (ibatch + idx[0] + idx[1]* idx[1] + 1))
#init = lambda ibatch, idx : np.exp(1.0 / (ibatch + idx[0]  + 1))
x = np.zeros(shape=np.append(nbatch, length), dtype=float)
for ibatch in range(nbatch):
    for idx in (list(itertools.product(*[range(l) for l in length]))):
        x[ibatch][idx] = init(ibatch, idx)
print("x:")
print(x)

X = rckernels.rcfft_even(x, length, nbatch)

print("X:")
print(X)

xlength = np.append(nbatch, length)
X0 = np.fft.rfftn(x, axes=range(1, len(xlength)))

print("X0:")
print(X0)
print(np.allclose(X, X0))

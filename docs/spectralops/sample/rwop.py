#!/usr/bin/env python3

import numpy as np
np.set_printoptions(suppress=True,linewidth=np.nan)

import cmath

import sys
sys.path.insert(1, '../../real/code')
import rckernels

import itertools, copy

# Batch of 1D transforms.
length = [4, 4]
nbatch = 1

# Allocate and initialize the (real) data:
rinit = lambda ibatch, idx : np.exp(1.0 / (ibatch + idx[0]  + 1))
if len(length) > 1:
    rinit = lambda ibatch, idx : np.exp(1.0 / (ibatch + idx[0] + idx[1]* idx[1] + 1))

print("input:")
x = np.zeros(shape=np.append(nbatch, length), dtype=float)
for ibatch in range(nbatch):
    for idx in (list(itertools.product(*[range(l) for l in length]))):
        x[ibatch][idx] = rinit(ibatch, idx)
print("x:")
print(x)

readop = None
readop = lambda readval, ibatch, idx: readval * cmath.exp( 2.0 * np.pi * 1j * (idx[0]) / length[0])
#readop = lambda readval, ibatch, idx: readval if (idx[0] % 2 == 0) else -readval

writeop = None
writeop = lambda readval, ibatch, idx: 2 * readval


print("np.fft.rfftn:")
X0 = np.fft.rfftn(x, axes=range(1, len(np.append(nbatch, length))))
print("X0:")
print(X0)

print("rcfft_even:")
X = rckernels.rcfft_even(x, length, nbatch, readop=readop, writeop=writeop)
print("X:")
print(X)
print("all close:", np.allclose(X, X0))
   
print("rcfft_embed:")
X = rckernels.rcfft_embed(x, length, nbatch, readop=readop, writeop=writeop)
print("X:")
print(X)
print("all close:", np.allclose(X, X0))

print("rcfft_pair:")
X = rckernels.rcfft_pair(x, length, nbatch, readop=readop, writeop=writeop)
print("X:")
print(X)
print("all close:", np.allclose(X, X0))

print("rcfft:")
X = rckernels.rcfft(x, length, nbatch, readop=readop, writeop=writeop)
print("X:")
print(X)
print("all close:", np.allclose(X, X0))



print()

# Allocate and initialize the (real) data:
cinit = lambda ibatch, idx : np.exp(1.0 / (ibatch + idx[0]  + 1))
if len(length) > 1:
    cinit = lambda ibatch, idx : np.exp(1.0 / (ibatch + idx[0] + idx[1]* idx[1] + 1))

hlength = copy.deepcopy(length)
hlength[-1] = hlength[-1] // 2 + 1 


X = np.zeros(shape=np.append(nbatch, hlength), dtype=complex)
for ibatch in range(nbatch):
    for idx in (list(itertools.product(*[range(l) for l in hlength]))):
        X[ibatch][idx] = cinit(ibatch, idx)
    if len(hlength) == 1:
        X[ibatch] = rckernels.symmetrize_1d(X[ibatch], length[0])
    elif len(hlength) == 2:
        X[ibatch] = rckernels.symmetrize_2d(X[ibatch], length[0], length[1])
    elif len(hlength) == 3:
        X[ibatch] = rckernels.symmetrize_3d(X[ibatch], length[0], length[1], length[2])
print("X:")
print(X)

ireadop = lambda readval, ibatch, idx: 2 * readval
iwriteop = lambda readval, ibatch, idx: 1 * readval

print("x0")
x0 = np.fft.irfftn(X, axes=range(1, len(np.append(nbatch, length)))) * np.prod(length)
print(x0)

print("crfft_even")
x = rckernels.crfft_even(X, length, nbatch, ireadop, iwriteop)
print(x)
print("all close:", np.allclose(x, x0))

print("crfft_embed")
x = rckernels.crfft_embed(X, length, nbatch, ireadop, iwriteop)
print(x)
print("all close:", np.allclose(x, x0))

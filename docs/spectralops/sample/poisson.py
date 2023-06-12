#!/usr/bin/env python3

import numpy as np
import cmath
import copy

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
N = 128
length = [N, N]
nbatch = 1


# Allocate and initialize the data:

# Initialize the data:
init = lambda ibatch, idx : np.cos( 2.0 * np.pi * idx[0] / length[0] )
if len(length) == 2:
    init = lambda ibatch, idx : np.cos( 2.0 * np.pi * idx[0] / length[0] ) + np.cos( 3.0 * 2.0 * np.pi * idx[1] / length[1] )
    
f_input = np.zeros(shape=np.append(nbatch, length), dtype=float)

for ibatch in range(len(f_input)):
    for idx in (list(itertools.product(*[range(l) for l in length]))):
        f_input[ibatch][tuple(idx)] = init(ibatch, tuple(idx))
        
#print(x[0])

if True:
    if len(length) == 1:
        plt.plot( f_input[0], label="input" )
        plt.legend()
    elif len(length) == 2:
        plt.imshow( f_input[0])
        plt.title("input")
    plt.show()


# Index to wavenumber function for non-symmetrized dimensions:
# (cf: fftfreq, but, well, better?)
idx2ik = lambda ix, Nx, dx: 2.0 * np.pi * (ix if ix < Nx // 2 else Nx  - ix) / dx
#idx2ik = lambda ix, Nx, dx: 2.0 * np.pi * (ix if ix < Nx // 2 else Nx  - ix) / dx
    
def poisson_midop1(readval, ibatch, idx):
    global length
    global domain

    # Norm is for the FFT round-trip norm thing.
    norm = 1.0 / functools.reduce(operator.mul, length, 1)
    if idx[0] == 0:
        return 0
    kx = 2.0 * np.pi * idx[0] / domain[0]
    return -norm * readval / ( kx * kx )

def poisson_midop2(readval, ibatch, idx):
    global length
    global domain

    # Norm is for the FFT round-trip norm thing.
    norm = 1.0 / functools.reduce(operator.mul, length, 1)
    if idx[0] == 0 and idx[1] == 0:
        return 0
    
    # Even-ky case:
    ikx = ( idx[0] if idx[0] < (length[0] // 2) else ( length[0] - idx[0] ) )
    #iky = ( idx[1] if idx[1] < (length[1] // 2) else ( length[1] - idx[1] ) )
    iky = idx[1]
    
    kx = 2.0 * np.pi * ikx / domain[0]
    ky = 2.0 * np.pi * iky / domain[1]

    #print(idx[0], ikx, idx[1], iky)
    
    return -norm * readval / ( kx * kx + ky * ky)
   
# The readop and writeop lambdas would be used like callbacks in
# rocFFT; the goal is to avoid a global memory round-trip.  In this
# sample python code, we're doing extra read/writes, but this isn't
# representative of the 

# Move each batch one index over:
#readop = lambda readval, ibatch, ixd: readval * cmath.exp(1j * 2.0 * np.pi * ( ixd[0] * ibatch) / length[0] )
readop = None

# cf: https://math.stackexchange.com/questions/1809871/solve-poisson-equation-using-fft


norm = 1.0 / functools.reduce(operator.mul, length, 1)

   
# Perform a function on the output that depends frequency and batch:

#writeop = lambda writeval, ibatch, idx: writeval * norm
writeop = None

# Perform the pointwise read op, perform the batched 1D

if len(length) == 1:
    phi = rckernels.rfft_round(f_input, length, nbatch, readop=readop, midop=poisson_midop1, writeop=writeop)
elif len(length) == 2:
    phi = rckernels.rfft_round(f_input, length, nbatch, readop=readop, midop=poisson_midop2, writeop=writeop)
        
if True:
    if len(length) == 1:
        plt.plot(phi[0], label="phi")
        plt.legend()
        plt.show()
    elif len(length) == 2:
        #fig, ax = plt.subplots(1)
        plt.imshow(phi[0])
        #ax.text(5, 5, "phi")
        plt.title("phi")
        plt.show()


eps = 1e-10

# Let's compute the finite-difference of phi and compare it with x:
for ibatch in range(nbatch):
    maxerr = 0.0
    lap_phi = np.empty(shape=length, dtype=float)
    if len(length) == 1:
        for ix in range(length[0]):
            dx = domain[0] / length[0]
            ixp = (ix + 1) % length[0]
            ixm = (ix - 1) % length[0]
            lap_phi[ix] = ( phi[ibatch][ixm] - 2.0 * phi[ibatch][ix] + phi[ibatch][ixp] ) / ( dx * dx )
            err = np.abs( lap_phi[ix] - f_input[ibatch][ix] )
            print(lap_phi[ix], f_input[ibatch][ix], err)

            norm = np.abs(f_input[ibatch][ix])
            #print(norm)
            maxerr = max( maxerr, err / (norm + eps))
            #maxerr = max( maxerr, err / max(np.abs(lap_phi[ix]), np.abs(f_input[ibatch][ix]), eps) )
        print("max err:", maxerr)
        if len(length) == 1:
            plt.plot(lap_phi, label="lap_phi")
            plt.plot(f_input[0], label="f")
            plt.legend()
        elif len(length) == 2:
            plt.imshow(laph_phi, label="lap_phi")
            plt.title("lap_phi")

        plt.show()
    elif len(length) == 2:
        dx = domain[0] / length[1]
        dy = domain[1] / length[1]
        for ix in range(length[0]):
            ixp = (ix + 1) % length[0]
            ixm = (ix - 1) % length[0]
            for iy in range(length[1]):
                iyp = (iy + 1) % length[1]
                iym = (iy - 1) % length[1]
                lap_phi[ix, iy] = ( phi[ibatch][ixm, iy] - 2.0 * phi[ibatch][ix, iy] + phi[ibatch][ixp, iy] ) / ( dx * dx ) \
                    + ( phi[ibatch][ix, iym] - 2.0 * phi[ibatch][ix, iy] + phi[ibatch][ix, iyp] ) / ( dy * dy )
                err = np.abs( lap_phi[ix, iy] - f_input[ibatch][ix, iy] )
                norm = np.abs(f_input[ibatch][ix, iy])
                
                #maxerr = max( maxerr, err / (norm + eps) )
                maxerr = max( maxerr, err )
                
        print("max err:", maxerr)

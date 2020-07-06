#!/usr/bin/python3

import math
import cmath
import numpy as np
import sys

from rckernels import postkernel, prekernel

Nx = 4 if len(sys.argv) < 2 else int(sys.argv[1])
Ny = 4 if len(sys.argv) < 3 else int(sys.argv[2])
Nz = 4 if len(sys.argv) < 4 else int(sys.argv[3])

print("Nx:", Nx)
print("Ny:", Ny)
print("Nz:", Nz)

Nzhalf = Nz // 2
Nzcomplex = Nzhalf + 1

x = np.empty([Nx,Ny,Nz], dtype=float)
for i in range(0,Nx):
    for j in range(0,Ny):
        for k in range(0,Nz):
            x[i,j,k] = i + j + k
print("input:")
print(x)

print("np rfftn:")
Xref = np.fft.rfftn(x)
print(Xref)

print("our rfft3:")
# Cast to complex:
z = np.empty([Nx,Ny,Nzhalf], dtype=complex)
for i in range(0, Nx):
    for j in range(0, Ny):
        for k in range(0, Nzhalf):
            z[i,j,k] = complex(x[i, j, 2*k], x[i, j, 2*k + 1])
print(z)

# z-direction r2c fft
Z = np.empty([Nx, Ny, Nzhalf], dtype=complex)
for i in range(0, Nx):
    for j in range(0, Ny):
        Z[i,j,:] = np.fft.fft(z[i,j,:])
X = np.empty([Nx,Ny,Nzcomplex], dtype=complex)
for i in range(0, Nx):
    for j in range(0, Ny):
        X[i,j,:] = postkernel(Z[i,j,:])

# 2D c2c fft in x-y
for k in range(0, Nzcomplex):
    X[:,:,k] = np.fft.fft2(X[:,:,k])
    
            
print(X)

maxerr = 0.0
for i in range(0, Nx):
    for j in range(0, Ny):
        for k in range(0, Nzcomplex):
            diff = abs(X[i,j,k] - Xref[i,j,k])
            if diff > maxerr:
                maxerr = diff
print("maxerr: " + str(maxerr))



# inverse transform

print("np irfftn(rfftn):")
xref = np.fft.irfftn(np.fft.rfftn(x))*(Nx * Ny * Nz)
print(xref)

print("our round-trip:")
# preprocess in z direction:
for i in range(0, Nx):
    for j in range(0, Ny):
        Z[i,j,:] = prekernel(X[i,j,:])
    
# 3D c2c fft
z = np.fft.ifftn(Z) * Z.size
    
# cast from complex to real:
for i in range(0, Nx):
    for j in range(0, Ny):
            for k in range(0, Nzhalf):
                x[i, j, 2*k] = z[i,j,k].real
                x[i, j, 2*k+1] = z[i,j,k].imag
print(x)

maxerr = 0.0
for i in range(0, Nx):
    for j in range(0, Ny):
        for k in range(0, Nzcomplex):
            diff = abs(X[i,j,k] - Xref[i,j,k])
            if diff > maxerr:
                maxerr = diff
print("maxerr: " + str(maxerr))

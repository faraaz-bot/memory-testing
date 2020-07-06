#!/usr/bin/python3

import math
import cmath
import numpy as np
import sys

from rckernels import postkernel, prekernel

Nx = 8 if len(sys.argv) < 2 else int(sys.argv[1])
Ny = 16 if len(sys.argv) < 3 else int(sys.argv[2])

print("Nx:", Nx)
print("Ny:", Ny)

Nyhalf = Ny // 2
Nycomplex = Nyhalf + 1

np.set_printoptions(precision=2)

import random
random.seed(3)

print("input:")
x = np.empty([Nx,Ny], dtype=float)
for i in range(0,Nx):
    for j in range(0,Ny):
        x[i,j] = random.random()
print(x)

print("np rfft2:")
Xref = np.fft.rfft2(x)
print(Xref)

print("our rfft2:")
# cast to complex
z = np.empty([Nx,Nyhalf], dtype=complex)
for i in range(0, Nx):
    for j in range(0, Nyhalf):
        z[i,j] = complex(x[i, 2*j], x[i, 2*j + 1])

# y-direction c2c fft
Z = np.empty([Nx,Nyhalf], dtype=complex)
for i in range(0, Nx):
    Z[i,:] = np.fft.fft(z[i,:])
X = np.empty([Nx,Nycomplex], dtype=complex)
for i in range(0, Nx):
    X[i,:] = postkernel(Z[i,:])

# x-direction c2c fft
for i in range(0, Nycomplex):
    X[:,i] = np.fft.fft(X[:,i])
print(X)

maxerr = 0.0
for i in range(0, Nx):
    for j in range(0, Nycomplex):
        diff = abs(X[i,j] - Xref[i,j])
        if diff > maxerr:
            maxerr = diff
print("maxerr: " + str(maxerr))


# inverse transform
print()

print("np irfft2(rfft2):")
xref = np.fft.irfft2(np.fft.rfft2(x))*(Nx * Ny)
print(xref)

print("our round-trip:")

# x-direction c2c ifft, X->X
for i in range(0, Nycomplex):
    X[:,i] = np.fft.ifft(X[:,i]) * Nx
print(X)

# preprocess in y-direction, X->Z
for i in range(0, Nx):
     Z[i,:] = prekernel(X[i,:])

print("after prekernel:")
print(Z)

# y-direction c2c fft, Z->Z
for i in range(0, Nx):
    Z[i,:] = np.fft.ifft(Z[i,:]) * Nyhalf
print(Z / (Nx * Ny))

print("back to real:")
# cast to real: Z->x
x = np.zeros([Nx,Ny], dtype=float)
for i in range(0, Nx):
    for j in range(0, Nyhalf):
        x[i, 2 * j] = Z[i,j].real
        x[i, 2 * j + 1] = Z[i,j].imag
print(x)
        
maxerr = 0.0
for i in range(0, Nx):
    for j in range(0, Ny):
        diff = abs(x[i,j] - xref[i,j])
        if diff > maxerr:
            maxerr = diff
print("maxerr: " + str(maxerr))

# # preprocess in y-direction:
# for i in range(0, Nx):
#     Z[i,:] = prekernel(X[i,:])

# print("after prekernel:")
# print(Z)
    
# # 2D c2c fft:
# #z = np.fft.ifft2(Z) * Z.size

# # 2D c2c fft, line by line.
# for i in range(0, Nx):
#     z[i,:] = np.fft.ifft(Z[i,:]) * Nyhalf
# print("yffft:")
# print(z)

# for i in range(0, Nyhalf):
#     #print(i, "->", z[:,i])
#     z[:,i] = np.fft.ifft(z[:,i]) * Nx
#     #print(z[:,i])
# # print ("xfft:")
# # print(z / (Nx * Ny))

# # cast from complex to real:
# for i in range(0, Nx):
#     for j in range(0, Nyhalf):
#         x[i, 2*j] = z[i,j].real
#         x[i, 2*j+1] = z[i,j].imag
# print("output:")
# print(x / (Nx * Ny))

# maxerr = 0.0
# for i in range(0, Nx):
#     for j in range(0, Ny):
#         diff = abs(x[i,j] - xref[i,j])
#         if diff > maxerr:
#             maxerr = diff
# print("maxerr: " + str(maxerr))

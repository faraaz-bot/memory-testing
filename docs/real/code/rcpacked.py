#!/usr/bin/python3

import math
import cmath
import numpy as np

from rckernels import postkernel, prekernel, unpackbatch, repackbatch, unpackND, repackND
       
N = 8

print("N: " + str(N))

Nhalf = N // 2
Ncomplex = Nhalf + 1

x = np.empty([N])
y = np.empty([N])
for i in range(0, N):
    x[i] = 1.0 * i
    y[i] = 1.0 * i * i

print("input:")
print(x)
print(y)

print("np rfft:")
Xref = np.fft.rfft(x)
Yref = np.fft.rfft(y)

print("our rfft:")
z = np.empty([N], dtype=complex)
for i in range(0, N):
    z[i] = complex(x[i], y[i])

Z = np.fft.fft(z)

X, Y = unpackbatch(Z)
print(X)
print(Y)

errmax = 0.0
for i in range(0, Ncomplex):
    err = abs(X[i] - Xref[i])
    if err > errmax:
        errmax = err
print("X errmax: " + str(errmax))

errmax = 0.0
for i in range(0, Ncomplex):
    err = abs(Y[i] - Yref[i])
    if err > errmax:
        errmax = err
print("Y errmax: " + str(errmax))

print("our packed inverse transform:")
Z = repackbatch(X,Y,N)

z = np.fft.ifft(Z) * len(Z)

xback = np.empty([N])
yback = np.empty([N])

for i in range(0, N):
    xback[i] = z[i].real
    yback[i] = z[i].imag
print(xback)
print(yback)

xref = np.fft.irfft(np.fft.rfft(x),len(x))*len(x)
yref = np.fft.irfft(np.fft.rfft(y),len(y))*len(y)

errmax = 0.0
for i in range(0, Ncomplex):
    err = abs(xback[i] - xref[i])
    if err > errmax:
        errmax = err
print("x errmax: " + str(errmax))

errmax = 0.0
for i in range(0, Ncomplex):
    err = abs(yback[i] - yref[i])
    if err > errmax:
        errmax = err
print("y errmax: " + str(errmax))
    

print("2D direct transform")

Nx = 4
Ny = 4

hermshape = [ Nx, Ny // 2 + 1]

x = np.empty([Nx, Ny])
for i in range(0, Nx):
    for j in range(0, Ny):
        x[i, j] = 1.0 * i + 10 * j
print(x)
        
xplanar = np.empty([Nx // 2, Ny], dtype=complex)
for i in range(0, Nx // 2):
    for j in range(0, Ny):
        xplanar[i, j] = complex(x[2 * i, j], x[ 2 * i + 1, j])
print(xplanar)

Xplanar = np.empty([Nx // 2, Ny], dtype=complex)
for idx in range(Nx // 2):
    Xplanar[idx] = np.fft.fft(xplanar[idx])

print(Xplanar)
    
X = unpackND(Xplanar, hermshape)

print(X)

for idx in range(Ny // 2 + 1):
    X[:,idx] = np.fft.fft(X[:,idx])
print(X)

Xref = np.fft.rfft2(x)
print(Xref)
print(np.allclose(X, Xref))


print("2D inverse transform")

for idx in range(Ny // 2 + 1):
    X[:,idx] = np.fft.ifft(X[:,idx])
    
print(X)

Xplanar = repackND(X, [Nx, Ny])

print(Xplanar)

for idx in range(Nx // 2):
    xplanar[idx] = np.fft.ifft(Xplanar[idx])

print(xplanar)

for i in range(0, Nx // 2):
    for j in range(0, Ny):
        #xplanar[i, j] = complex(x[2 * i, j], x[ 2 * i + 1, j])
        x[2 * i, j] = xplanar[i,j].real
        x[2 * i + 1, j] = xplanar[i,j].imag
        
print(x)

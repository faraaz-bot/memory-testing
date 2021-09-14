#!/usr/bin/python3

import math
import cmath
import numpy as np
import random

nx = 4
ny = 4
nyp = ny // 2 + 1

def symmetrize_1d(data, nx):
    data[0] = data[0].real
    if nx % 2 == 0:
        nxp = nx // 2 + 1
        data[nxp - 1] = data[nxp - 1].real
    return data

def symmetrize_2d(data, nx, ny):
    nyp = ny // 2 + 1
    data[0][0] = data[0][0].real
    if ny % 2 == 0:
        data[0][nyp -1] = data[0][nyp -1].real
    if nx % 2 == 0:
        data[nx // 2][0] = data[nx // 2][0].real
        if ny % 2 == 0:
            data[nx // 2][nyp -1] = data[nx // 2][nyp - 1].real
    for i in range(1, nx // 2):
        data[nx - i][0] = data[i][0].conj()
    if ny % 2 == 0:
        for i in range(1, nx // 2):
            data[nx - i][nyp - 1] = data[i][nyp - 1].conj()
    return data

def is_symmetric_1d(data, nx):
    if not np.isclose(data[0].image, 0):
        return False
    if nx % 2 == 0:
        nxp = nx // 2 + 1
        if not np.isclose(data[nxp - 1].image, 0):
            return False
    return True

def is_symmetric_2d(data, nx, ny):
    if not np.isclose(data[0][0].imag, 0):
        return False
    nyp = ny // 2 + 1
    if ny % 2 ==0:
        if not np.isclose(data[0][nyp-1].imag, 0.0):
            return False
    for i in range(1, nx // 2):
        if not np.isclose(data[nx - i][0], data[i][0].conj()):
            return False
    if nx % 2 == 0:
        if not np.isclose(data[nx // 2][0].imag, 0.0):
            return False
        if ny % 2 ==0:
            if not np.isclose(data[nx // 2][nyp - 1].imag, 0.0):
                return False
        for i in range(1, nx // 2):
            if not np.isclose(data[nx - i][nyp - 1], data[i][nyp - 1].conj()):
                return False
    return True

def r2c_1d(rdata, nx):
    cdata = np.empty([nx], dtype=complex)
    for i in range(nx):
        cdata[i] = rdata[i]
    cdata = np.fft.fft(cdata)
    return cdata[0:nx // 2 + 1]

def c2r_1d(hdata, nx):
    cdata = np.empty([nx], dtype=complex)
    for i in range(nx // 2 + 1):
        cdata[i] = hdata[i]
    for i in range(nx // 2 + 1, nx):
        cdata[i] = hdata[nx - i].conj()
    cdata = np.fft.ifft(cdata)
    return cdata.real

def r2c_2d(rdata, nx, ny):
    cdata = np.empty([nx, ny], dtype=complex)
    for i in range(nx):
        for j in range(ny):
            cdata[i][j] = rdata[i][j]
    cdata = np.fft.fft2(cdata)
    return cdata[0:nx,0:ny // 2 + 1]

def c2r_2d(hdata, nx, ny):
    cdata = np.zeros([nx, ny], dtype=complex)
    for i in range(nx):
        for j in range(ny // 2 + 1):
            cdata[i][j] = hdata[i][j];
    for i in [0, nx // 2]:
        for j in range(ny // 2 + 1, ny):
            cdata[i][j] = hdata[i][ny - j].conj();
    for i in range(1, nx // 2):
        for j in range(ny // 2 + 1, ny):
            print(i,j)
            cdata[i][j] = hdata[nx - i][ny - j].conj();
            cdata[nx - i][j] = hdata[i][ny - j].conj();
    cdata = np.fft.ifft2(cdata)
    return cdata.real
    

print(nx)

x = np.empty([nx])
for i in range(nx):
    x[i] = random.random()
print(x)
X = np.fft.rfft(x)
print(X)
print(r2c_1d(x, nx))

print(np.fft.irfft(X,nx)) 
print(c2r_1d(X, nx))
print(x)

print()
print(nx, ny)

x = np.empty([nx, ny])

for i in range(nx):
    for j in range(ny):
        x[i][j] = random.random()
print(x)

X = np.fft.rfft2(x)

print(X)
if is_symmetric_2d(X, nx, ny):
    print("X is symmetric")
else:
    print("X isn't symmetric")

X0 = r2c_2d(x, nx, ny)
print(X0)
print(np.allclose(X, X0))

xx = np.fft.irfft2(X, [nx, ny])
print(xx)
print(np.allclose(x, xx))
xx0 = c2r_2d(X, nx, ny)
print(xx0)
print(np.allclose(xx, xx0))
print(np.isclose(xx, xx0))

print()

Z = np.empty([nx, nyp], dtype=complex)
for i in range(nx):
    for j in range(nyp):
        Z[i,j] = complex(random.random(), random.random())

print(Z)
Z = symmetrize_2d(Z, nx, ny)
if is_symmetric_2d(Z, nx, ny):
    print("Z is symmetric")
else:
    print("Z isn't symmetric")
print(Z)


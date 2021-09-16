#!/usr/bin/python3

import math
import cmath
import numpy as np
import random

nx = 4
ny = 4
nyp = ny // 2 + 1

def symmetrize_1d(hdata, nx):
    hdata[0] = hdata[0].real
    if nx % 2 == 0:
        nxp = nx // 2 + 1
        hdata[nxp - 1] = hdata[nxp - 1].real
    return hdata

def symmetrize_2d(hdata, nx, ny):
    nyp = ny // 2 + 1
    hdata[0][0] = hdata[0][0].real
    if ny % 2 == 0:
        hdata[0][nyp -1] = hdata[0][nyp -1].real
    if nx % 2 == 0:
        hdata[nx // 2][0] = hdata[nx // 2][0].real
        if ny % 2 == 0:
            hdata[nx // 2][nyp -1] = hdata[nx // 2][nyp - 1].real
    for i in range(1, nx // 2):
        hdata[nx - i][0] = hdata[i][0].conj()
    if ny % 2 == 0:
        for i in range(1, nx // 2):
            hdata[nx - i][nyp - 1] = hdata[i][nyp - 1].conj()
    return hdata

def is_symmetric_1d(hdata, nx):
    if not np.isclose(hdata[0].imag, 0):
        return False
    if nx % 2 == 0:
        nxp = nx // 2 + 1
        if not np.isclose(hdata[nxp - 1].imag, 0):
            return False
    return True

def is_symmetric_2d(hdata, nx, ny):
    if not np.isclose(hdata[0][0].imag, 0):
        return False
    nyp = ny // 2 + 1
    if ny % 2 ==0:
        if not np.isclose(hdata[0][nyp-1].imag, 0.0):
            return False
    for i in range(1, nx // 2):
        if not np.isclose(hdata[nx - i][0], hdata[i][0].conj()):
            return False
    if nx % 2 == 0:
        if not np.isclose(hdata[nx // 2][0].imag, 0.0):
            return False
        if ny % 2 ==0:
            if not np.isclose(hdata[nx // 2][nyp - 1].imag, 0.0):
                return False
        for i in range(1, nx // 2):
            if not np.isclose(hdata[nx - i][nyp - 1], hdata[i][nyp - 1].conj()):
                return False
    return True

def r2c_1d(rdata, nx, impose_hermitian=False):
    cdata = np.empty([nx], dtype=complex)
    for i in range(nx):
        cdata[i] = rdata[i]
    cdata = np.fft.fft(cdata)
    if impose_hermitian:
        cdata = symmetrize_1d(cdata, nx)
    return cdata[0:nx // 2 + 1]

def c2r_1d(hdata, nx, impose_hermitian=False):
    if impose_hermitian:
        symmetrize_1d(hdata, nx)
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

def r2c_2d_decomp(rdata, nx, ny, impose_hermitian=False):
    hdata = np.empty([nx, ny // 2 + 1], dtype=complex)
    for i in range(nx):
        hdata[i] = r2c_1d(rdata[i], ny, impose_hermitian)
    for j in range(ny // 2 + 1):
        hdata[:,j] = np.fft.fft(hdata[:,j])
    return hdata

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
            cdata[i][j] = hdata[nx - i][ny - j].conj();
            cdata[nx - i][j] = hdata[i][ny - j].conj();
    cdata = np.fft.ifft2(cdata)
    return cdata.real
    
def c2r_2d_decomp(hdata, nx, ny, impose_hermitian=False):
    cdata = np.empty([nx, ny // 2 + 1], dtype=complex)
    for j in range(ny // 2 + 1):
        cdata[:,j] = np.fft.ifft(hdata[:,j])
    rdata = np.empty([nx, ny])
    for i in range(nx):
        rdata[i] = c2r_1d(cdata[i], ny, impose_hermitian)
    return rdata



print()
print("checking our impose Hermitian code:")

print()
print("1D")
x = np.empty([nx])
for i in range(nx):
    x[i] = random.random()
X = np.fft.rfft(x)
print(X)
print(is_symmetric_1d(X, nx))
print(np.allclose(X,  symmetrize_1d(X, nx)))


print()
print("2D")

x = np.empty([nx, ny])
for i in range(nx):
    for j in range(ny):
        x[i][j] = random.random()
X = np.fft.rfft2(x)
print(X)
print(is_symmetric_2d(X, nx, ny))
print(np.allclose(X,  symmetrize_2d(X, nx, ny)))

        
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


print()
print("1D:", nx)

print()
print("direct")

x = np.empty([nx])
for i in range(nx):
    x[i] = random.random()
print(x)
X = np.fft.rfft(x)
print("np.fft.rfft:")
print(X)
print("embedded:")
X0 = r2c_1d(x, nx)
print(X0)
print(np.allclose(X, X0))
print("embedded with imposed symmetry:")
X00 = r2c_1d(x, nx, True)
print(X0)
print(np.allclose(X, X00))

print()
print("inverse")
print("np.fft.irfft")
xx = np.fft.irfft(X,nx)
print(xx)
print("embedded:")
xx0 = c2r_1d(X, nx)
print(xx0)
print(np.allclose(xx, xx0))
print("embedded with Hermitian imposed:")
xxx0 = c2r_1d(X, nx, True)
print(xxx0)
print(np.allclose(xx, xxx0))

print()
print("2D:", nx, ny)

print()
print("direct")

x = np.empty([nx, ny])

for i in range(nx):
    for j in range(ny):
        x[i][j] = random.random()

print("np.fft.rfft2:")
X = np.fft.rfft2(x)
print(X)
if is_symmetric_2d(X, nx, ny):
    print("X is symmetric")
else:
    print("X isn't symmetric")
print("2D embedded:")
X0 = r2c_2d(x, nx, ny)
print(X0)
print(np.allclose(X, X0))
print("1D embedded:")
X00 = r2c_2d_decomp(x, nx, ny)
print(X00)
print(np.allclose(X, X00))
print("1D embedded with 1D Hermitian imposed:")
X000 = r2c_2d_decomp(x, nx, ny, True)
print(X000)
print(np.allclose(X, X000))

print()
print("inverse")

print("np.fft.irfft2:")
xx = np.fft.irfft2(X, [nx, ny])
print(xx)
print(np.allclose(x, xx))
print("2D embedded:")
xx0 = c2r_2d(X, nx, ny)
print(xx0)
print(np.allclose(xx, xx0))
print("1D embedded:")
xx00 = c2r_2d_decomp(X, nx, ny)
print(xx00)
print(np.allclose(xx, xx00))
print("1D embedded with 1D Hermitian imposed:")
xx000 = c2r_2d_decomp(X, nx, ny, True)
print(xx000)
print(np.allclose(xx, xx000))

print()
print("1D inverse on malformed data:")
X = np.empty([nx], dtype=complex)
for i in range(nx):
    X[i] = complex(random.random(), random.random())
print("np.fft.irfft")
xx = np.fft.irfft(X,nx)
print(xx)
print("embedded:")
xx0 = c2r_1d(X, nx)
print(xx0)
print(np.allclose(xx, xx0))
print("embedded with Hermitian imposed:")
xxx0 = c2r_1d(X, nx, True)
print(xxx0)
print(np.allclose(xx, xxx0))
    
print()
print("2D inverse on malformed data:")

X = np.empty([nx, nyp], dtype=complex)
for i in range(nx):
    for j in range(nyp):
        X[i,j] = complex(random.random(), random.random())

print("np.fft.irfft2:")
xx = np.fft.irfft2(X, [nx, ny])
print(xx)
print("2D embedded:")
xx0 = c2r_2d(X, nx, ny)
print(xx0)
print(np.allclose(xx, xx0))
print("1D embedded:")
xx00 = c2r_2d_decomp(X, nx, ny)
print(xx00)
print(np.allclose(xx, xx00))
print("1D embedded with 1D Hermitian imposed:")
xx000 = c2r_2d_decomp(X, nx, ny, True)
print(xx000)
print(np.allclose(xx, xx000))

print(X)
X = symmetrize_2d(X, nx, ny)
print(X)
cdata = np.empty([nx, ny // 2 + 1], dtype=complex)
for j in range(ny // 2 + 1):
    cdata[:,j] = np.fft.ifft(X[:,j])
print(cdata)
rdata = np.empty([nx, ny])
for i in range(nx):
    rdata[i] = np.fft.irfft(cdata[i], ny)
print(rdata)

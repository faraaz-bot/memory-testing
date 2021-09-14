#!/usr/bin/python3

import math
import cmath
import numpy as np
import random

nx = 4
ny = 4
nyp = ny // 2 + 1

print(nx, ny)

z = np.empty([nx, nyp], dtype=complex)
for i in range(nx):
    for j in range(nyp):
        z[i,j] = complex(random.random(), random.random())

print(z)

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

z = symmetrize_2d(z, nx, ny)

if is_symmetric_2d(z, nx, ny):
    print("it's symmetric")
else:
    print("it's not symmetric")

print(z)


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


def r2c_1d(data, nx):
    zdata = np.empty([nx], dtype=complex)
    for i in range(nx):
        zdata[i] = data[i]
    print(data)

#!/usr/bin/python3

import sys

import math
import cmath
import numpy as np

from rckernels import postkernel, prekernel

N = 6
if len(sys.argv) == 2:
    N = int(sys.argv[1])

if N % 2 != 0:
    print(N, "is not even")
    sys.exit(1)

print("N:", N)
    
Nhalf = N // 2
Ncomplex = Nhalf + 1

import random

x = np.empty([N])
for i in range(0, N):
    x[i] = random.random()

print("input:", x)

npX = np.fft.rfft(x)
print("np rfft:", npX)

print("our rfft:")
z = np.empty([Nhalf], dtype=complex)
for i in range(0, Nhalf):
    z[i] = complex(x[2 * i], x[2 *i + 1])
#print(z)

Z = np.fft.fft(z)
#print(Z)

X = postkernel(Z)
print(X)

maxerr = 0.0
for i in range(0, Nhalf+1):
    diff = abs(X[i] - npX[i])
    if diff > maxerr:
        maxerr = diff
print("maxerr: " + str(maxerr))
print()

npx = np.fft.irfft(np.fft.rfft(x))*(len(x))
print("np irfft(rfft):", npx)

print("our irfft(rfft):", )
Z = prekernel(X)
#print(Z)
z = np.fft.ifft(Z) * len(Z)
#print(z)

for i in range(0, Nhalf):
    x[2 * i] = z[i].real
    x[2 *i + 1] = z[i].imag
print(x)

maxerr = 0.0
for i in range(0, N):
    diff = abs(x[i] - npx[i])
    if diff > maxerr:
        maxerr = diff
print("maxerr: " + str(maxerr))


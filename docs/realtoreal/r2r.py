#!/usr/bin/python3

import numpy as np
import numpy.random as nr
import scipy

N = 4
#x = nr.rand(N)
x = np.arange(N) + 1


print("DST-I")
X0 = scipy.fft.dst(x, type=1)
print(X0)


xtilde = np.empty(2 * N + 2)
xtilde[0] = 0
xtilde[N] = 0
for i in range(N):
    xtilde[i + 1] = x[i]
    xtilde[2 * N + 1 - i] = -x[i]
#print(xtilde)

X1 = -scipy.fft.fft(xtilde).imag[1:N+1]
print(X1)
print(np.max(np.abs(X0 - X1)))



print("DST-II")
X0 = scipy.fft.dst(x, type=2) / 2
print(X0)

X2 = np.empty(N)
for k in range(N):
    X2[k] = 0.0
    for n in range(N):
        X2[k] += x[n] * np.sin((np.pi / N) * (n + 0.5) * (k+1))
print(X2)
print(np.max(np.abs(X0 - X2)))


xtilde = np.zeros(4*N)
for n in range(N):
    xtilde[2*n] = 0.0
    xtilde[4*N -2 *n -2] = 0.0
    xtilde[2*n + 1] = x[n]
    xtilde[4*N -(2 *n + 1)] = -x[n]
#print(xtilde)
X1 = -np.fft.fft(xtilde).imag[1:N+1] / 2
print(X1)
print(np.max(np.abs(X0 - X1)))

print("DCT-I")

X0 = scipy.fft.dct(x, type=1) / 2
print(X0)

xtilde = np.zeros(4 * N - 4)
xtilde[0] = x[0]
xtilde[N-1] = x[N-1]
for n in range(1, N-1):
    xtilde[n] = x[n]
    xtilde[2 * N - 2 - n] = x[n]
xtilde[2*N - 2] = x[0]
xtilde[3*N - 3] = x[N-1]
for n in range(1, N-1):
    xtilde[2 * N - 2 + n] = x[n]
    xtilde[4 * N - 4 - n] = x[n]
X1 = np.fft.fft(xtilde) / 4
#print(X1.real[:2*N:2])
print(np.max(np.abs(X1.real[:2*N:2] - X0)))

X2 = np.empty(N)
for k in range(N):
    X2[k] = 0.5 * (x[0] + (-1)**k * x[N-1])
    for n in range(1, N-1):
        X2[k] += x[n] * np.cos((np.pi / (N-1)) * n * k)
print(X2)
print(np.max(np.abs(X0 - X2)))



print("DCT-II")

X0 = scipy.fft.dct(x, type=2) / 2
print(X0)

X2 = np.empty(N)
for k in range(N):
    X2[k] = 0.0
    for n in range(N):
        X2[k] += x[n] * np.cos((np.pi / N) * (n + 0.5) * k)
    
print(X2)
print(np.max(np.abs(X0 - X2)))

xtilde = np.zeros(4*N)
for n in range(N):
    xtilde[2*n] = 0.0
    xtilde[4*N -2 *n -2] = 0.0
    xtilde[2*n + 1] = x[n]
    xtilde[4*N -(2 *n + 1)] = x[n]
#print(xtilde)
X1 = np.fft.fft(xtilde).real[0:N] / 2
print(X1)
print(np.max(np.abs(X0 - X1)))


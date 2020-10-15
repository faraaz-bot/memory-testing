#!/usr/bin/python3

import numpy as np
import numpy.random as nr
import scipy

N = 4
#x = nr.rand(N)
x = np.arange(N) + 1
print(x)

print("\nDCT-I")

X0 = scipy.fft.dct(x, type=1) / 2
print(X0)

print("padded:")
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
X1 = np.fft.rfft(xtilde).real[:2*N:2] / 4
print(X1)
print(np.max(np.abs(X1 - X0)))

print("direct:")
X2 = np.empty(N)
for k in range(N):
    X2[k] = 0.5 * (x[0] + (-1)**k * x[N-1])
    for n in range(1, N-1):
        X2[k] += x[n] * np.cos((np.pi / (N-1)) * n * k)
print(X2)
print(np.max(np.abs(X0 - X2)))


print("\nDST-I")
X0 = scipy.fft.dst(x, type=1) / 2
print(X0)


print("direct:")
X2 = np.empty(N)
for k in range(N):
    X2[k] = 0
    fact = np.pi / (N + 1)
    for n in range(N):
        X2[k] += x[n] * np.sin(fact * (n+1) * (k+1))
print(X2)
print(np.max(np.abs(X0 - X2)))


print("padded:")
xtilde = np.empty(2 * N + 2)
xtilde[0] = 0
xtilde[N] = 0
for n in range(N):
    xtilde[n + 1] = x[n]
    xtilde[2 * N + 1 - n] = -x[n]
#print(xtilde)

X1 = -scipy.fft.fft(xtilde).imag[1:N+1] / 2
print(X1)
print(np.max(np.abs(X0 - X1)))




print("\nDCT-II")

X0 = scipy.fft.dct(x, type=2) / 2
print(X0)

print("direct:")
X2 = np.empty(N)
for k in range(N):
    X2[k] = 0
    fact = np.pi / N
    for n in range(N):
        X2[k] += x[n] * np.cos(fact * (n + 0.5) * k)
print(X2)
print(np.max(np.abs(X0 - X2)))

print("padded:")
xtilde = np.empty(4*N)
for n in range(N):
    xtilde[2*n] = 0.0
    xtilde[4*N -2 *n -2] = 0.0
    xtilde[2*n + 1] = x[n]
    xtilde[4*N -(2 *n + 1)] = x[n]
#print(xtilde)
X1 = np.fft.fft(xtilde).real[0:N] / 2
print(X1)
print(np.max(np.abs(X0 - X1)))

print("Shao and Johnson, old way:")
# Shao and Johnson 2009.
ytilde = np.empty(N)
for n in range(N // 2):
    ytilde[n] = x[2 * n]
    ytilde[N - 1 -n] = x[2 * n + 1]
Z = np.fft.fft(ytilde)
X3 = np.empty(N)
X3[0] = Z[0].real
for k in range(1, N // 2):
    omega = np.exp(-2j * np.pi * k / (4 * N))
    X3[k] = (omega * Z[k]).real
    X3[N-k] = -(omega * Z[k]).imag
X3[N // 2] = Z[N // 2].real / np.sqrt(2)
print(X3)
print(np.max(np.abs(X0 - X3)))

print("\nDST-II")
X0 = scipy.fft.dst(x, type=2) / 2
print(X0)

print("direct:")
X2 = np.empty(N)
for k in range(N):
    X2[k] = 0.0
    for n in range(N):
        X2[k] += x[n] * np.sin((np.pi / N) * (n + 0.5) * (k+1))
print(X2)
print(np.max(np.abs(X0 - X2)))

print("padded:")
xtilde = np.zeros(4*N)
for n in range(N):
    xtilde[2*n] = 0.0
    xtilde[4*N -2 *n -2] = 0.0
    xtilde[2*n + 1] = x[n]
    xtilde[4*N -(2 *n + 1)] = -x[n]
#print(xtilde)
X1 = -np.fft.rfft(xtilde).imag[1:N+1] / 2
print(X1)
print(np.max(np.abs(X0 - X1)))


print("Shao and Johnson, old way:")
# Shao and Johnson 2009.
ytilde = np.empty(N)
for n in range(N // 2):
    ytilde[n] = x[2 * n]
    ytilde[N - 1 -n] = -x[2 * n + 1]
Z = np.fft.fft(ytilde)
X3 = np.empty(N)
X3[0] = Z[0].real
for k in range(1, N // 2):
    omega = np.exp(-2j * np.pi * k / (4 * N))
    X3[k] = (omega * Z[k]).real
    X3[N-k] = -(omega * Z[k]).imag
X3[N // 2] = Z[N // 2].real / np.sqrt(2)
X3 = np.flip(X3)
print(X3)
print(np.max(np.abs(X0 - X3)))





print("\nDCT-III")

X0 = scipy.fft.dct(x, type=3) / 2
print(X0)

# Direct:
print("Direct:")
X1 = np.empty(N)
for k in range(N):
    X1[k] = 0.5 * x[0]
    fact = np.pi / N
    for n in range(1, N):
        X1[k] += x[n] * np.cos(fact * n * (k + 0.5))
print(X1)
print(np.max(np.abs(X0 - X1)))


print("Padded:")
xtilde = np.empty(4 * N)
for n in range(N):
    xtilde[n] = x[n]
xtilde[N] = 0
for n in range(N):
    xtilde[N + 1 + n] = -x[N - n - 1]
for n in range(1, N):
    xtilde[2 * N + n] = -x[n]
xtilde[3 * N] = 0
for n in range(1, N):
    xtilde[3* N + n] = x[N - n]
#print(xtilde)
X2 = 0.25 * scipy.fft(xtilde).real[1:2*N:2]
print(X2)
print(np.max(np.abs(X0 - X2)))



print("\nDST-III")

X0 = scipy.fft.dst(x, type=3)  / 2
print(X0)

# Direct:
print("Direct:")
X1 = np.empty(N)
for k in range(N):
    X1[k] = x[N - 1] * 0.5 * (-1)**k
    fact = np.pi / N
    for n in range(N - 1):
        X1[k] += x[n] * np.sin( fact * (n + 1.0) * (k + 0.5) )
print(X1)
print(np.max(np.abs(X0 - X1)))


print("padded:")
xtilde = np.empty(4 * N)
xtilde[0] = 0
for n in range(N - 1):
    xtilde[n + 1] = x[n]
    xtilde[N + n + 1] = x[N - n - 2]
    xtilde[2 * N + n + 1] = -x[n]
    xtilde[3 * N + n + 1] = -x[N - n - 2]
xtilde[N] = x[N - 1]
xtilde[2 * N] = 0
xtilde[3 * N] = -x[N - 1]
#print(xtilde)
X2 = -scipy.fft.rfft(xtilde)[1:2*N+1:2].imag * 0.25
print(X2)
print(np.max(np.abs(X0 - X2)))


print("\nDCT-IV")
X0 = scipy.fft.dct(x, type=4)  / 2
print(X0)

print("Direct:")
X1 = np.empty(N)
for k in range(N):
    fact = np.pi / N
    X1[k] = 0
    for n in range(N):
        X1[k] += x[n] * np.cos( fact * (n + 0.5) * (k + 0.5) )
print(X1)
print(np.max(np.abs(X0 - X1)))

print("Padded:")
xtilde = np.zeros(8 * N)
for n in range(N):
    xtilde[2 * n + 1] = x[n]
    xtilde[4 * N - (2 *n + 1)] = -x[n]
    xtilde[4 * N + 2 * n + 1] = -x[n]
    xtilde[8 * N - (2 *n + 1)] = x[n]
X2 = 0.25 * scipy.fft.rfft(xtilde).real[1:2*N:2]
print(X2)
print(np.max(np.abs(X0 - X2)))



print("\nDST-IV")

X0 = scipy.fft.dst(x, type=4) / 2
print(X0)

# Direct:
print("Direct:")
X1 = np.empty(N)
for k in range(N):
    X1[k] = 0
    fact = np.pi / N
    for n in range(N):
        X1[k] += x[n] * np.sin(fact * (n + 0.5) * (k + 0.5))
print(X1)
print(np.max(np.abs(X0 - X1)))

print("padded:")
xtilde = np.zeros(8 * N)
for n in range(N):
    xtilde[2 * n + 1] = x[n]
    xtilde[4 * N - (2 *n + 1)] = x[n]
    xtilde[4 * N + 2 * n + 1] = -x[n]
    xtilde[8 * N - (2 *n + 1)] = -x[n]
#print(xtilde)
X2 = -0.25 * scipy.fft.rfft(xtilde).imag[1:2*N:2]
print(X2)
print(np.max(np.abs(X0 - X2)))

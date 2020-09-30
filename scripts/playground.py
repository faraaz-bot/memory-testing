#!/usr/bin/python3
#
# Matt's exploratory FFTs.
#

import numpy as np
import numpy.linalg as la
import numpy.random as nr

def roots(N, k):
    a = -2j*np.pi/N * k
    return np.exp(a * np.arange(N))

# Naive DFT
def naive(x):
    N = x.size
    X = np.zeros(N, np.complex64)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j*np.pi/N * n * k)
    return X

# Naive DFT using matrix mult
def naive_mmul(x):
    N = x.size
    A = np.zeros((N, N), np.complex64)
    for k in range(N):
        A[k, :] = roots(N, k)
    return np.dot(A, x)

# Naive DFT using vector mult
def naive_rmul(x):
    N = x.size
    X = np.zeros(N, np.complex64)
    for k in range(N):
        X[k] = np.dot(x, roots(N, k))
    return X

# Recursive Cooley-Tukey, assuming length is a power of 2
def cooley_tukey_pow2(x):
    N = x.size
    if N == 1:
        return x

    N2 = int(N/2)
    E = cooley_tukey_pow2(x[0::2])
    O = cooley_tukey_pow2(x[1::2])

    X = np.zeros(N, np.complex64)
    for k in range(N2):
        r = np.exp(-2j*np.pi * k / N)
        X[k]    = E[k] + r * O[k]
        X[k+N2] = E[k] - r * O[k]

    return X

# The dreaded bit-reverse:
def bitreverse(x, Nbit):
    result = 0
    for i in range(Nbit):
        result <<= 1
        result |= x & 1
        x >>= 1
    return result

# Iterative Cooley-Tukey, assuming length is a power of 2, DIF
def icooley_tukey_pow2(x):
    # DIF: decimation in frequency, which refers to the fact that the
    # finest stage (ie neighbouring values) is done last.
    N = x.size
    halfN = int(N / 2)
    log2N = int(np.log(N) / np.log(2))
    
    # Length log(N) loop:
    for s in range(0, log2N):
        a = int(pow(2, s))
        b = int(N / a)
        halfb = int(b / 2)
        # Length N/2 loop:
        for l in range(halfb):
            for k in range(a):
                p = l + k * b
                q = p + halfb
                r = np.exp(-2j * np.pi * l / b)
                xp = x[p]
                xq = x[q]
                x[p] = xp + xq
                x[q] = r * (xp - xq)
                
    # bit-reverse:
    for p in range(N):
        q = bitreverse(p, log2N)
        if(p > q):
            x[p], x[q] = x[q], x[p]
        
    return x


# Cooley-Tukey, assuming length is a product N1 N2
def cooley_tukey_decomp(x, N1, N2):
    X = np.zeros(N1*N2, np.complex64)
    Y = np.zeros(N1*N2, np.complex64)

    dft = naive_rmul

    for n1 in range(N1):
        Y[n1::N1] = dft(x[n1::N1])  # strided Y write; strided x read

    if N1 == 4:
        # explicit butterfly with twiddles...
        for k2 in range(N2):
            i1, i2 = 4*k2, 4*(k2+1)
            T = np.exp(-2j*np.pi*k2/4/N2 * np.arange(4)) * Y[i1:i2]
            X[k2     ] = T[0] +      T[1] + T[2] +      T[3]
            X[k2+  N2] = T[0] - 1j * T[1] - T[2] + 1j * T[3]
            X[k2+2*N2] = T[0] -      T[1] + T[2] -      T[3]
            X[k2+3*N2] = T[0] + 1j * T[1] - T[2] - 1j * T[3]
    else:
        for k2 in range(N2):
            i1, i2 = k2*N1, (k2+1)*N1
            t = np.exp(-2j*np.pi*k2/N1/N2 * np.arange(N1))
            Y[i1:i2] = t * Y[i1:i2]     # contiguous Y read/write
            X[k2::N2] = dft(Y[i1:i2])   # strided X write; contiguous Y read

    return X


def compare(y, f1, f2):
    k1, k2 = f1(y), f2(y)
    return la.norm(k1-k2) / la.norm(k1)

N = 64
y = nr.rand(N) + 1j * nr.rand(N)
print('rmul', compare(y, naive, naive_rmul))
print('mmul', compare(y, naive, naive_mmul))
print('pow2', compare(y, naive, cooley_tukey_pow2))
print('dec8', compare(y, naive, lambda y: cooley_tukey_decomp(y, 4, 16)))
print('ipow2', compare(y, naive, icooley_tukey_pow2))

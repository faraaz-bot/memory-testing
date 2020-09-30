#!/usr/bin/python3


import numpy as np
import numpy.random as nr

# The dreaded bit-reverse:
def bitreverse(x, Nbit):
    result = 0
    for i in range(Nbit):
        result <<= 1
        result |= x & 1
        x >>= 1
    return result

# Iterative Cooley-Tukey, assuming length is a power of 2, DIT
def icooley_tukey_pow2_DIT(x):
    # DIF: decimation in frequency, which refers to the fact that the
    # finest stage (ie neighbouring values) is done last.
    N = x.size
    halfN = N // 2
    log2N = int(np.log(N) / np.log(2))

    X = np.copy(x)
    
    # Length log(N) loop:
    for s in range(0, log2N):
        Nb = int(pow(2, s)) # Number of butterflies
        lb = N // (2 * Nb) # Length of butterfly
        print(Nb, lb)
        # Length N/2 loop:
        for n in range(lb):
            for b in range(Nb):
                p = n + b * 2* lb
                q = p + lb
                print("\t\t", p, q)
                r = np.exp(-2j * np.pi * n / (2 * lb))
                print("\t", r)
                Xp = X[p]
                Xq = X[q]
                X[p] = Xp + Xq
                X[q] = r * (Xp - Xq)
                
    # bit-reverse:
    for p in range(N):
        q = bitreverse(p, log2N)
        if(p > q):
            X[p], X[q] = X[q], X[p]
        
    return X



# Iterative Cooley-Tukey, assuming length is a power of 2, DIF
def icooley_tukey_pow2_DIF(x):
    # DIF: decimation in frequency, which refers to the fact that the
    # finest stage (ie neighbouring values) is done last.
    N = x.size
    halfN = N // 2
    log2N = int(np.log(N) / np.log(2))

    X = np.copy(x)
    
    # bit-reverse:
    for p in range(N):
        q = bitreverse(p, log2N)
        if(p > q):
            X[p], X[q] = X[q], X[p]

    # Length log(N) loop:
    for s in range(log2N):
        Nb = N // int(pow(2, s + 1)) # Number of butterflies
        lb = N // (Nb * 2)           # Length of butterfly
        print(Nb, lb)
        for n in range(lb):
            r = np.exp(-2j * np.pi * n / (2*lb) )
            for b in range(Nb):
                p = b * lb * 2 + n
                q = p + lb
                print("\t", p, q)
                print("\t", r)
                Xp = X[p]
                Xq = X[q]
                X[p] = Xp + r * Xq
                X[q] = Xp - r * Xq
        
    return X


N = 8
print("Length:", N)

x = nr.rand(N) + 1j * nr.rand(N)

X0 = np.fft.fft(x)

X = icooley_tukey_pow2_DIT(x)

print("DIT L-inf error:", np.max(np.abs(X - X0)))

print(X0)

X = icooley_tukey_pow2_DIF(x)
print(X)

print("DIF L-inf error:", np.max(np.abs(X - X0)))


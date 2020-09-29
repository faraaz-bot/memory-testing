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
    
    # Length log(N) loop:
    for s in range(0, log2N):
        a = int(pow(2, s))
        b = N // a
        halfb = b // 2
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



# Iterative Cooley-Tukey, assuming length is a power of 2, DIF
def icooley_tukey_pow2_DIF(x):
    # DIF: decimation in frequency, which refers to the fact that the
    # finest stage (ie neighbouring values) is done last.
    N = x.size
    halfN = N // 2
    log2N = int(np.log(N) / np.log(2))
    
    # bit-reverse:
    for p in range(N):
        q = bitreverse(p, log2N)
        if(p > q):
            x[p], x[q] = x[q], x[p]

    # Length log(N) loop:
    for s in range(1, log2N + 1):
        m = int(pow(2, s))
        a = N // m
        b = N // a
        print(a, b)
        for k in range(a):
            for l in range(b // 2):
                p = b * k + l
                q = p + b // 2
                print("\t\t", p, q)
                r = np.exp(-2j * np.pi * k / a)
                xp = x[p]
                xq = x[q]
                x[p] = xp + xq
                x[q] = r * (xp - xq)
        
    return x


N = 4

x = nr.rand(N) + 1j * nr.rand(N)

X0 = np.fft.fft(x)

X = icooley_tukey_pow2_DIT(x)

print("DIT L-inf error:", np.max(np.abs(X - X0)))


X = icooley_tukey_pow2_DIF(x)
print(X)
print(X0)
print("DIF L-inf error:", np.max(np.abs(X - X0)))


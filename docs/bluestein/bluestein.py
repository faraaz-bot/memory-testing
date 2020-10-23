#!/usr/bin/python3

import math
import numpy as np
import numpy.linalg as la
import numpy.random as nr
from numpy.fft import fft, ifft

use_mp = False

if use_mp:
    from mpmath import mp
    mp.dps = 75

def bluestein(x):
    N = x.size
    M = 2**int(math.ceil(math.log2(N))+1)
    assert(2*N <= M)

    n = np.arange(N)

    # A = x * exp() term; A is indexed by n
    if use_mp:
        a = np.zeros(N, x.dtype)
        for k in range(N):
            a[k] = complex(mp.exp(-1j * mp.pi / N * k**2))
    else:
        a = np.exp(-1j * np.pi / N * n**2)
    A = np.zeros(M, x.dtype)
    A[:N] = x * a

    # B = exp() term; B is indexed by k-n
    if use_mp:
        b = np.zeros(N, x.dtype)
        for k in range(N):
            b[k] = complex(mp.exp(1j * mp.pi / N * k**2))
    else:
        b = np.exp(1j * np.pi / N * n**2)
    B = np.zeros(M, x.dtype)
    B[0] = b[0]
    for i in range(1, N):
        B[i] = b[i]
        B[M-i] = b[i]

    return a * ifft(fft(A) * fft(B))[:N]


def compare(y, f1, f2):
    y1, y2 = y.copy(), y.copy()
    k1, k2 = f1(y1), f2(y2)
    return la.norm(k1-k2) / la.norm(k1)

N = 101
y = nr.rand(N) + 1j * nr.rand(N)

print('rel error', compare(y, fft, bluestein))

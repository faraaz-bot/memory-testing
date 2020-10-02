#!/usr/bin/python3

import numpy as np
import numpy.random as nr
import scipy

N = 4
x = nr.rand(N)


X0 = scipy.fft.dst(x, type=1)
print(X0)


xtilde = np.empty(2 * N + 2)
xtilde[0] = 0
xtilde[N] = 0
for i in range(N):
    xtilde[i + 1] = x[i]
    xtilde[2 * N + 1 - i] = -x[i]
#print(xtilde)

X1 = scipy.fft.fft(xtilde)
print(-X1.imag[1:N+1])

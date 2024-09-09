#!/usr/bin/env python3

import numpy as np
import cmath

np.set_printoptions(suppress=True, linewidth=np.nan)

import sys
sys.path.insert(1, '../../real/code')
import rckernels

import matplotlib.pyplot as plt


# 1D transform
length = [1024]
nbatch = 1

freq = 10

# Allocate and initialize the data:
x = np.zeros(shape=np.append(nbatch, length), dtype=float)
init = lambda ibatch, ix : 0.2 * ix / length[-1] + np.sin(freq * 2 * np.pi * ix / length[-1])
for ibatch in range(len(x)):
    for ix in range(len(x[ibatch])):
        x[ibatch, ix] = init(ibatch, ix)

#print(x)

#plt.plot(x[0])
#plt.show()

# Hann filter
readop = lambda val, ibatch, idx: val * 0.5 * (1 - np.cos(2* np.pi * idx[0] / length[-1]))
# Lots of other filters are possible: https://en.wikipedia.org/wiki/Window_function
# https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf


#readop = None
writeop = None

# Perform the pointwise read op, perform the batched 1D
X = rckernels.rcfft(x, length, nbatch, readop, writeop)

#print(X)

#plt.plot(np.abs(X[0]))
#plt.show()


threshold = 0.5 * max(abs(X[0]))
mask = abs(X[0]) > threshold
peaks = X[0][mask]
freqs = []
for ibatch in range(len(X)):
    for ix in range(len(X[ibatch])):
        if np.abs(X[ibatch][ix]) > threshold:
          freqs.append([ix, X[ibatch][ix]])


print("dominant frequencies:", freqs)

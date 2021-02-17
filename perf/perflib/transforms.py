
import hipfft
import numpy as np
import numpy.linalg as la
import numpy.random as nr

from dataclasses import dataclass
from numpy.fft import *
from typing import List, Any

import perflib.utils

def GiB(y):
    return y.nbytes/1024**3


def real_input(n, nbatch, dtype):
    y = np.zeros((nbatch, n), dtype)
    for i in range(nbatch):
        y[i] = nr.rand(n)
    return y


def complex_input(n, nbatch, dtype):
    return real_input(n, nbatch, dtype) + 1j * real_input(n, nbatch, dtype)


def enforce_hermitian(y):
    n = y.shape[1]
    y = y.copy()
    y[:,0]    = y[:,0].real
    y[:,n//2] = y[:,n//2].real
    y[:,-1]   = y[:,-1].real
    return y


def compare(k1, k2):
    reldiff = la.norm(k1-k2) / la.norm(k2)

    tolerance = {
        np.dtype(np.float32):    7.5e-7,
        np.dtype(np.complex64):  7.5e-7,
        np.dtype(np.float64):    1.0e-11,
        np.dtype(np.complex128): 1.0e-11
        }[k1.dtype]

    stolerance = tolerance * np.sqrt(np.log2(k2.size))
    if reldiff > stolerance:
        print('REL DIFF is: ', reldiff, tolerance, stolerance)
        raise ValueError

    return reldiff


def complex_forward(n, ntrials, nbatch, dtype, verify):
    results = []
    for trial in range(ntrials):
        y = complex_input(n, nbatch, dtype)
        z, t = hipfft.forward(y, time=True, batched=True)
        if verify:
            r = fftn(y, axes=[1])
            compare(z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': GiB(y)})
    return results


def complex_backward(n, ntrials, nbatch, dtype, verify):
    results = []
    for trial in range(ntrials):
        y = complex_input(n, nbatch, dtype)
        z, t = hipfft.backward(y, time=True, batched=True)
        if verify:
            r = ifftn(y, axes=[1])
            s = np.asarray(1.0/z.shape[1], dtype)
            compare(s*z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': GiB(y) })
    return results


def real_forward(n, ntrials, nbatch, dtype, verify):
    results = []
    for trial in range(ntrials):
        y = real_input(n, nbatch, dtype)
        z, t = hipfft.forward(y, real=True, time=True, batched=True)
        if verify:
            r = rfftn(y, axes=[1])
            compare(z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': GiB(y) })
    return results


def real_backward(n, ntrials, nbatch, dtype, verify):
    if n % 2 > 0:
        return []
    results = []
    for trial in range(ntrials):
        y = enforce_hermitian(complex_input(n, nbatch, dtype))
        z, t = hipfft.backward(y, real=True, time=True, batched=True)
        if verify:
            r = irfftn(y, axes=[1])
            s = np.asarray(1.0/z.shape[1], dtype)
            compare(s*z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': GiB(y) })
    return results


@dataclass
class FFTRunner:
    label: str
    transform: Any
    lengths: List[int]
    ntrials: int
    nbatch: int
    dtype: Any
    verify: bool

    def run(self):
        self.results = [ self.transform(x, self.ntrials, self.nbatch, self.dtype, self.verify) for x in self.lengths ]

    def write(self, fname):
        results = sum(self.results, [])
        for length in self.lengths:
            # XXX assumes 1d...
            seconds = [ t['time'] for t in results if t['n'] == length ]
            perflib.utils.write_dat(fname, [length], self.nbatch, seconds)

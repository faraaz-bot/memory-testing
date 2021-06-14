
try:
    import hipfft
except ImportError as e:
    pass

import numpy as np
import numpy.linalg as la
import numpy.random as nr

from typing import List, Any
from dataclasses import dataclass

from numpy.fft import *

import perflib.utils


def shape(n, nbatch):
    if isinstance(n, list) or isinstance(n, tuple):
        return [nbatch] + list(n)
    return [nbatch, n]


def product(x):
    p = 1
    for f in x:
        p *= f
    return p


def real_input(n, nbatch, dtype):
    s = shape(n, nbatch)
    y = np.zeros(s, dtype)
    for i in range(nbatch):
        y[i] = nr.rand(*s[1:])
    return y


def complex_input(n, nbatch, dtype):
    return real_input(n, nbatch, dtype) + 1j * real_input(n, nbatch, dtype)


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
            r = fftn(y, s=y.shape[1:])
            compare(z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes})
    return results


def complex_backward(n, ntrials, nbatch, dtype, verify):
    results = []
    for trial in range(ntrials):
        y = complex_input(n, nbatch, dtype)
        z, t = hipfft.backward(y, time=True, batched=True)
        if verify:
            r = ifftn(y, s=y.shape[1:])
            s = np.asarray(1.0/product(z.shape[1:]), dtype)
            compare(s*z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes })
    return results


def real_forward(n, ntrials, nbatch, dtype, verify):
    results = []
    for trial in range(ntrials):
        y = real_input(n, nbatch, dtype)
        z, t = hipfft.forward(y, real=True, time=True, batched=True)
        if verify:
            r = rfftn(y, s=y.shape[1:])
            compare(z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes })
    return results


def real_backward(n, ntrials, nbatch, dtype, verify):
    results = []
    rshape = shape(n, nbatch)
    if rshape[-1] % 2:
        return []
    for trial in range(ntrials):
        y = hipfft.forward(real_input(n, nbatch, dtype), real=True, batched=True)
        z, t = hipfft.backward(y, real=True, time=True, batched=True)
        if verify:
            r = irfftn(y, s=rshape[1:])
            s = np.asarray(1.0/product(rshape[1:]), dtype)
            compare(s*z, r)
        results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes })
    return results


@dataclass
class HIPFFTTestRunner:
    label: str = 'UNKNOWN'
    transform: Any = None
    lengths: List[Any] = list
    ntrials: int = 1
    nbatch: int = 1
    dtype: Any = None
    verify: bool = False

    def run(self):
        timings = []
        for length in self.lengths:
            timings.extend(
                self.transform.transform(
                    length, self.ntrials, self.nbatch, self.dtype, self.verify))
        return timings

    def write(self, dname, fname, results, title=None):
        for length in self.lengths:
            seconds = [ t['time'] for t in results if t['n'] == length ]
            perflib.utils.write_dat(dname / fname, length, self.nbatch, seconds, title=title)

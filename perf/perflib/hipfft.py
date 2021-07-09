
import hipfft
import numpy as np

from typing import List, Any
from dataclasses import dataclass

from numpy.fft import fftn, ifftn, rfftn, irfftn

import perflib.utils


@dataclass
class HIPFFTTestRunner:
    label: str = 'UNKNOWN'
    transform: Any = None
    lengths: List[Any] = list
    ntrials: int = 1
    nbatch: int = 1
    dtype: Any = None
    verify: bool = False
    placement: Any = None

    def is_inplace(self):
        if self.placement is None:
            return False
        return self.placement.label == 'inplace'


    def complex_forward(self, n):
        results = []
        for trial in range(self.ntrials):
            y = perflib.utils.complex_input(n, self.nbatch, self.dtype)
            z, t = hipfft.forward(y, time=True, batched=True, inplace=self.is_inplace())
            if self.verify:
                r = fftn(y, s=y.shape[1:])
                perflib.utils.compare(z, r)
            results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes})
        return results


    def complex_backward(self, n):
        results = []
        for trial in range(self.ntrials):
            y = perflib.utils.complex_input(n, self.nbatch, self.dtype)
            z, t = hipfft.backward(y, time=True, batched=True, inplace=self.is_inplace())
            if self.verify:
                r = ifftn(y, s=y.shape[1:])
                s = np.asarray(1.0 / perflib.utils.product(z.shape[1:]), self.dtype)
                perflib.utils.compare(s * z, r)
            results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes})
        return results


    def real_forward(self, n):
        results = []
        for trial in range(self.ntrials):
            y = perflib.utils.real_input(n, self.nbatch, self.dtype)
            z, t = hipfft.forward(y, real=True, time=True, batched=True, inplace=self.is_inplace())
            if self.verify:
                r = rfftn(y, s=y.shape[1:])
                perflib.utils.compare(z, r)
            results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes})
        return results


    def real_backward(self, n):
        results = []
        rshape = perflib.utils.shape(n, self.nbatch)
        if rshape[-1] % 2:
            return []
        for trial in range(self.ntrials):
            y = hipfft.forward(perflib.utils.real_input(n, self.nbatch, self.dtype), real=True, batched=True)
            z, t = hipfft.backward(y, real=True, time=True, batched=True, inplace=self.is_inplace())
            if self.verify:
                r = irfftn(y, s=rshape[1:])
                s = np.asarray(1.0 / perflib.utils.product(rshape[1:]), self.dtype)
                perflib.utils.compare(s * z, r)
            results.append({'n': n, 'method': 'hipfft', 'time': t, 'size': y.nbytes})
        return results


    def run(self):
        trns = getattr(self, self.transform.label)
        return perflib.utils.flatten([trns(length) for length in self.lengths])


    def write(self, dname, fname, results, title=None):
        for length in self.lengths:
            seconds = [t['time'] for t in results if t['n'] == length]
            perflib.utils.write_dat(dname / fname, length, self.nbatch, seconds, title=title)

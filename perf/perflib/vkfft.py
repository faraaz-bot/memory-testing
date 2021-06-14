

import numpy as np
import pyvkfft.opencl as vk
import pyopencl as cl
import pyopencl.array as cla

import logging
import re
import subprocess
import timeit
import math

import perflib.utils

from typing import List, Any
from dataclasses import dataclass

from perflib.utils import complex_input, real_input

from numpy.fft import fftn, ifftn, rfftn, irfftn



@dataclass
class VKFFTTestRunner:
    label: str = 'UNKNOWN'
    transform: Any = None
    lengths: List[Any] = list
    ntrials: int = 1
    nbatch: int = 1
    dtype: Any = None
    verify: bool = False

    def tic(self):
        self.queue.finish()
        t, _ = self.device.device_and_host_timer()
        return t


    def _vk_trans(self, f, y, real_forward=False):

        x = cla.to_device(self.queue, y)

        if real_forward:
            s = list(y.shape)
            s[-1] = s[-1] // 2 + 1
            b = cla.zeros(self.queue, tuple(s), y.dtype)

            start = self.tic()
            f(x, b)
            end = self.tic()

            z = b.get(self.queue)
        else:
            start = self.tic()
            f(x)
            end = self.tic()

            z = x.get(self.queue)

        return z, (end - start) * 1e-6


    def complex_forward(self, n):
        results = []
        y = perflib.utils.complex_input(n, self.nbatch, self.dtype)
        app = vk.VkFFTApp(y.shape, y.dtype, self.queue, inplace=True, ndim=len(y.shape)-1)
        for trial in range(self.ntrials):
            y = perflib.utils.complex_input(n, self.nbatch, self.dtype)
            z, t = self._vk_trans(app.fft, y)
            if self.verify:
                r = fftn(y, s=y.shape[1:])
                perflib.utils.compare(z, r)
            results.append({'n': n, 'method': 'vkfft', 'time': t, 'size': y.nbytes})
        return results


    def complex_backward(self, n):
        results = []
        y = perflib.utils.complex_input(n, self.nbatch, self.dtype)
        app = vk.VkFFTApp(y.shape, y.dtype, self.queue, inplace=True, ndim=len(y.shape)-1)
        for trial in range(self.ntrials):
            y = perflib.utils.complex_input(n, self.nbatch, self.dtype)
            z, t = self._vk_trans(app.ifft, y)
            if self.verify:
                r = ifftn(y, s=y.shape[1:])
                perflib.utils.compare(z, r)
            results.append({'n': n, 'method': 'vkfft', 'time': t, 'size': y.nbytes})
        return results


    def real_forward(self, n):
        return []


    def real_backward(self, n):
        return []


    def run(self, progress=True):

        platforms = cl.get_platforms()
        device = platforms[0].get_devices()[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        self.device, self.queue = device, queue

        trns = getattr(self, self.transform.label)
        return perflib.utils.flatten([trns(length) for length in self.lengths])


    def write(self, dname, fname, results, title=None):
        for length in self.lengths:
            seconds = [t['time'] for t in results if t['n'] == length]
            perflib.utils.write_dat(dname / fname, length, self.nbatch, seconds, title=title)

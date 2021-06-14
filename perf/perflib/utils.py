"""A few small utilities."""

import numpy as np
import numpy.linalg as la
import numpy.random as nr

from dataclasses import dataclass
from pathlib import Path
from typing import List

from functools import reduce


#
# Join shortcuts
#

def join(sep, s):
    """Return 's' joined with 'sep'.  Coerces to str."""
    return sep.join(str(x) for x in list(s))


def sjoin(s):
    """Return 's' joined with spaces."""
    return join(' ', s)


def njoin(s):
    """Return 's' joined with newlines."""
    return join('\n', s)


def cjoin(s):
    """Return 's' joined with commas."""
    return join(',', s)


def tjoin(s):
    """Return 's' joined with tabs."""
    return join('\t', s)


#
# Misc
#

def shape(n, nbatch):
    """Return NumPy shape."""
    if isinstance(n, (list, tuple)):
        return [nbatch] + list(n)
    return [nbatch, n]


def product(xs):
    """Return product of factors."""
    return reduce(lambda x, y: x * y, xs, 1)


#
# DAT files
#

@dataclass
class Sample:
    """Dyna-rider/rider timing sample: list of times for a given length+batch."""

    lengths: List[int]
    nbatch: int
    times: List[float]


def write_dat(fname, length, nbatch, seconds, title=None):
    """Append record to dyna-rider/rider .dat file."""
    path = Path(fname)
    if not path.exists():
        dat = ['# title: ' + title]
    else:
        dat = path.read_text().splitlines()
    if isinstance(length, int):
        length = [length]
    record = [len(length)] + list(length) + [nbatch, len(seconds)] + seconds
    dat.append(tjoin(record))
    path.write_text(njoin(dat) + '\n')


def read_dat(fname):
    """Read dyna-rider/rider .dat file."""
    dat = Path(fname).read_text()
    records = {}
    for line in dat.splitlines():
        if line.startswith('#'):
            continue
        words   = line.split("\t")
        dim     = int(words[0])
        lengths = tuple(map(int, words[1:dim + 1]))
        nbatch  = int(words[dim + 1])
        times   = list(map(float, words[dim + 3:]))
        records[lengths] = Sample(list(lengths), nbatch, times)
    return records


#
# FFT input and comparison
#

def real_input(n, nbatch, dtype):
    """Return random real inputs."""
    s = shape(n, nbatch)
    y = np.zeros(s, dtype)
    for i in range(nbatch):
        y[i] = nr.rand(*s[1:])
    return y


def complex_input(n, nbatch, dtype):
    """Return random complex inputs."""
    return real_input(n, nbatch, dtype) + 1j * real_input(n, nbatch, dtype)


def compare(k1, k2):
    """Compare arrays."""
    reldiff = la.norm(k1 - k2) / la.norm(k2)
    tolerance = {
        np.dtype(np.float32):    7.5e-7,
        np.dtype(np.complex64):  7.5e-7,
        np.dtype(np.float64):    1.0e-11,
        np.dtype(np.complex128): 1.0e-11
    }[k1.dtype]
    stolerance = tolerance * np.sqrt(np.log2(k2.size))
    if reldiff > stolerance:
        raise ValueError(f"Relative difference {reldiff} is too large ({stolerance}).")
    return reldiff

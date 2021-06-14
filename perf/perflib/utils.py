"""A few small utilities."""

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

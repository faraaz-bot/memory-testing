
from pathlib import Path as path


def sjoin(s):
    return ' '.join(list(s))


def njoin(s):
    return '\n'.join(list(s))


def cjoin(s):
    return ','.join(list((map(str, s))))


def write_dat(fname, length, nbatch, seconds):
    if isinstance(length, int):
        length = [length]
    d = [ len(length) ] + list(length) + [ nbatch, len(seconds) ] + seconds
    t = '\t'.join([str(x) for x in d]) + '\n'
    with path(fname).open('a') as f:
        f.write(t)

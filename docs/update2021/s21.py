
from itertools import product
from perflib.generators import Problem


def mktag(tag, precision, direction, inplace, real):
    t = [tag,
         precision,
         {-1: 'forward', 1: 'backward'}[direction],
         {True: 'real', False: 'complex'}[real],
         {True: 'in-place', False: 'out-of-place'}[inplace]]
    return "_".join(t)


def md1():
    lengths = [ (192,96,84), ]
    precisions = ['single']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]
    nbatch     = 1

    for length, precision, direction, inplace, real in product(lengths, precisions, directions, inplaces, reals):
        yield Problem(length,
                      tag=mktag('md', precision, direction, inplace, real),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=inplace,
                      real=real,
                      precision=precision)


def large1d():

    lengths = [8192, 10752, 18816, 21504, 32256, 43008, 16384, 16807]
    precisions = ['single']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]
    nbatch     = 1000

    for length, precision, direction, inplace, real in product(lengths, precisions, directions, inplaces, reals):
        yield Problem(legnth,
                      tag=mktag('large1d', precision, direction, inplace, real),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=inplace,
                      real=real,
                      precision=precision)

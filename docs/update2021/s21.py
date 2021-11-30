
from itertools import product
from perflib.generators import Problem


def mktag(tag, precision, direction, inplace, real):
    t = [tag,
         precision,
         {-1: 'forward', 1: 'backward'}[direction],
         {True: 'real', False: 'complex'}[real],
         {True: 'in-place', False: 'out-of-place'}[inplace]]
    return "_".join(t)


md = [ (100,100,100), (160,160,168), (160,168,168), (160,168,192),
       (160,72,72), (160,80,72), (160,80,80), (168,168,192),
       (168,192,192), (168,80,80), (192,192,192),
       (192,192,200), (192,200,200), (192,84,84),
       (192,96,84), (192,96,96), (200,100,96), (200,200,200),
       (200,96,96), (208,100,100), (216,104,100),
       (216,104,104), (224,104,104), (224,108,104),
       (224,108,108), (240,108,108), (240,112,108),
       (240,112,112), (60,60,60), (64,64,52), (64,64,64),
       (72,72,52), (72,72,72), (80,80,80), (84,84,72),
       (96,96,96), (108,108,80), (216,216,216),
       (128,128,256), (240,224,224), (64,64,64), (80,84,14),
       (80,84,144), (25,20,20), (42,32,32), (75,55,55) ]

def md1():

    precisions = ['single']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]
    nbatch     = 1

    for length, precision, direction, inplace, real in product(md, precisions, directions, inplaces, reals):
        yield Problem(length,
                      tag=mktag('md', precision, direction, inplace, real),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=inplace,
                      real=real,
                      precision=precision)

def md1000():

    precisions = ['single']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]
    nbatch     = 1

    for length, precision, direction, inplace, real in product(md, precisions, directions, inplaces, reals):
        yield Problem(length,
                      tag=mktag('md', precision, direction, inplace, real),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=inplace,
                      real=real,
                      precision=precision)


def cholla1d():

    lengths = [8192, 10752, 18816, 21504, 32256, 43008, 16384, 16807]
    precisions = ['double']
    directions = [-1]
    inplaces   = [True]
    reals      = [False]
    nbatch     = 1000

    for length, precision, direction, inplace, real in product(lengths, precisions, directions, inplaces, reals):
        yield Problem(legnth,
                      tag=mktag('cholla1d', precision, direction, inplace, real),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=inplace,
                      real=real,
                      precision=precision)

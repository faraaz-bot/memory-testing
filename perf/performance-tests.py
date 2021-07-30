#
# performance tests
#

import functools
import itertools
import numpy as np
import sympy
import types

NS = types.SimpleNamespace

dtypes = [
    NS(label='double', dtype=np.float64),
    NS(label='single', dtype=np.float32),
    ]

transforms = [
    NS(label='complex_forward'),
    NS(label='complex_backward'),
    NS(label='real_forward'),
    NS(label='real_backward'),
    ]


def product(xs):
    """Return the product of the factors."""
    if xs:
        return functools.reduce(lambda a, b: a * b, xs)
    return 1


def make_suite(suite):
    def fft_test_generator():
        for transform in transforms:
            for dtype in dtypes:
                label = '_'.join([suite.label, transform.label, dtype.label])
                if getattr(suite, 'size', None) is not None:
                    suite.nbatch = suite.size // product(suite.lengths)
                yield dict(label=label,
                           transform=transform,
                           lengths=suite.lengths,
                           nbatch=suite.nbatch,
                           dtype=dtype)
    return fft_test_generator


#
# Basic suites
#

pow2  = NS(label='pow2', lengths=[ 2**k for k in range(9,17) ], nbatch=10000)
pow5  = NS(label='pow5', lengths=[ 5**k for k in range(4,8) ], nbatch=5000)
pow7  = NS(label='pow7', lengths=[ 7**k for k in range(3,7) ], nbatch=5000)
prime = NS(label='prime', lengths=list(sympy.sieve.primerange(11, 1000)), nbatch=10000)
mixed = NS(label='mixed', lengths=[225, 240, 300, 486, 600, 900, 958, 1014, 1139,
                                   1250, 1427, 1463, 1480, 1500, 1568, 1608, 1616, 1638, 1656,
                                   1689, 1696, 1708, 1727, 1744, 1752, 1755, 1787, 1789, 1828,
                                   1833, 1845, 1860, 1865, 1875, 1892, 1897, 1899, 1900, 1903,
                                   1905, 1912, 1933, 1938, 1951, 1952, 1954, 1956, 1961, 1964,
                                   1976, 1997, 2004, 2005, 2006, 2012, 2016, 2028, 2033, 2034,
                                   2038, 2069, 2100, 2113, 2116, 2123, 2136, 2152, 2160, 2167,
                                   2181, 2182, 2187, 2205, 2208, 2242, 2250, 2251, 2288, 2306,
                                   2342, 2347, 2352, 2355, 2359, 2365, 2367, 2383, 2385, 2387,
                                   2389, 2429, 2439, 2445, 2448, 2462, 2467, 2474, 2478, 2484,
                                   2486, 2496, 2500, 2503, 2519, 2525, 2526, 2533, 2537, 2556,
                                   2558, 2559, 2566, 2574, 2576, 2594, 2604, 2607, 2608, 2612,
                                   2613, 2618, 2632, 2635, 2636, 2641, 2652, 2654, 2657, 2661,
                                   2663, 2678, 2688, 2690, 2723, 2724, 2728, 2729, 2733, 2745,
                                   2755, 2760, 2772, 2773, 2780, 2786, 2789, 2790, 2805, 2807,
                                   2808, 2812, 2815, 2816, 2820, 2826, 2830, 2834, 2841, 2847,
                                   2848, 2850, 2852, 2853, 2872, 2877, 2882, 2883, 2886, 2887,
                                   2892, 2893, 2917, 2922, 2924, 2926, 2928, 2929, 2932, 2933,
                                   2934, 2938, 2951, 2960, 2970, 2979, 2990, 2994, 2998, 2999,
                                   3000, 3001, 3003, 3004, 3008, 3034, 3035, 3039, 3040, 3042,
                                   3048, 3052, 3055, 3060, 3065, 4000, 12000, 24000], nbatch=1000)

#
# Generated lengths, see kernel_generator.py
#

lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20,
           21, 22, 24, 25, 26, 27, 28, 30, 32, 36, 40, 42, 44, 45, 48,
           49, 50, 52, 54, 56, 60, 64, 72, 75, 80, 81, 84, 88, 90, 96,
           100, 104, 108, 112, 120, 121, 125, 128, 135, 144, 150, 160,
           162, 168, 169, 176, 180, 192, 200, 208, 216, 224, 225, 240,
           243, 250, 256, 270, 288, 300, 320, 324, 336, 343, 360, 375,
           384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600,
           625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900,
           960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250,
           1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728,
           1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250,
           2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000,
           3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
           3888, 4000, 4050, 4096]

generated1d = NS(label='generated1d', lengths=lengths, nbatch=1000)

lengths2d = list(filter(lambda x: x <= 1024, lengths))
generated2d = NS(label='generated2d', lengths=list(zip(lengths2d, lengths2d)), nbatch=100)

lengths3d = list(filter(lambda x: x <= 512, lengths))
generated3d = NS(label='generated3d', lengths=list(zip(lengths3d, lengths3d, lengths3d)), nbatch=1)

#
# Special lengths
#

cholla1d = NS(label='cholla1d', lengths=[
    8192, 10752, 18816, 21504, 32256, 43008, 16384, 16807], nbatch=10000)

cholla10k = NS(label='cholla10k', lengths=[
    10000], nbatch=10000)

cholla2d = NS(label='cholla2d', lengths=[(256,256)], nbatch=100)

vasp1d = NS(label='vasp1d', lengths=[56, 336], nbatch=10000)

vasp3d = NS(label='vasp3d', lengths=[(336,336,56)], nbatch=10)

gromacs3d = NS(label='gromacs3d',
               lengths=[
                   (100,100,100),
                   (160,160,168),
                   (160,168,168),
                   (160,168,192),
                   (160,72,72),
                   (160,80,72),
                   (160,80,80),
                   (168,168,192),
                   (168,192,192),
                   (168,80,80),
                   (192,192,192),
                   (192,192,200),
                   (192,200,200),
                   (192,84,84),
                   (192,96,84),
                   (192,96,96),
                   (200,100,96),
                   (200,200,200),
                   (200,96,96),
                   (208,100,100),
                   (216,104,100),
                   (216,104,104),
                   (224,104,104),
                   (224,108,104),
                   (224,108,108),
                   (240,108,108),
                   (240,112,108),
                   (240,112,112),
                   (60,60,60),
                   (64,64,52),
                   (64,64,64),
                   (72,72,52),
                   (72,72,72),
                   (80,80,80),
                   (84,84,72),
                   (96,96,96),
               ],
               nbatch=10)

namd3d = NS(label='namd3d',
            lengths=[
                (108,108,80),
                (216,216,216),
            ],
            nbatch=10)

amber3d = NS(label='amber3d',
             lengths=[
                 (128,128,256),
                 (240,224,224),
                 (64,64,64),
                 (80,84,14),
                 (80,84,144),
             ],
             nbatch=10)

cp2k = NS(label='cp2k',
          lengths=[
              (25,20,20),
              (42,32,32),
              (75,55,55)
          ],
          nbatch=10)

warpx3d = NS(label='warpx3d',
             lengths=[3*(2**k + 16,) for k in range(5, 9)],
             nbatch=10)

mi2002d = NS(label='mi2002d',
             lengths=[
                 (4096, 4096),
                 (336, 18816),
             ],
             nbatch=1)

mi2003d = NS(label='mi2003d',
             lengths=[
                 (256, 256, 256),
                 (336, 336, 56),
             ],
             nbatch=1)

#
# Everything!
#

def all():
    """All of the above!."""
    generators = [pow2,
                  pow5,
                  pow7,
                  prime,
                  mixed,
                  generated1d,
                  generated2d,
                  generated3d,
                  cholla1d,
                  cholla2d,
                  vasp1d,
                  vasp3d,
                  gromacs3d,
                  namd3d,
                  amber3d,
                  cp2k,
                  warpx3d,
                  mi2002d,
                  mi2003d]
    return itertools.chain(*[ make_suite(f)() for f in generators ])


def clients():
    """Only 'client' tests."""

    # batched, complex forward, double, out-of-place
    transform = NS(label='complex_forward')
    dtype     = NS(label='double', dtype=np.float64)
    placement = NS(label='outplace')
    for suite in [cholla1d]:
        label = '_'.join([suite.label, transform.label, dtype.label, placement.label])
        yield dict(label=label,
                   transform=transform,
                   lengths=suite.lengths,
                   nbatch=suite.nbatch,
                   dtype=dtype,
                   placement=placement)

    # one, complex forward, double, out-of-place
    transform = NS(label='complex_forward')
    dtype     = NS(label='double', dtype=np.float64)
    placement = NS(label='outplace')
    for suite in [vasp3d]:
        label = '_'.join([suite.label, transform.label, dtype.label, placement.label])
        yield dict(label=label,
                   transform=transform,
                   lengths=suite.lengths,
                   nbatch=1,
                   dtype=dtype,
                   placement=placement)

    # batched, complex inverse, double, out-of-place
    transform = NS(label='complex_backward')
    dtype     = NS(label='double', dtype=np.float64)
    placement = NS(label='outplace')
    for suite in [cholla10k]:
        label = '_'.join([suite.label, transform.label, dtype.label, placement.label])
        yield dict(label=label,
                   transform=transform,
                   lengths=suite.lengths,
                   nbatch=suite.nbatch,
                   dtype=dtype,
                   placement=placement)

    # one, real forward/backward, single, out-of-place
    transforms = [NS(label='real_forward'), NS(label='real_backward')]
    dtype      = NS(label='single', dtype=np.float32)
    placement  = NS(label='outplace')
    for transform in transforms:
        for suite in [gromacs3d, amber3d, namd3d, cp2k]:
            label = '_'.join([suite.label, transform.label, dtype.label, placement.label])
            yield dict(label=label,
                       transform=transform,
                       lengths=suite.lengths,
                       nbatch=1,
                       dtype=dtype,
                       placement=placement)

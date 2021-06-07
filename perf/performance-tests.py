#
# performance tests
#

import numpy as np
import sympy
import types

import perflib.transforms

import functools
import itertools

NS = types.SimpleNamespace

dtypes = [
    NS(label='double', dtype=np.float64, rider=['--double']),
    NS(label='single', dtype=np.float32, rider=[]),
    ]

transforms = [
    NS(label='complex_forward', transform=perflib.transforms.complex_forward, rider=['-t', '0']),
    NS(label='complex_backward', transform=perflib.transforms.complex_backward, rider=['-t', '1']),
    NS(label='real_forward', transform=perflib.transforms.real_forward, rider=['-t', '2']),
    NS(label='real_backward', transform=perflib.transforms.real_backward, rider=['-t', '3']),
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

pow2  = make_suite(NS(label='pow2', lengths=[ 2**k for k in range(9,17) ], nbatch=2048))
pow5  = make_suite(NS(label='pow5', lengths=[ 5**k for k in range(4,8) ], nbatch=2000))
pow7  = make_suite(NS(label='pow7', lengths=[ 7**k for k in range(3,7) ], nbatch=1000))
prime = make_suite(NS(label='prime', lengths=list(sympy.sieve.primerange(11, 1000)), nbatch=10000))
mixed = make_suite(NS(label='mixed', lengths=[225, 240, 300, 486, 600, 900, 958, 1014, 1139,
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
                                      3048, 3052, 3055, 3060, 3065, 4000, 12000, 24000], nbatch=1000))

#
# Generated lengths, see kernel_generator.py
#

powers = {
    17: [17**k for k in range(4)],
    13: [13**k for k in range(5)],
    11: [11**k for k in range(5)],
    7: [7**k for k in range(6)],
    5: [5**k for k in range(7)],
    3: [3**k for k in range(9)],
    2: [2**k for k in range(13)],
}

lengths = [ p2 * p3 * p5 * p7 * p11 * p13 * p17 for p2, p3, p5, p7, p11, p13, p17 in itertools.product(powers[2], powers[3], powers[5], powers[7], powers[11], powers[13], powers[17])]
lengths += [7, 14, 21, 28, 42, 49, 56, 84, 112, 168, 224, 336, 343]
lengths += [11, 22, 44, 88, 121, 176]
lengths += [13, 26, 52, 104, 169, 208]
lengths = sorted(set(filter(lambda x: x <= 4096 and x > 1, lengths)))

generated1d = make_suite(NS(label='generated1d', lengths=lengths, nbatch=10000))

lengths2d = list(filter(lambda x: x <= 32, lengths))
generated2d = make_suite(NS(label='generated2d', lengths=list(zip(lengths2d, lengths2d)), nbatch=100))

lengths3d = list(filter(lambda x: x <= 512, lengths))
generated3d = make_suite(NS(label='generated3d', lengths=list(zip(lengths3d, lengths3d, lengths3d)), nbatch=1))

#
# Special lengths
#

cholla1d = make_suite(NS(label='cholla1d', lengths=[
    8192, 10752, 18816, 21504, 32256, 43008, 16384, 16807, 10000 ], nbatch=1000))

cholla2d = make_suite(NS(label='cholla2d', lengths=[(256,256)], nbatch=100))

vasp1d = make_suite(NS(label='vasp1d', lengths=[56, 336], nbatch=10000))

vasp3d = make_suite(NS(label='vasp3d', lengths=[(336,336,56)], nbatch=10))

gromacs3d = make_suite(NS(label='gromacs3d',
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
                          nbatch=10))

namd3d = make_suite(NS(label='namd3d',
                       lengths=[
                           (108,108,80),
                           (216,216,216),
                       ],
                       nbatch=10))

amber3d = make_suite(NS(label='amber3d',
                        lengths=[
                            (128,128,256),
                            (240,224,224),
                            (64,64,64),
                            (80,84,14),
                            (80,84,144),
                        ],
                        nbatch=10))

cp2k = make_suite(NS(label='cp2k',
                     lengths=[
                         (25,20,20),
                         (42,32,32),
                         (75,55,55)
                     ],
                     nbatch=10))

warpx3d = make_suite(NS(label='warpx3d',
                        lengths=[3*(2**k + 16,) for k in range(5, 10)],
                        nbatch=10))

#
# Everything!
#

def all():
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
                  warpx3d]
    return itertools.chain(*[ f() for f in generators ])

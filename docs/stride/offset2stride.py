#!/usr/bin/env python3


import math
import sys
import itertools
import numpy as np
import copy

import  valid 

def get_offsets(length, stride):
    if len(length) == 0:
        yield 0
    else:
        for offset in get_offsets(length[1:], stride[1:]):
            for i in range(length[0]):
                yield i * stride[0] + offset

def simple_linear_diophantine_i(a, b, debug=False):
    qs = []
    while True:
        q, r = divmod(a, b)
        if debug:
            print(f'a={a}, b={b}, q={q}, r={r}')
        a = b
        b = r
        if r != 0:
            qs.append(q)
        else:
            break
    x, y = b, a
    for q in qs[::-1]:
        if debug:
            print(f'x={x}, y={y}, q={q}')
        x, y = y, x - q * y
    return [x, y]

def getidx(offset, s, l, X):
    x = copy.deepcopy(X)
    g = math.gcd(s[0], s[1])
    og = offset // g
    x[0] *= og
    x[1] *= og
    
    n = 0
    if x[0] < 0:
        n = (x[0] // s[1])
    if x[1] < 0:
        n = x[1] // s[0]
    
    x[0] = x[0] + n * s[1]
    x[1] = x[1] - n * s[0]

    if x[0] >= l[0]:
        n -= 1
        x[0] -= s[1]
        x[1] += s[0]
        
    if x[1] >= l[1]:
        n += 1
        x[0] += s[1]
        x[1] -= s[0]
    
    return x
    
if True:
    s = [3, 5]
    l = [5, 4]
    #s0 = 258
    #s1 = 147

    print("s:", s)
    print("l:", l)
    if not valid.is_valid2(s[0], s[1], l[0], l[1]):
        print("invalid array format")
        sys.exit(1)

    sgcd = math.gcd(s[0], s[1])
    
    idx00 = simple_linear_diophantine_i(s[0], s[1])
    print("idx00:", idx00)
    print(sgcd)
    print(np.dot(idx00, s))

    for idx in (list(idx) for idx in itertools.product(range(l[0]), range(l[1]))):
        offset = np.dot(idx, s)
        print(idx, offset)
        idx0 = getidx(offset, s, l, idx00)
        offset0 = np.dot(s, idx0)
        print("\t", idx0, offset0 )
        
        if offset == offset0 and idx0 == idx:
            print("good")
        else:
            print("FAIL!")


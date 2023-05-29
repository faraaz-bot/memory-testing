#!/usr/bin/env python3


import math
import sys
import itertools


from valid import *

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
    return x, y

def getidx(offset, s0, s1, l0, l1, x, y):
    g = math.gcd(s0, s1)
    og = offset // g
    x *= og
    y *= og
    
    n = 0
    if x < 0:
        n = (x // s1)
    if y < 0:
        n = y // s0
    
    x = x + n * s1
    y = y - n * s0

    if x >= l0:
        n -= 1
        x -= s1
        y += s0
        
    if y >= l1:
        n += 1
        x += s1
        y -= s0
    
    return x, y
    
if True:
    s0 = 3
    s1 = 5
    #s0 = 258
    #s1 = 147

    l0 = 5
    l1 = 4
    print("s:", (s0, s1))
    print("l:", (l0, l1))
    print(lcm(s0, s1))
    #cols = strides.collisions2(s0, s1, l0, l1)
    print(is_valid2(s0, s1, l0, l1))
    #print(cols)
    lengths = [l0, l1]
    strides = [s0, s1]

    x,y = simple_linear_diophantine_i(s0, s1)
    x0, y0 = x // math.gcd(s0, s1), y // math.gcd(s0, s1), 
    print(x, y)
    print(math.gcd(s0, s1))
    print((s0 * x0 + s1 * y0))
    
    for idx in itertools.product(range(l0), range(l1)):
        offset = idx[0] * s0 + idx[1] * s1
        print(idx, offset)
        x, y = getidx(offset, s0, s1, l0, l1, x0, y0)
        offset0 = s0 * x + s1 * y
        print("\t", x, y, offset0 )
        if offset == offset0 and x == idx[0] and y == idx[1]:
            print("good")
        else:
            print("FAIL!")
                
        
            

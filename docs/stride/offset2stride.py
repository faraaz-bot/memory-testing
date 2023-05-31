#!/usr/bin/env python3


import math
import sys
import itertools
import numpy as np
import copy

import valid 

def ceildiv(num, dem):
    return (num + dem - 1) // dem

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

def getidx2(offset, sgcd, S, l, X):
    x = copy.deepcopy(X)
    s = copy.deepcopy(S)


    og = offset // sgcd
    x[0] *= og
    x[1] *= og
    s[0] //= sgcd
    s[1] //= sgcd
    #print("sgcd:", sgcd)
    
    #print("x:", x, "s:", s, "l:", l, np.dot(x, s))
    #print("og:", og)
   

    # If any indices are out-of-bounds, shift so that they're in-bounds.
    #print(x)

    if True:
        while x[0] >= l[0] or x[1] < 0:
            x[0] -= s[1]
            x[1] += s[0]
            #print("asdf", x)
        while x[1] >= l[1] or x[0] < 0:
            x[0] += s[1]
            x[1] -= s[0]
            #print("qwer", x)
    else:
        n = 0
        if x[0] >= l[0]:
            #print("x =", x, "l[0] =", l[0], "s[1] =", s[1] )
            n = (l[0] - 1 - x[0]) // s[1]
        elif x[0] < 0:
            n = (-x[0]) // s[1]

        x[0] += n * s[1]
        x[1] -= n * s[0]
        if x[1] < 0:
            x[0] -= s[1]
            x[1] += s[0]

        n = 0
        if x[1] >= l[1]:
            #print("x =", x, "l[1] = ", l[1])
            n = -((l[1] - 1 - x[1]) // s[0])
        elif x[1] < 0:
            n = (-x[1]) // s[0]

        x[0] += n * s[1]
        x[1] -= n * s[0]
        
        if x[0] < 0:
            x[0] += s[1]
            x[1] -= s[0]
        
        #print("n:", n)
    
    return x
    
if True:
    #s = [3, 5]
    s = [6, 10]
    l = [5, 4]
    #s0 = 258
    #s1 = 147

    print("s:", s)
    print("l:", l)
    if not valid.is_valid2(s, l):
        print("invalid array format")
        sys.exit(1)

    sgcd = math.gcd(s[0], s[1])
    
    idx00 = simple_linear_diophantine_i(s[0] // sgcd, s[1] // sgcd)
    #idx00[0] *= sgcd
    #idx00[1] *= sgcd
    print("idx00:", idx00)
    print(sgcd)
    print(np.dot(idx00, s))

    for idx in (list(idx) for idx in itertools.product(*[range(l0) for l0 in l])):
        offset = np.dot(idx, s)
        print("forward", idx, "dot", s, "=", offset)
        idx0 = getidx2(offset, sgcd, s, l, idx00)
        offset0 = np.dot(s, idx0)
        print("\t", idx0, offset0 )
        
        if offset == offset0 and idx0 == idx:
            print("good")
        else:
            print("FAIL!")
            sys.exit(1)


lmax = 5
            
fails2 = []
for s in (list(ss) for ss in itertools.product(range(2, lmax), range(2, lmax))):
    sgcd = math.gcd(s[0], s[1])
    idx00 = simple_linear_diophantine_i(s[0] // sgcd, s[1] // sgcd)
    print("idx00:", idx00)
    print("gcd:", sgcd)
    #print(np.dot(idx00, s))
    for l in (list(lens) for lens in itertools.product(range(2, lmax), range(2, lmax))):
        if valid.is_valid2(s, l):
            print("s:", s, "l:", l)
            for idx in (list(idx) for idx in itertools.product(*[range(l0) for l0 in l])):
                offset = np.dot(idx, s)
                print("forward:", idx, offset)
                idx0 = getidx2(offset, sgcd, s, l, idx00)
                offset0 = np.dot(s, idx0)
                print("\t", idx0, offset0 )
                
                if offset == offset0 and idx0 == idx:
                    print("good")
                else:
                    print("FAIL!")
                    fails2.append([s,l])
                    sys.exit(1)
      
fails3 = []      

# Parametrize a 3D multi-index solution.
def idx3gen(idx, s, n, m):
    g = math.gcd(s[0], s[1])
    h = math.gcd(s[0], s[2])
    pidx = [idx[0] - n * s[1] // g  - m * s[2] // h,
            idx[1] + n * s[0] // g,
            idx[2]  + m * s[0] // h]
    return pidx

def getidx3(offset, s, l, idx0):
    
    x = [idx0[0] * offset, idx0[1] * offset, idx0[2] * offset]
    
    g = math.gcd(s[0], s[1])
    h = math.gcd(s[0], s[2])
    
    itmax = 40
    it = 0
    print(x)

    # Get index 1 and 2 to zero.
    n = g * x[1] // s[0]
    m = h * x[2] // s[0]

    x = idx3gen(x, s, m, n)

    if x[0] < 0:
        x[0] += s[1] // g
        x[1] -= s[0] // g
        x[0] += s[2] // h
        x[2] -= s[0] // h

    while x[0] >= l[0] and it < itmax:
        it += 1
        if x[1] < l[1] - s[0] // g and x[0] >= s[1] // g:
            x[0] -= s[1] // g
            x[1] += s[0] // g
        if x[2] < l[2] - s[0] // h and x[0] >= s[2] // h:
            x[0] -= s[2] // h
            x[2] += s[0] // h
        print(x)
    
    while False and (x[0] < 0 or x[0] >= l[0]):
        #it += 1
        print(x, l)
        if x[0] < 0:
            if x[2] < l[2]:
                x[0] += s[1] // g
                x[1] -= s[0] // g
                continue
            if x[1] < l[1]:
                x[0] += s[2] // h
                x[2] -= s[0] // h
                continue
            x[0] += s[1] // g
            x[1] -= s[0] // g
            x[0] += s[2] // h
            x[2] -= s[0] // h
        else:
            x[0] -= s[1] // g
            x[1] += s[0] // g

            if x[1] < 0:
                x[0] -= s[1] // g
                x[1] += s[0] // g
                continue
            x[0] -= s[2] // h
            x[2] += s[0] // h

            if x[2] < 0:
                x[0] -= s[2] // h
                x[2] += s[0] // h
                continue
                
    return x
    
lmax = 6
for s in (list(ss) for ss in itertools.product(range(1, lmax), range(1, lmax), range(1, lmax))):
    
    sgcd = math.gcd(s[0], s[1], s[2])
    #print(s, l, sgcd)

    s2 = [s[0], math.gcd(s[1],s[2])]
    s2gcd = math.gcd(s2[0], s2[1])
    s2[0] //= s2gcd
    s2[1] //= s2gcd
    idx2 =  simple_linear_diophantine_i(s2[0], s2[1])
    #print("\tidx2:", idx2, s2, s2gcd, np.dot(s2, idx2))

    s3 = [s[1], s[2]]
    s3gcd = math.gcd(s3[0], s3[1])
    s3[0] //= s3gcd
    s3[1] //= s3gcd
    idx3 = simple_linear_diophantine_i(s3[0], s3[1])
    #print("\tidx3:", idx3, s3, np.dot(s3, idx3))

    idx0 = [idx2[0], idx2[1] * idx3[0], idx2[1] * idx3[1]]
    #print("\tidx0:", idx0, np.dot(idx0, s))
    
    for l in (list(ss) for ss in itertools.product(range(2, lmax), range(2, lmax), range(2, lmax))):
        valid3 = valid.is_valid3(s, l)
        if valid3:
            for idx in (list(idx) for idx in itertools.product(*[range(l0) for l0 in l])):
                offset = np.dot(s, idx)
                print("\t\tidx:", idx, s, offset)
                idx00 = [idx0[0] * offset, idx0[1] * offset,idx0[2] * offset]
                #pidx = idx3gen(idx00, s, 1, 2)

                pidx = getidx3(offset, s, l, idx0)

                loffset = np.dot(pidx, s)
                print("\t\t",idx, pidx, offset, loffset, offset==loffset)
                # TODO: re-index
                if offset != loffset:
                    fails3.append([s,l])
                if pidx != idx:
                    print(pidx, idx, s, l)
                    sys.exit(0)

print(fails3)
                

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

maxoknsize = 0

def getidx2(offset, sgcd, S, l, X, alg):
    x = copy.deepcopy(X)
    s = copy.deepcopy(S)

    #print(x)
    og = offset // sgcd
    x[0] *= og
    x[1] *= og
    #print(x, s, np.dot(x,s), offset)
    #s[0] //= sgcd
    #s[1] //= sgcd
    #print("sgcd:", sgcd)

    ss = copy.deepcopy(s)
    ss[0] //= sgcd
    ss[1] //= sgcd
    
    #print("x:", x, "s:", s, "l:", l, np.dot(x, s))
    #print("ss:", ss)
    #print("og:", og)
   
    if offset == 0:
        return [0, 0]
    
    # If any indices are out-of-bounds, shift so that they're in-bounds.
    #print(x)

    if alg == 0:
        while x[0] >= l[0] or x[1] < 0:
            x[0] -= ss[1]
            x[1] += ss[0]
            #print("asdf", x)
        while x[1] >= l[1] or x[0] < 0:
            #print("\t\t",x)
            x[0] += ss[1]
            x[1] -= ss[0]
            #print("qwer", x)
    elif alg == 1:
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
    elif alg == 2:

        if x[0] >= 0 and x[0] < l[0] and x[1] >= 0 and x[1] < l[1]:
            return x
        
        print(x, s, l, offset)
        # Allowable values for x[0]:
        n = None
                
        r0 = -np.sign(x[0])* (np.abs(x[0])// ss[1])
        
        r1 = np.sign((l[1] - x[0])) * ceildiv(np.abs(l[0] - x[0]) , ss[1])
        #print("r0:", r0)
        #print("r1:", r1)
        okn = set(())
        for n0 in range(min(r0, r1) - 1, max(r0, r1) + 1):
            #n0 = r0 + in0 * s[1] 
            x0 = x[0] + n0 * ss[1]
            #print("\tn0:", n0, "x0:", x0)
            #print("\tn0:", n0)
            if x0 >= 0 and x0 < l[0]:
                okn.add(int(n0))
        print("okn:", okn)
        global maxoknsize
        maxoknsize = max(maxoknsize, len(okn))
        for n0 in okn:
            x1 = x[1] - n0 * ss[0]
            #print("\tn0:", n0, "x1:", x1)
            #print("x1:", x1)
            if x1 >= 0 and x1 < l[1]:
                n = n0
                break
        if n == None:
            print("didn't find n")
            sys.exit(1)
            
        x[0] += n * ss[1]
        x[1] -= n * ss[0]
        #print("found n=", n0, "->", x)
    return x
    
if False:
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
    
    idx00 = simple_linear_diophantine_i(s[0], s[1])
    # idx00[0] *= sgcd
    # idx00[1] *= sgcd
    print("idx00:", idx00)
    print(sgcd)
    print(np.dot(idx00, s))

    for idx in (list(idx) for idx in itertools.product(*[range(l0) for l0 in l])):
        offset = np.dot(idx, s)
        print("forward: dot(x,s):", idx, s, "->", offset)
        idx0a = getidx2(offset, sgcd, s, l, idx00, 0)
        idx0 = getidx2(offset, sgcd, s, l, idx00, 2)
        offset0 = np.dot(s, idx0)
        print("\t", idx0, offset0 )
        
        if offset == offset0 and idx0 == idx:
            print("good")
        else:
            print("FAIL!")
            sys.exit(1)


lmax = 2
            
fails2 = []
for s in (list(ss) for ss in itertools.product(range(2, lmax), range(2, lmax))):
    sgcd = math.gcd(s[0], s[1])
    idx00 = simple_linear_diophantine_i(s[0] // sgcd, s[1] // sgcd)
    print("idx00:", idx00)
    print("gcd:", sgcd)
    print(np.dot(idx00, s))
    
    for l in (list(lens) for lens in itertools.product(range(2, lmax), range(2, lmax))):
        if valid.is_valid2(s, l):
            print("s:", s, "l:", l)
            for idx in (list(idx) for idx in itertools.product(*[range(l0) for l0 in l])):
                offset = np.dot(idx, s)
                print("forward: dot(x,s):", idx, s, "->", offset)
                idx0a = getidx2(offset, sgcd, s, l, idx00, 0)
                idx0 = getidx2(offset, sgcd, s, l, idx00, 2)
                offset0 = np.dot(s, idx0)
                print("\t", idx0, offset0 )
                
                if offset == offset0 and idx0 == idx:
                    print("good")
                else:
                    print("FAIL!")
                    fails2.append([s,l])
                    sys.exit(1)
print("fails2:", fails2)
      
fails3 = []
print("lmax:", lmax)
print("maxoknsize:", maxoknsize)


# Parametrize a 3D multi-index solution.
def idx3gen(idx, s, m, n, offset):
    # idx is a solution to the case where s is coprime.
    g = math.gcd(s[0], s[1])
    h = math.gcd(s[0], s[2])
    k = math.gcd(s[1], s[2])

    l = math.gcd(k, s[0])

    # TODO: move solvers out of this function.
    w = simple_linear_diophantine_i(s[0], k)
    
    v = simple_linear_diophantine_i(s[1], s[2])
    
    
    sgcd = math.gcd(s[0], s[1], s[2])
    og = offset // sgcd

    print(s)
    print(k)
    print(l)
    print(s[0], s[2] // k, s[2])
    print(s[0], s[1] // k, s[1])

    # m = 1
    # n = 1
    
    print(m, n)

    
    pidx = [og * idx[0] + m * k // l,
            v[0] * (og * w[1] - m * s[0] // l ) // k + n * s[2] // k,
            v[1] * (og * w[1] - m * s[0] // l ) // k - n * s[1] // k]

    print(pidx)
    return pidx

# Find the 3D index which is in the bounded domain, given the initial solution idx0.
def getidx3(offset, s, L, idx):

    if offset == 0:
        return [0, 0, 0]
    
    sgcd = math.gcd(s[0], s[1], s[2])

    og = offset // sgcd
    
    g = math.gcd(s[0], s[1])
    h = math.gcd(s[0], s[2])
    k = math.gcd(s[1], s[2])
    
    l = math.gcd(k, s[0])
    
    # TODO: move solvers out of this function.
    w = simple_linear_diophantine_i(s[0], k)
    
    v = simple_linear_diophantine_i(s[1], s[2])
    
    
    #print(x)

    print(idx3gen(idx, s, -4, -2, offset))
    
    print("L:", L)
    if True:
        okm = set(())
        if True:
            # Find the valid values for m:
            r0 = -np.sign( og * idx[0] )* ( np.abs( og * idx[0] ) // k // l)
            r1 = np.sign(L[0] - og * idx[0]) * ceildiv(np.abs(L[0] - og * idx[0]) , k // l)
            print("r0, r1:", r0, r1)

            for m0 in range(min(r0, r1) - 1, max(r0, r1) + 1):
                x0 = og * ( idx[0] ) + m0 * k // l
                print("m0:", m0, "->", x0)
                if x0 >= 0 and x0 < L[0]:
                    okm.add(int(m0))
        print("okm:", okm)

        m = None
        n = None

        for m0 in okm:
            x1m = v[0] * (og * w[1] - m0 * s[0] // l ) // k 
            x2m = v[1] * (og * w[1] - m0 * s[0] // l ) // k
            print(m0, x1m, x2m)
            q0 = -np.sign( x1m ) * ( np.abs( x1m ) // (s[2] // k) )
            q1 =  np.sign( L[1] - x1m ) * ceildiv(np.abs( L[1] - x1m ) , s[2] // k )
            print( np.abs( L[1] - x1m ), s[2], k, s[2] // k )
            print("q0, q1:", q0, q1)
            for n0 in range(min(q0, q1) - 1, max(q0, q1) + 1):
                x1 = x1m + n0 * s[2] // k
                x2 = x2m - n0 * s[1] // k
                print("n0:", n0, "->", x1, x2)
                if x1 >= 0 and x2 >= 0 and x1 < L[1] and x2 < L[2]:
                    n = n0
                    m = m0
                    break
            if m!= None and n != None:
                break
            
        if n == None or m == None:
            print("no solution found")
            print("k:", k)
            print("l:", l)
            print("L:", L)
            print("s:", s)
            print("idx0:", idx0)
            print("offset:", offset)
            print("og:", og)
            print("g, h:", g, h)
            sys.exit(1)
    
    #x = [idx[0] * og, idx[1] * og, idx[2] * og]
    x = idx3gen(idx, s, m, n, offset)

    
    return x
    
lmax = 5
print("lmax:", lmax)
for s in (list(ss) for ss in itertools.product(range(1, lmax), range(1, lmax), range(1, lmax))):
    
    sgcd = math.gcd(s[0], s[1], s[2])

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
    print("idx0:", idx0, "s:", s)
    
    for l in (list(ss) for ss in itertools.product(range(2, lmax), range(2, lmax), range(2, lmax))):
        valid3 = valid.is_valid3(s, l)
        if valid3:
            for idx in (list(idx) for idx in itertools.product(*[range(l0) for l0 in l])):
                offset = np.dot(s, idx)
                print("forward: dot(x,s):", idx, s, "->", offset)
                #print("\t\tidx:", idx, s, offset)
                #idx00 = [idx0[0] * offset, idx0[1] * offset,idx0[2] * offset]
                #pidx = idx3gen(idx00, s, 1, 2)

                pidx = getidx3(offset, s, l, idx0)

                loffset = np.dot(pidx, s)
                print("\t\t",idx, pidx, offset, loffset, offset==loffset)

                if offset != loffset:
                    print("offset doesn't match")
                    fails3.append([s,l])
                    sys.exit(0)
                if pidx != idx:
                    print("index doesn't match")
                    print(pidx, idx, s, l)
                    sys.exit(0)

print("fails3:", fails3)
                

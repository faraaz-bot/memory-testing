#!/usr/bin/python3

import math
import sys

def lcm(x, y):
    return x * y // math.gcd(x, y)

if False:
    s0 = 3
    s1 = 5
    l0 = 6
    l1 = 4
    print(lcm(s0, s1))
    cols = collisions2(s0, s1, l0, l1)
    print(is_valid2(s0, s1, l0, l1))
    print(cols)

def collisions2(s0, s1, l0, l1):
    flat = []
    for i in range(l0):
        for j in range(l1):
            pos = i * s0 + j * s1
            flat.append([[i, j], pos])
    cols = []
    for i in range(len(flat)):
        for j in range(i + 1 , len(flat)):
            if flat[i][1] == flat[j][1]:
                cols.append([flat[i][0], flat[j][0], flat[i][1]])
    return cols

def check_valid2(s0, s1, l0, l1):
    flat = set([])
    for i in range(l0):
        for j in range(l1):
            pos = i * s0 + j * s1
            if pos in flat:
                return False
            else:
                flat.add(pos)
    return True


def is_valid2(s0, s1, l0, l1):

    # The indices need to be different:
    if s0 == s1:
        return False
    c = lcm(s0, s1)
    #print("c:", c)

    # If the lengths are too short to get to the lcm individually,
    # we're ok:
    if not (s0 * (l0 - 1) >= c) and  not (s1 * (l1 - 1) >= c):
        return True

    # # What about if we can get to them with the max index?
    # if (s0 * (l0 - 1) + s1 * (l1 - 1)) >= c:
    #     return True
    ## This doesn't work.  eg: length = (2,2), stride = 2,2

    # See if we can get to the lcm more than once:
    solfound = False
    a0 = l0 - 1
    while a0 >= 0:
        a1 = (c - a0 * s0) // s1
        #print("index:", a0, a1, "->", a0 * s0 + a1 * s1)
        if a1 < l1 and a1 >= 0:
            if a0 * s0 + a1 * s1 == c:
                if solfound:
                    return False
                solfound = True
        a0 -= 1
    return True



valid = []
invalid = []

lmax = 100

verbose = True

fails = []
for l0 in range(2, lmax):
    for l1 in range(2, lmax):
        print("length:", l0, l1)
        for s0 in range(1, 2 * l0 + 2):
            for s1 in range(1, 2 * l1 +2 ):
                if verbose:
                    print()
                    print("length:", l0, l1, "strides:", s0, s1, end='\t')
                testval = is_valid2(s0, s1, l0, l1)
                #cols = collisions2(s0, s1, l0, l1)
                #checkval = len(cols) == 0
                checkval = check_valid2(s0, s1, l0, l1)
                if (checkval != testval):
                    fails.append([s0, s1, l0, l1])
                    if verbose:
                        print("fail!")
                        print(l0, l1, s0, s1)
                        cols = collisions2(s0, s1, l0, l)
                        print(cols)
                        print("testval:", testval)
                        
                else:
                    if testval:
                        if verbose:
                            print("valid")
                        valid.append([s0, s1, l0, l1])
                    else:
                        if verbose:
                            print("not valid")
                        invalid.append([s0, s1, l0, l1])

if len(fails) == 0:
    print("test succesful!")
else:
    print("test FAILED!")
    print("fails:", fails)

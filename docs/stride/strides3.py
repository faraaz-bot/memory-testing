#!/usr/bin/python3

import math
import sys

from valid import *

lmin = 2
lmax = 17
def the_smax(l0, l1, l2):
    return l0 * l1 * l2

def check_valid3(s0, s1, s2, l0, l1, l2):
    flat = set([])
    for i0 in range(l0):
        for i1 in range(l1):
            for i2 in range(l2):
                pos = i0 * s0 + i1 * s1 + i2 * s2
                if pos in flat:
                    return False
                else:
                    flat.add(pos)
    return True

def collisions3(s0, s1, s2, l0, l1, l2):
    flat = []
    for i0 in range(l0):
        for i1 in range(l1):
            for i2 in range(l2):
                pos = i0 * s0 + i1 * s1 + i2 * s2
                flat.append([[i0, i1, i2], pos])
    #print(flat)
    cols = []
    for i in range(len(flat)):
        p = flat[i][1]
        icols = []
        icols.append(flat[i][0])
        for j in range(i + 1 , len(flat)):
            if p == flat[j][1]:
                icols.append(flat[j][0])
        if len(icols) > 1:
            cols.append([p, icols])
    return cols

def double3(sl):
    s = [x[0] for x in sl]
    l = [x[1] for x in sl]
    for i0 in range(l[0]):
        for i1 in range(l[1]):
            q = i0 * s[0] + i1 * s[1]
            #print(q, i0, i1)
            if q > 0:
                #print(q, S2, L2)
                if q % s[2]  == 0:
                    if q <= (l[2] - 1) * s[2]:

                        return False
    return True
    

def perm(a):
    b = []
    for i in range(len(a)):
        b.append( a[(i + 1) % len(a)])
    return b


def is_valid3(s0, s1, s2, l0, l1, l2):
    if s0 == s1 or s0 == s2 or s1 == s2:
        return False
    
    if not is_valid2(s0, s1, l0, l1):
        return False
    if not is_valid2(s0, s2, l0, l2):
        return False
    if not is_valid2(s1, s2, l1, l2):
        return False

    sl = [[s0, l0], [s1, l1], [s2, l2]]
    # #print(ls)
    # LS = sorted(ls, key=lambda x: x[0])
    # print(LS)
          
    # S0 = LS[0][0]
    # L0 = LS[0][1]
    # S1 = LS[1][0]
    # L1 = LS[1][1]
    # S2 = LS[2][0]
    # L2 = LS[2][1]
    for i in range(3):
        if not double3(sl):
            return False
        sl = perm(sl)
    
    return True


fails = []
nextl = 0
l0 = lmin
l1 = lmin
l2 = lmin
while l0 < lmax and l1 < lmax and l2 < lmax:
    smax = the_smax(l0, l1, l2)
    for s0 in range(1, smax):
        for s1 in range(1, smax  ):
            gcd01 = math.gcd(s0, s1)
            for s2 in range(1, smax ):
                gcd012 = math.gcd(gcd01, s2)
                if gcd012 == 1 and s0 * s1 * s2 < l0 * l1 * l2 * l0 *l1 * l2:
                    print("length:", l0, l1, l2, "stride", s0, s1, s2, end = "\t")
                    testval = is_valid3(s0, s1, s2, l0, l1, l2)
                    checkval = check_valid3(s0, s1, s2, l0, l1, l2)
                    if testval == checkval:
                        print("valid") if testval else print("invalid")
                    else:
                        print("FAIL: test says", testval)
                        fails.append([l0, l1, l2, s0, s1, s2])
                        if testval:
                            cols = collisions3(s0, s1, s2, l0, l1, l2)
                            print(cols)
                        else:
                            exit(1)
                        exit(1)
    if nextl == 0:
        l0 += 1
    if nextl == 1:
        l1 += 1
    if nextl == 2:
        l2 += 1
    nextl = (nextl + 1 ) % 3
                                
print("fails:", len(fails))
print(fails)

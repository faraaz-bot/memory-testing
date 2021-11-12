#!/usr/bin/python3

import math
import sys

from valid import *

lmax = 10

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


    
def is_valid3(s0, s1, s2, l0, l1, l2):
    if not is_valid2(s0, s1, l0, l1):
        return False
    if not is_valid2(s0, s2, l0, l2):
        return False
    if not is_valid2(s1, s2, l1, l2):
        return False

    # This isn't actually enough.
    
    return True


    

for l0 in range(2, lmax):
    for l1 in range(2, lmax):
        for l2 in range(2, lmax):
            for s0 in range(1, 2 * l0 + 2):
                for s1 in range(1, 2 * l1 +2 ):
                    for s2 in range(1, 2 * l2 +2 ):
                        testval = is_valid3(s0, s1, s2, l0, l1, l2)
                        checkval = check_valid3(s0, s1, s2, l0, l1, l2)
                        if testval == checkval:
                            print("yay!")
                        else:
                            print("FAIL")

#!/usr/bin/python3

def lcm(x, y):
    from math import gcd
    return x * y // gcd(x, y)

s0 = 3
s1 = 5

l0 = 6
l1 = 4


print(lcm(s0, s1))

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

cols = collisions2(s0, s1, l0, l1)
print(cols)

def is_valid2(s0, s1, l0, l1):
    c = lcm(s0, s1)
    if ((l0+1) < c // s0) and ((l1+1) < c // s1):
        return True
    if (s0 * (l0 -1) + s1 * (l1 -1)) < c:
        return True
    return False

print(is_valid2(s0, s1, l0, l1))



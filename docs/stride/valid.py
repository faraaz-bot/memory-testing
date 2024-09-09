
import math
def lcm(x, y):
    return x * y // math.gcd(x, y)

def is_valid2(s, l):

    # The indices need to be different:
    if s[0] == s[1]:
        return False
    c = lcm(s[0], s[1])
    #print("c:", c)

    # If the lengths are too short to get to the lcm individually,
    # we're ok:
    if not ((s[0] * (l[0] - 1) >= c) and (s[1] * (l[1] - 1) >= c)):
        return True

    return False    

def perm(a):
    b = []
    for i in range(len(a)):
        b.append( a[(i + 1) % len(a)])
    return b


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
    
def is_valid3(s, l):
    if s[0] == s[1] or s[0] == s[2] or s[1] == s[2]:
        return False
    
    if not is_valid2([s[0], s[1]], [l[0], l[1]]):
        return False
    if not is_valid2([s[0], s[2]], [l[0], l[2]]):
        return False
    if not is_valid2([s[1], s[2]], [l[1], l[2]]):
        return False

    sl = [[s[0], l[0]], [s[1], l[1]], [s[2], l[2]]]
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


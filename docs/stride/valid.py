
import math
def lcm(x, y):
    return x * y // math.gcd(x, y)

def is_valid2(s0, s1, l0, l1):

    # The indices need to be different:
    if s0 == s1:
        return False
    c = lcm(s0, s1)
    #print("c:", c)

    # If the lengths are too short to get to the lcm individually,
    # we're ok:
    if not ((s0 * (l0 - 1) >= c) and (s1 * (l1 - 1) >= c)):
        return True

    # if (s0 * (l0 - 1) <= c) and (s1 * (l1 - 1) <= c):
    #     return True
    return False    
    
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

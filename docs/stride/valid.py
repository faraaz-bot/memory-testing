
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

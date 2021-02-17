
from pathlib import Path as path
import pandas as pd


def sjoin(s):
    return ' '.join(list(s))


def njoin(s):
    return '\n'.join(list(s))


def cjoin(s):
    return ','.join(list((map(str, s))))


# remove this at some point and then remove "import pandas" above
def write_csv(fname, results):
    if results:
        df = pd.DataFrame(sum(results, []))
        df.to_csv(fname, index=False)


def write_dat(fname, length, nbatch, seconds):
    d = [ len(length) ] + length + [ nbatch, len(seconds) ] + seconds
    t = '\t'.join([str(x) for x in d]) + '\n'
    with path(fname).open('a') as f:
        f.write(t)

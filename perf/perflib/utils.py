"""A few small utilities."""

from pathlib import Path


def join(sep, s):
    """Return 's' joined with 'sep'.  Coerces to str."""
    return sep.join(str(x) for x in list(s))


def sjoin(s):
    """Return 's' joined with spaces."""
    return join(' ', s)


def njoin(s):
    """Return 's' joined with newlines."""
    return join('\n', s)


def cjoin(s):
    """Return 's' joined with commas."""
    return join(',', s)

def tjoin(s):
    """Return 's' joined with tabs."""
    return join('\t', s)


def write_dat(fname, length, nbatch, seconds, title=None):
    """Append record to .dat file."""
    path = Path(fname)
    if not path.exists():
        dat = ['# title: ' + title]
    else:
        dat = path.read_text().splitlines()

    if isinstance(length, int):
        length = [length]

    record = [ len(length) ] + list(length) + [ nbatch, len(seconds) ] + seconds
    dat.append(tjoin(record))

    path.write_text(njoin(dat) + '\n')

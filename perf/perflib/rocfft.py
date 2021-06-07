
import logging
import re
import subprocess

import perflib.utils

from typing import List, Any
from dataclasses import dataclass

@dataclass
class RIDERFFTTestRunner:
    label: str = 'UNKNOWN'
    transform: Any = None
    lengths: List[Any] = list
    ntrials: int = 1
    nbatch: int = 1
    dtype: Any = None
    rider: Any = None
    builds: List[Any] = list

    def dyna_rider(self, length):
        cmd = [self.rider,
               '--lib', self.builds[0] / 'lib' / 'librocfft.so',
               '--lib', self.builds[1] / 'lib' / 'librocfft.so' ]
        cmd += ['--length']
        if isinstance(length, int):
            cmd += [length]
        else:
            cmd += length
        cmd += ['-N', self.ntrials]
        cmd += ['-b', self.nbatch]
        cmd += self.transform.rider
        cmd += self.dtype.rider
        cmd = [str(x) for x in cmd]

        logging.info('DYNA: ' + perflib.utils.sjoin(cmd))
        stdout = subprocess.check_output(cmd, universal_newlines=True)

        results = []
        for i, m in enumerate(re.finditer('Execution gpu time: ([ 0-9.]*) ms', stdout, re.MULTILINE)):
            times = list(map(float, m.group(1).split(' ')))
            for t in times:
                results.append({'n': length, 'method': 'dyna-rider', 'time': t, 'build': self.builds[i]})
        return results

    def run(self, progress=True):
        timings = []
        for length in self.lengths:
            timings.extend(self.dyna_rider(length))
            if progress:
                print('.', end='', flush=True)
        return timings

    def write(self, dname, fname, results, title=None):
        # XXX better names for libraries
        for build in self.builds:
            dir = dname / str(build)
            dir.mkdir(exist_ok=True)
            for length in self.lengths:
                seconds = [ t['time'] for t in results if t['n'] == length and t['build'] == build ]
                perflib.utils.write_dat(dir / fname, length, self.nbatch, seconds, title=title)

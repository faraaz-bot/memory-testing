#!/usr/bin/env python3
"""Fabric file to launch timing runs.

Install: pip3 install fabric

To run the mi100 runs:

- setup: fab --hosts=mi100-host-name setup
- build: fab --hosts=mi100-host-name build-rocfft
- run:   fab --hosts=mi100-host-name run-rocfft --device=mi100

To run the a100 runs:

- setup: fab --hosts=a100-host-name setup
- build: fab --hosts=a100-host-name build-docker
- build: fab --hosts=a100-host-name build-hipfft
- run:   fab --hosts=a100-host-name run-hipfft --device=a100

To follow along, add '--echo' to any of the above.

"""

from fabric import task

import dataclasses
import os
import pathlib
import subprocess


@dataclasses.dataclass
class Run:
    tag: str
    device: str
    rider: str
    suite: str


@dataclasses.dataclass
class Repo:
    url: str
    commit: str

    @property
    def name(self):
        name = self.url.split(':')[1]
        name = name.split('/')[1]
        return name.split('-')[0]

    @property
    def path(self):
        return pathlib.Path(self.name + '-' + self.commit)


#
# configuration
#

user       = os.getenv('USER')
comparison = 'update2021'
suite_file = 'docs/update2021/s21.py'

rocfft  = Repo('git@github.com:memmett/rocFFT-internal.git', '122f348')
hipfft  = Repo('git@github.com:ROCmSoftwarePlatform/hipFFT-internal.git', '9b150b0')
fftperf = Repo('git@github.com:memmett/rocFFT-internal.git', '122f348')
#fftperf = Repo('git@github.com:ROCmSoftwarePlatform/rocFFT-internal.git', '0a6f539')
fftmisc = Repo('git@github.com:ROCmSoftwarePlatform/rocFFT-misc.git', 'f055e4a')

runs = [
    Run('rocfft-mi100-md',     'mi100', 'rocfft', 's21:md1'),
    Run('rocfft-mi100-cholla', 'mi100', 'rocfft', 's21:cholla1d'),
    Run('rocfft-mi200-md',     'mi200', 'rocfft', 's21:md1'),
    Run('rocfft-mi200-cholla', 'mi200', 'rocfft', 's21:cholla1d'),
    Run('cufft-v100-md',        'v100', 'hipfft', 's21:md1'),
    Run('cufft-v100-cholla',    'v100', 'hipfft', 's21:cholla1d'),
    Run('cufft-a100-md',        'a100', 'hipfft', 's21:md1'),
    Run('cufft-a100-cholla',    'a100', 'hipfft', 's21:cholla1d'),
]

#
# shortcuts
#

rocfft_perf  = fftperf.path / 'scripts' / 'perf' / 'rocfft-perf'
rocfft_rider = rocfft.path / 'build' / 'clients' / 'staging' / 'rocfft-rider'
hipfft_rider = hipfft.path / 'build' / 'clients' / 'staging' / 'hipfft-rider'

#
# helpers
#

def remote_exists(c, fname):
    """Test whether 'fname' exists on the remote host."""
    return c.run(f'test -e {fname}', hide=True, warn=True).exited == 0


def docker(command, workdir='/x'):
    """Return 'docker run' command to run 'command' in 'workdir'."""
    return f'docker run --rm --gpus all --volume /home/{user}:/x --workdir {workdir} rocfft-cuda-perf {command}'


def pull(c, dname):
    """Sync directory 'dname' from remote host."""
    subprocess.run(['rsync', '-a', c.host + ':' + comparison + '/' + dname + '/', dname + '/'])


#
# tasks
#

@task
def setup(c):
    """Setup 'comparison' directory with necessary repos."""

    if not remote_exists(c, comparison):
        c.run(f'mkdir {comparison}')

    with c.cd(comparison):
        for repo in [fftperf, fftmisc]:
            if not remote_exists(c, repo.path):
                c.run(f'git clone {repo.url} {repo.path}')
            with c.cd(repo.path):
                c.run(f'git checkout {repo.commit}')

        spath = fftmisc.path / suite_file
        sname = spath.name
        c.run(f'cp {spath} {sname}')


@task
def build_rocfft(c):
    """Build rocFFT rider (native)."""
    with c.cd(comparison):
        if not remote_exists(c, rocfft.path):
            c.run(f'git clone {rocfft.url} {rocfft.path}')

        with c.cd(rocfft.path):
            c.run(f'git checkout {rocfft.commit}')
            c.run('mkdir -p build')
            with c.cd('build'):
                sqlite = '-DSQLITE_SRC_URL=file:///home/maemmett/sqlite-amalgamation-3360000.zip'  # XXX
                c.run(f'cmake -G Ninja -DCMAKE_CXX_COMPILER=hipcc -DBUILD_CLIENTS_RIDER=ON -DAMDGPU_TARGETS= {sqlite} ..')
                c.run('ninja')


@task
def run_rocfft(c, device):
    """Run rocFFT runs."""
    rruns = [x for x in runs if x.rider == 'rocfft' and x.device == device]
    with c.cd(comparison):
        for run in rruns:
            c.run(f'rm -rf {run.tag}')
            c.run(f'{rocfft_perf} run --rider {rocfft_rider} --suite {run.suite} --out {run.tag}')
            pull(c, run.tag)


@task
def build_docker(c):
    """Build CUDA docker image."""
    with c.cd(pathlib.Path(comparison) / 'rocFFT-misc' / 'perf' / 'docker'):
        c.run('docker build -t rocfft-cuda-perf .')


@task
def build_hipfft(c):
    """Build hipFFT rider (docker)."""
    with c.cd(comparison):
        if not remote_exists(c, hipfft.path):
            c.run(f'git clone {hipfft.url} {hipfft.path}')

        with c.cd(hipfft.path):
            c.run(f'git checkout {hipfft.commit}')
            c.run('mkdir -p build')

            build = f'/x/{comparison}/{hipfft.path}/build'
            c.run(docker('cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -DBUILD_WITH_LIB=CUDA ..', workdir=build))
            c.run(docker('make', workdir=build))


@task
def run_hipfft(c, device):
    """Run hipFFT runs."""
    rruns = [x for x in runs if x.rider == 'hipfft' and x.device == device]
    with c.cd(comparison):
        for run in rruns:
            work = f'/x/{comparison}'
            c.run(docker(f'rm -rf {run.tag}', workdir=work))
            c.run(docker(f'{rocfft_perf} run --rider {hipfft_rider} --suite {run.suite} --out {run.tag}', workdir=work))
            pull(c, run.tag)

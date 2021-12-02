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

import collections
import os
import pathlib
import subprocess

#
# configuration
#

comparison = 'update2021'
suite_file = 'rocFFT-misc/docs/update2021/s21.py'

rocfft_repo        = 'git@github.com:memmett/rocFFT-internal.git'
rocfft_commit      = 'cholla1d-optimize'
# rocfft_repo        = 'git@github.com:ROCmSoftwarePlatform/rocFFT-internal.git'
# rocfft_commit      = 'develop'
hipfft_repo        = 'git@github.com:ROCmSoftwarePlatform/hipFFT-internal.git'
hipfft_commit      = 'develop'
rocfft_perf_repo   = 'git@github.com:ROCmSoftwarePlatform/rocFFT-internal.git'
rocfft_misc_repo   = 'git@github.com:ROCmSoftwarePlatform/rocFFT-misc.git'

Run = collections.namedtuple('Run', ['tag', 'device', 'rider', 'suite'])

user = os.getenv('USER')

runs = [
    Run('rocfft-mi100-md', 'mi100', 'rocfft', 's21:md1'),
    Run('rocfft-mi100-cholla', 'mi100', 'rocfft', 's21:cholla1d'),
    Run('rocfft-mi200-md', 'mi200', 'rocfft', 's21:md1'),
    Run('rocfft-mi200-cholla', 'mi200', 'rocfft', 's21:cholla1d'),
    Run('cufft-v100-md', 'v100', 'hipfft', 's21:md1'),
    Run('cufft-v100-cholla', 'v100', 'hipfft', 's21:cholla1d'),
    Run('cufft-a100-md', 'a100', 'hipfft', 's21:md1'),
    Run('cufft-a100-cholla', 'a100', 'hipfft', 's21:cholla1d'),
]

#
# shortcuts
#

rocfft_perf  = pathlib.Path('rocFFT-internal') / 'scripts' / 'perf' / 'rocfft-perf'
hipfft_perf  = pathlib.Path('rocFFT-internal') / 'scripts' / 'perf' / 'rocfft-perf'
rocfft_rider = pathlib.Path(f'rocFFT-{rocfft_commit}') / 'build' / 'clients' / 'staging' / 'rocfft-rider'
hipfft_rider = pathlib.Path(f'hipFFT-{hipfft_commit}') / 'build' / 'clients' / 'staging' / 'hipfft-rider'

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
    """Pull directory 'dname' from remote host."""
    s = c.sftp()
    s.chdir(comparison)
    d = pathlib.Path(dname)
    d.mkdir(exist_ok=True)
    for f in s.listdir(dname):
        fname = str(d / f)
        s.get(fname, fname)


#
# tasks
#

@task
def setup(c):
    """Setup 'comparison' directory with necessary repos."""

    if not remote_exists(c, comparison):
        c.run(f'mkdir {comparison}')

    with c.cd(comparison):
        if not remote_exists(c, 'rocFFT-internal'):
            c.run(f'git clone {rocfft_perf_repo}')
        else:
            with c.cd('rocFFT-internal'):
                c.run('git pull')

        if not remote_exists(c, 'rocFFT-misc'):
            c.run(f'git clone {rocfft_misc_repo}')
        else:
            with c.cd('rocFFT-misc'):
                c.run('git pull')

        spath = pathlib.Path(suite_file)
        sname = spath.name
        c.run(f'cp {spath} {sname}')


@task
def build_rocfft(c):
    """Build rocFFT rider (native)."""
    with c.cd(comparison):
        if not remote_exists(c, f'rocFFT-{rocfft_commit}'):
            c.run(f'git clone {rocfft_repo} rocFFT-{rocfft_commit}')

        with c.cd(f'rocFFT-{rocfft_commit}'):
            c.run(f'git checkout {rocfft_commit}')
            c.run('git pull')
            c.run('mkdir -p build')
            with c.cd('build'):
                sqlite = '-DSQLITE_SRC_URL=file:///home/maemmett/sqlite-amalgamation-3360000.zip'
                c.run(f'cmake -G Ninja -DCMAKE_CXX_COMPILER=hipcc -DBUILD_CLIENTS_RIDER=ON -DAMDGPU_TARGETS= {sqlite} ..')
                c.run('ninja')


@task
def run_rocfft(c, device):
    """Run rocFFT runs."""
    rruns = [x for x in runs if x.rider == 'rocfft' and x.device == device]
    with c.cd(comparison):
        for tag, _, rider, suite in rruns:
            c.run(f'rm -rf {tag}')
            c.run(f'{rocfft_perf} run --rider {rocfft_rider} --suite {suite} --out {tag}')
            pull(c, tag)


@task
def build_docker(c):
    """Build CUDA docker image."""
    with c.cd(pathlib.Path(comparison) / 'rocFFT-misc' / 'perf' / 'docker'):
        c.run('docker build -t rocfft-cuda-perf .')


@task
def build_hipfft(c):
    """Build hipFFT rider (docker)."""
    with c.cd(comparison):
        if not remote_exists(c, f'hipFFT-{hipfft_commit}'):
            c.run(f'git clone {hipfft_repo} hipFFT-{hipfft_commit}')

        with c.cd(f'hipFFT-{hipfft_commit}'):
            c.run(f'git checkout {hipfft_commit}')
            c.run('git pull')
            c.run('mkdir -p build')

            build = f'/x/{comparison}/hipFFT-{hipfft_commit}/build'
            c.run(docker('cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -DBUILD_WITH_LIB=CUDA ..', workdir=build))
            c.run(docker('make', workdir=build))


@task
def run_hipfft(c, device):
    """Run hipFFT runs."""
    rruns = [x for x in runs if x.rider == 'hipfft' and x.device == device]
    with c.cd(comparison):
        for tag, _, rider, suite in rruns:
            work = f'/x/{comparison}'
            c.run(docker(f'rm -rf {tag}', workdir=work))
            c.run(docker(f'{hipfft_perf} run --rider {hipfft_rider} --suite {suite} --out {tag}', workdir=work))
            pull(c, tag)

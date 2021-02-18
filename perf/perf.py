#!/usr/bin/env python3
'''Performance tracker for rocFFT vs cuFFT.'''

import click
import rich.console
import rich.traceback

import numpy as np
import os
import sys
import subprocess

from pathlib import Path as path
from types import SimpleNamespace as NS

top = path(__file__).resolve().parent
sys.path.append(str(top))

import perflib.utils
import perflib.git as git

console = rich.console.Console()
rich.traceback.install()

#
# build
#

env = NS()
env.docker = False

def sjoin(s):
    '''Join `s` with spaces.'''
    return ' '.join(list(s))


def local(cmd, echo=True, **kwargs):
    '''Run `cmd` using the shell.

    Keyword arguments are passed down to `subprocess.run`.
    '''

    if echo:
        console.print('[magenta]local:[/magenta] ' + cmd)
    return subprocess.run(cmd, shell=True, **kwargs)



def build_rocfft(commit, dest=None):
    '''Build public rocFFT (at specified git `commit`) and install into `dest`.'''

    top = path('.').resolve() / ('rocFFT-' + commit)

    if not top.exists():
        git.clone('git@github.com:ROCmSoftwarePlatform/rocFFT.git', top)
    git.checkout(top, commit)

    if git.is_dirty(top):
        print(f'working directory {top} is dirty!')
        raise SystemExit

    build = top / 'build'
    build.mkdir(exist_ok=True)
    defs = [ '-DCMAKE_CXX_COMPILER=hipcc',
             '-DSINGLELIB=ON',
             '-BUILD_CLIENTS_RIDER=ON',
             '-DAMDGPU_TARGETS=' ]
    if dest:
        defs += [ f'-DCMAKE_INSTALL_PREFIX={dest}' ]

    local(f'cmake {sjoin(defs)} ..', cwd=build, check=True)
    local('make -j 8', cwd=build, check=True)

    if dest:
        local('make install', cwd=build, check=True)


def build_hipfft(commit, dest, cuda):
    '''Build public hipFFT (at specified git `commit`) and install into `dest`.'''

    top = path('.').resolve() / ('hipFFT-' + commit)

    if not top.exists():
        git.clone('git@github.com:ROCmSoftwarePlatform/hipFFT-internal.git', top)
    git.checkout(top, commit)

    if git.is_dirty(top):
        print(f'working directory {top} is dirty!')
        raise SystemExit

    build = top / 'build'
    build.mkdir(exist_ok=True)
    if cuda:
        defs = [ '-DBUILD_WITH_LIB=CUDA',
                 f'-DCMAKE_PREFIX_PATH={dest}',
                 f'-DCMAKE_INSTALL_PREFIX={dest}' ]
    else:
        defs = [ '-DCMAKE_CXX_COMPILER=hipcc',
                 f'-DCMAKE_PREFIX_PATH={dest}',
                 f'-DCMAKE_INSTALL_PREFIX={dest}' ]
    local(f'cmake {sjoin(defs)} ..', cwd=build, check=True)
    local('make -j 8 install', cwd=build, check=True)


def build_wrapper(dest, cuda):
    '''Build Python hipFFT wrapper from hipFFT version installed in `dest`.'''
    from sysconfig import get_paths

    includes = [ get_paths()['include'], np.get_include(), dest / 'hipfft' / 'include' ]
    includes = sjoin([ '-I' + str(x) for x in includes ])

    libs = [ dest / 'lib' ]
    libs = sjoin([ '-L' + str(x) for x in libs ])

    src = top.parent / 'pyhipfft' / 'pyhipfft.cpp'
    so  = dest / 'hipfft.so'
    defs = None
    if cuda:
        defs = sjoin([ '-Xcompiler', '-fPIC', '-std=c++14', '-shared' ])
    else:
        defs = sjoin([ '-fPIC', '-std=c++14', '-shared' ])
    local(f'hipcc {defs} {includes} {src} -o {so} {libs} -lhipfft', check=True)


#
# command line interface
#

@click.group()
def cli():
    pass


@cli.command()
@click.option('--rocfft', type=str, default=None, help='rocFFT git branch/tag/commit.')
@click.option('--hipfft', type=str, default=None, help='hipFFT git branch/tag/commit.')
@click.option('--cuda', type=bool, default=False, is_flag=True, help='Use CUDA backend for hipFFT.')
@click.option('--destination', type=str, default=None, help='Destination directory for builds.')
def build(destination, hipfft, rocfft, cuda):
    '''Clone and build rocFFT and/or hipFFT.

    Builds are installed into the 'build' directory.  All shared
    library RPATHs are set using 'patchelf'.

    '''

    build = path('.').resolve()
    if destination is None:
        build = build / 'build'
    else:
        build = build / destination

    build.mkdir(exist_ok=True)

    if rocfft:
        build_rocfft(rocfft, build)

    if hipfft:
        build_hipfft(hipfft, build, cuda)
        build_wrapper(build, cuda)

    libdir = build / 'lib'
    rpath = []
    if cuda:
        rpath.append(os.getenv('CUDA_PATH', '/usr/local/cuda') + '/lib64')
    rpath.append(str(libdir))
    rpath.append('/opt/rocm/lib')
    for lib in list(libdir.glob('*.so*')) + [ build / 'hipfft.so' ]:
        local(f'patchelf --set-rpath {":".join(rpath)} {lib}', check=True)


def load_suite(suite):

    tdef = top / 'performance-tests.py'
    code = compile(tdef.read_text(), str(tdef), 'exec')
    ns = {}
    exec(code, ns)
    return ns[suite]

# XXX add a (mandatory) argument to run to specify an output directory
# XXX save machine specs into the output directory as well

@cli.command()
@click.option('--ntrials', type=int, default=10, help='Number of trials (default 10).')
@click.option('--verify', type=bool, default=False, is_flag=True, help='Verify results (default False).')
@click.option('--suite', type=str, default='all', help='Test suite name (generator in performance-tests.py, default "all").')
@click.option('--build', type=str, default='build', help='Build directory to use libraries from.')
def run(ntrials, verify, suite, build):
    '''Run performance tests using a single build.

    Tests are loaded from 'performance-tests.py'.

    Tests are performed using the libraries installed in the 'build'
    directory.

    '''

    sys.path.insert(0, build)
    import hipfft
    console.print(f'Using hipfft wrapper: [red]{hipfft.__file__}[/red]')

    generator = load_suite(suite)

    for test in generator(ntrials, verify):
        console.print(f'[green]# running {test.label}[/green]')
        test.run()
        test.write(test.label + '.dat')


if __name__ == '__main__':
    cli()

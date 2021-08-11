#!/usr/bin/env python3
'''Performance tracker for rocFFT vs cuFFT.'''

import click

import logging
import numpy as np
import os
import scipy.stats
import statistics
import subprocess
import sys
import types

from pathlib import Path as path

top = path(__file__).resolve().parent
sys.path.append(str(top))

import perflib.utils
import perflib.git as git

from perflib.utils import sjoin


#
# build
#

def local(cmd, echo=True, **kwargs):
    """Run `cmd` using the shell.

    Keyword arguments are passed down to `subprocess.run`.
    """
    if echo:
        print('local: ' + cmd)
    logging.info('local: ' + cmd)
    return subprocess.run(cmd, shell=True, **kwargs)


def build_rocfft(commit, dest=None, repo='git@github.com:ROCmSoftwarePlatform/rocFFT-internal.git', ccache=False):
    """Build public rocFFT (at specified git `commit`) and install into `dest`."""

    top = path('.').resolve() / ('rocFFT-' + commit)

    if not top.exists():
        git.clone(repo, top)
    git.checkout(top, commit)

    if git.is_dirty(top):
        print(f'working directory {top} is dirty!')
        raise SystemExit

    build = top / 'build'
    build.mkdir(exist_ok=True)
    defs = ['-DCMAKE_CXX_COMPILER=hipcc',
            '-DBUILD_CLIENTS_RIDER=ON',
            '-DROCFFT_CALLBACKS_ENABLED=OFF',
            '-DSINGLELIB=ON',
            '-DAMDGPU_TARGETS=']
    if dest:
        defs += [f'-DCMAKE_INSTALL_PREFIX={dest}']
    if ccache:
        defs += ['-DCMAKE_CXX_COMPILER_LAUNCHER=ccache']

    local(f'cmake {sjoin(defs)} ..', cwd=build, check=True)
    local('make -j $(nproc)', cwd=build, check=True)

    if dest:
        local('make install', cwd=build, check=True)
        local(f'cp {build}/clients/staging/* {dest}')


def build_hipfft(commit, dest, cuda, repo='git@github.com:ROCmSoftwarePlatform/hipFFT-internal.git'):
    """Build public hipFFT (at specified git `commit`) and install into `dest`."""

    top = path('.').resolve() / ('hipFFT-' + commit)

    if not top.exists():
        git.clone(repo, top)
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
    """Build Python hipFFT wrapper from hipFFT version installed in `dest`."""
    from sysconfig import get_paths

    includes = [ get_paths()['include'], np.get_include(), dest / 'hipfft' / 'include' ]
    includes = sjoin([ '-I' + str(x) for x in includes ])

    libs = [ dest / 'lib' ]
    libs = sjoin([ '-L' + str(x) for x in libs ])

    src = top.parent / 'pyhipfft' / 'pyhipfft.cpp'
    so  = dest / 'hipfft.so'
    defs = None
    if cuda:
        defs = ['-Xcompiler', '-fPIC', '-std=c++14', '-shared']
    else:
        defs = ['-fPIC', '-std=c++14', '-shared']

        hdr = dest / 'hipfft' / 'include' / 'hipfft.h'
        have_meta = 'hipfftGetMeta' in hdr.read_text()
        if have_meta:
            defs += ['-DHAVE_HIPFFT_META']

    local(f'hipcc {sjoin(defs)} {includes} {src} -o {so} {libs} -lhipfft', check=True)


#
# command line interface
#

@click.group()
def cli():
    pass


@cli.command()
@click.option('--rocfft', type=str, default=None, help='rocFFT git branch/tag/commit.')
@click.option('--hipfft', type=str, default=None, help='hipFFT git branch/tag/commit.')
@click.option('--ccache', type=bool, default=False, is_flag=True, help='Use ccache when building rocFFT.')
@click.option('--cuda', type=bool, default=False, is_flag=True, help='Use CUDA backend for hipFFT.')
@click.option('--destination', type=str, default=None, help='Destination directory for builds.')
@click.option('--copy-hipfft-from', type=str, default=None, help='Copy hipFFT and wrapper from...')
@click.option('--user', type=str, default='ROCmSoftwarePlatform', help='Git user')
def build(destination, hipfft, rocfft, copy_hipfft_from, cuda, ccache, user):
    """Clone and build rocFFT and/or hipFFT.

    Builds are installed into the 'build' directory.  All shared
    library RPATHs are set using 'patchelf'.

    """

    build = path('.').resolve()
    if destination is None:
        build = build / 'build'
    else:
        build = build / destination

    build.mkdir(exist_ok=True)

    if rocfft:
        build_rocfft(rocfft, build, repo=f'git@github.com:{user}/rocFFT-internal.git', ccache=ccache)

    if hipfft:
        build_hipfft(hipfft, build, cuda, repo=f'git@github.com:{user}/hipFFT-internal.git')
        build_wrapper(build, cuda)

    if copy_hipfft_from:
        src = path(copy_hipfft_from)
        libbld = build / 'lib'
        libsrc = src / 'lib'
        for lib in libsrc.glob('**/*hipfft.so*'):
            local(f'cp {lib} {libbld}')
        wrapper = src / 'hipfft.so'
        if wrapper.exists():
            local(f'cp {wrapper} {build}')


    libdir = build / 'lib'
    rpath = []
    if cuda:
        rpath.append(os.getenv('CUDA_PATH', '/usr/local/cuda') + '/lib64')
    rpath.append(str(libdir))
    rpath.append('/opt/rocm/lib')
    for lib in build.glob('**/*.so*'):
        if not lib.is_symlink():
            local(f'patchelf --set-rpath {":".join(rpath)} {lib}', check=True)


def load_suite(suite):
    """Load performance suite from performance-tests.py."""

    tdef = top / 'performance-tests.py'
    code = compile(tdef.read_text(), str(tdef), 'exec')
    ns = {}
    exec(code, ns)
    s = ns[suite]
    if isinstance(s, types.FunctionType):
        return s
    return ns['make_suite'](s)


def run_dyna(build1, build2, ntrials, suite, output):
    """Compare performance of two builds using dynamic library loading."""

    from perflib.rocfft import RIDERFFTTestRunner as TestRunner

    generator = load_suite(suite)

    output = path(output)
    output.mkdir(exist_ok=True)

    for build in [build1, build2]:
        specs = output / build / 'specs.txt'
        specs.parent.mkdir(exist_ok=True)
        specs.write_text(str(perflib.get_machine_specs(0)))

    dino = path(build1) / 'dyna-rocfft-rider'

    for test in generator():
        runner = TestRunner(**test,
                            rider=dino, ntrials=ntrials,
                            builds=[path(build1), path(build2)])
        print(f'# running {runner.label}')
        results = runner.run()
        runner.write(output, runner.label + '.dat', results, title=runner.label)



@cli.command()
@click.option('--ntrials', type=int, default=10, help='Number of trials (default 10).')
@click.option('--verify', type=bool, default=False, is_flag=True, help='Verify results (default False).')
@click.option('--suite', type=str, default='all', help='Test suite name (generator in performance-tests.py, default "all").')
@click.option('--build', type=str, default='build', help='Build directory to use libraries from.')
@click.option('--output', type=str, default='.', help='Output directory to save results in.')
@click.option('--use-vkfft', type=str, default=False, is_flag=True, help='Use vkFFT.')
def run(ntrials, verify, suite, build, output, use_vkfft):
    """Run performance tests using a single build.

    Tests are loaded from 'performance-tests.py'.

    Tests are performed using the hipFFT libraries installed in the
    'build' directory unless '--use-vkfft' is set.

    """

    if not use_vkfft:
        sys.path.insert(0, build)
        import hipfft
        from perflib.hipfft import HIPFFTTestRunner as TestRunner
        print(f'Using hipFFT wrapper: {hipfft.__file__}')
    else:
        from perflib.vkfft import VKFFTTestRunner as TestRunner
        print('Using vkFFT')

    generator = load_suite(suite)

    output = path(output)
    output.mkdir(exist_ok=True)

    specs = output / 'specs.txt'
    specs.write_text(str(perflib.get_machine_specs(0)))

    for test in generator():
        runner = TestRunner(**test, ntrials=ntrials, verify=verify)
        print(f'# running {runner.label}')
        results = runner.run()
        runner.write(output, runner.label + '.dat', results, title=runner.label)

@cli.command()
@click.option('--suite', type=str, default='all', help='Test suite name (generator in performance-tests.py, default "all").')
def list(suite):
    generator = load_suite(suite)
    for test in generator():
        print(test)


@cli.command()
@click.argument('build1', type=str)
@click.argument('build2', type=str)
@click.option('--ntrials', type=int, default=10, help='Number of trials (default 10).')
@click.option('--suite', type=str, default='all', help='Test suite name (generator in performance-tests.py, default "all").')
@click.option('--output', type=str, default='.', help='Output directory to save results in.')
def dyna(build1, build2, ntrials, suite, output):
    """Compare performance of two builds using dynamic library loading."""
    run_dyna(build1, build2, ntrials, suite, output)


@cli.command()
@click.option('--host', type=str, default=None)
@click.option('--workdir', type=str, default=None)
@click.option('--reference-branch', type=str, default=None)
@click.option('--reference-repository', type=str, default=None)
@click.option('--branch', type=str, default=None)
@click.option('--repository', type=str, default=None)
@click.option('--suite', type=str, default=None)
@click.option('--user', type=str, envvar='PERF_USER')
def autodyna(host, workdir, reference_branch, reference_repository, branch, repository, suite, user):
    """Compare performance of two builds automagically."""

    if host is None:
        host = click.prompt('Host', default='localhost', type=str)

    if reference_branch is None:
        reference_branch = click.prompt('Reference branch', default='develop', type=str)

    if reference_repository is None:
        reference_repository = click.prompt('Reference repo',
                                            default='git@github.com:ROCmSoftwarePlatform/rocFFT-internal.git', type=str)

    if branch is None:
        branch = click.prompt('PR branch', type=str)

    if repository is None:
        repository = f'git@github.com:{user}/rocFFT-internal.git'
        repository = click.prompt('PR repo', default=repository, type=str)

    if suite is None:
        suite = click.prompt('Test suite', default='all', type=str)

    if workdir is None:
        workdir = click.prompt('Working directory', default='autodyna-' + branch, type=str)

    if host != 'localhost':
        cmd = ['ssh', host, 'nohup', 'perf', 'autodyna']
        cmd += ['--host', 'localhost']
        cmd += ['--workdir', workdir]
        cmd += ['--reference-branch', reference_branch]
        cmd += ['--reference-repository', reference_repository]
        cmd += ['--branch', branch]
        cmd += ['--repository', repository]
        cmd += ['--suite', suite]
        local(sjoin(cmd))
        return

    top = path(workdir).resolve()
    build1  = top / f'build-{reference_branch}'
    build2  = top / f'build-{branch}'
    output  = top / f'dyna-{branch}'

    os.chdir(str(top))

    lib1 = build1 / 'lib' / 'librocfft.so'
    lib1.parent.mkdir(parents=True, exist_ok=True)
    if not lib1.exists():
        build_rocfft(reference_branch, dest=build1, repo=reference_repository, ccache=True)

    lib2 = build2 / 'lib' / 'librocfft.so'
    lib2.parent.mkdir(parents=True, exist_ok=True)
    if not lib2.exists():
        build_rocfft(branch, dest=build2, repo=repository, ccache=True)

    run_dyna(build1, build2, ntrials=10, suite=suite, output=output)



@cli.command()
def specs():
    """Print machine specs."""
    print(perflib.specs.get_machine_specs(0))

@cli.command()
@click.argument('runs', type=str, nargs=-1)
@click.option('--moods', type=float, default=0.05, help="Threshold for Mood's p-value reporting.")
@click.option('--percent', type=float, default=0.0, help="Threshold for median-time percent difference reporting.")
def moods(runs, percent, moods):
    """ """
    base = path(runs[0])

    regressions = []

    for dname in sorted(base.glob('**/*.dat')):
        reference_samples = perflib.utils.read_dat(dname)
        for run in runs[1:]:
            oname = path(run) / dname.name
            run_samples = perflib.utils.read_dat(oname)
            for length in reference_samples.keys():

                if length not in run_samples:
                    print(f"WARNING: length {length} missing from {oname}.")
                    continue
                if reference_samples[length].nbatch != run_samples[length].nbatch:
                    print(f"WARNING: length {length} batch counts differ from {oname}.")
                    continue

                s1 = reference_samples[length].times
                s2 = run_samples[length].times

                if not s1:
                    print(f"WARNING: missing samples for length {length} from {dname}.")
                    continue
                if not s2:
                    print(f"WARNING: missing samples for length {length} from {oname}.")
                    continue

                m1 = statistics.median(s1)
                m2 = statistics.median(s2)
                if m1 < m2 and abs(m1 - m2) / m1 > percent / 100.0:
                    _, p, _, _ = scipy.stats.median_test(s1, s2)
                    if p < moods:
                        diff = 100 * abs(m1 - m2) / m1
                        print(f"REGRESSION: length {str(length)}; median times {m1:.4f} vs {m2:.4f} ({diff:4.1f}%); Mood's p-value {p:.6f}; from {oname}.")
                        regressions.append(length)

    print("Regressions found in lengths:")
    for length in sorted(set(regressions), key=perflib.utils.product):
        print("--length " + perflib.utils.sjoin(length))


if __name__ == '__main__':
    logging.basicConfig(filename='perf.log',
                        format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)-8s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    cli()

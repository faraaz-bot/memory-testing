#!/usr/bin/env python3

MIN_SPEED_DELTA_PERCENT = 5.0

# Helper script for automatic git bisect.  Accepts a good/bad commit,
# ensures the good one is faster than the old one, then runs git
# bisect to figure out which commit made things slower.  Medians of
# the values reported by rider are used to make the faster/slower
# decision.
#
# Run like:
#   rocfft_autobisect.py
#     --url git@github.com:user/rocFFT.git
#     --good goodcommit
#     --bad badcommit
#     --dir /path/to/repo/dir
#     --cmake -DCMAKE_ARG_1=foo -DCMAKE_ARG_2=bar
#     --build ninja
#     --rider --length 12345 -N 20 -b 10 -t 3 -o
#
# Note: this script will remove and recreate the directory named by --dir.
#
# --build is optional and defaults to ninja.
#
# You almost certainly want to give a sensible AMDGPU_TARGETS value
# to cmake, and tell it to use ccache.

import subprocess
import sys
import statistics
import traceback
from enum import Enum
import shutil
import os
import time

def log(msg):
    print(time.strftime('rocfft_autobisect %H:%M:%S - ' + msg))

def build_and_run(root_dir, cmake_args, build_args, rider_args):
    # clean build dir if it exists
    build_dir = os.path.join(root_dir, 'build')
    shutil.rmtree(build_dir, ignore_errors=True)

    # create build dir
    os.mkdir(build_dir)

    # cmake
    log("cmake: " + ' '.join(cmake_args))
    subprocess.run(args=['cmake'] + cmake_args, check=True, capture_output=True, cwd=build_dir)

    # build
    log("building: " + ' '.join(build_args))
    subprocess.run(args=build_args, check=True, capture_output=True, cwd=build_dir)

    # run rider and get times out
    log("rider: " + ' '.join(rider_args))
    rider_path = os.path.join(build_dir, 'clients', 'staging', 'rocfft-rider')
    rider_proc = subprocess.Popen(args=[rider_path] + rider_args, stdout=subprocess.PIPE)

    for line in rider_proc.stdout:
        if b'Execution gpu time' not in line:
            continue
        # line should be like:
        # Execution gpu time: 0.1 0.1 0.1 ms
        times = [float(word) for word in line.split(b' ')[3:-1]]
        return statistics.median(times)
    # didn't find the line?
    raise RuntimeError('could not find results in output')

def is_faster_p(base, new):
    if new > base:
        return False
    delta = new / base
    return delta < (100.0 - MIN_SPEED_DELTA_PERCENT) / 100.0

def main():
    # cook up our own command line parser since argparse can't really
    # handle arbitrary --foo args separated by known --bar args
    url_args = []
    good_args = []
    bad_args = []
    dir_args = []
    cmake_args = []
    build_args = []
    rider_args = []

    class ParserState(Enum):
        INITIAL = 0,
        URL = 1,
        GOOD = 2,
        BAD = 3,
        DIR = 4,
        CMAKE = 5,
        BUILD = 6,
        RIDER = 7,
    state = ParserState.INITIAL
    for arg in sys.argv[1:]:
        if arg == '--url':
            state = ParserState.URL
        elif arg == '--good':
            state = ParserState.GOOD
        elif arg == '--bad':
            state = ParserState.BAD
        elif arg == '--dir':
            state = ParserState.DIR
        elif arg == '--cmake':
            state = ParserState.CMAKE
        elif arg == '--build':
            state = ParserState.BUILD
        elif arg == '--rider':
            state = ParserState.RIDER
        else:
            if state == ParserState.URL:
                url_args.append(arg)
            elif state == ParserState.GOOD:
                good_args.append(arg)
            elif state == ParserState.BAD:
                bad_args.append(arg)
            elif state == ParserState.DIR:
                dir_args.append(arg)
            elif state == ParserState.CMAKE:
                cmake_args.append(arg)
            elif state == ParserState.BUILD:
                build_args.append(arg)
            elif state == ParserState.RIDER:
                rider_args.append(arg)
            else:
                raise RuntimeError(f'invalid arg {arg}, state {state}')

    if len(url_args) != 1:
        raise RuntimeError('exactly 1 url required')
    if len(dir_args) != 1:
        raise RuntimeError('exactly 1 dir required')
    if len(good_args) != 1:
        raise RuntimeError('exactly 1 good commit required')
    if len(bad_args) != 1:
        raise RuntimeError('exactly 1 bad commit required')
    if len(cmake_args) == 0:
        raise RuntimeError('cmake args required')
    if len(build_args) == 0:
        # assume ninja
        build_args = ['ninja']
    if len(rider_args) == 0:
        raise RuntimeError('rider args required')

    url = url_args[0]
    root_dir = dir_args[0]
    good_commit = good_args[0]
    bad_commit = bad_args[0]

    # clean dir
    shutil.rmtree(root_dir, ignore_errors=True)

    # clone repo to dir
    subprocess.run(args=['git', 'clone', url, root_dir], check=True)

    # get median time for good commit
    log(f"building good commit {good_commit}")
    subprocess.run(args=['git', 'checkout', good_commit], check=True, capture_output=True, cwd=root_dir)
    good_median = build_and_run(root_dir, cmake_args, build_args, rider_args)

    # get median time for bad commit
    log(f"building bad commit {bad_commit}")
    subprocess.run(args=['git', 'checkout', bad_commit], check=True, capture_output=True, cwd=root_dir)
    bad_median = build_and_run(root_dir, cmake_args, build_args, rider_args)

    # if "bad" is better than "good", the commit range we started
    # with doesn't sound responsible
    if not is_faster_p(bad_median, good_median):
        log(f"good median {good_median} is not {MIN_SPEED_DELTA_PERCENT}% faster than bad median {bad_median}")
        sys.exit(1)

    log(f"good median {good_median}, bad median {bad_median}")

    # start bisecting
    subprocess.run(args=['git', 'bisect', 'start'], check=True, cwd=root_dir)
    subprocess.run(args=['git', 'bisect', 'good', good_commit], check=True, cwd=root_dir)
    subprocess.run(args=['git', 'bisect', 'bad', bad_commit], check=True, cwd=root_dir)

    while True:
        median = build_and_run(root_dir, cmake_args, build_args, rider_args)
        bisect_verb = 'good' if is_faster_p(bad_median, median) else 'bad'
        log(f'commit is {bisect_verb} (this={median} good={good_median} bad={bad_median})')
        proc = subprocess.run(args=['git', 'bisect', bisect_verb], check=True, capture_output=True, cwd=root_dir)
        # look at output to decide what to do.  more ideally, we'd
        # call into some lower level of git machinery, or expose a
        # script that "git bisect run" can call
        still_bisecting = (b'Bisecting: ' in proc.stdout) and (b'left to test after this' in proc.stdout)
        print(proc.stdout.decode('utf-8'))
        if not still_bisecting:
            log(f"done bisecting")
            sys.exit(1)

if __name__ == '__main__':
    main()

import os
import argparse
import textwrap
import operator
import subprocess
from itertools import accumulate
from visualizer import *

# This script is designed to wrap and run membench multiple times for varying
# size/# of gpus, and chart the results using the visualizer.py script.

dir_perms = 0o640  # rw-r-----


def is_power_of_two(x):
    if (x == 0):
        return false
    return (x & (x - 1) == 0)


# Execute and store output to designated file
# Send stderr to dev/null since it contains gbench preamble
def run_membench(n, g, trials, executable, bench_filter, file):
    proc = subprocess.run(args=[
        executable, '-n',
        str(n), '-g',
        str(g), '-t',
        str(trials), '-r', bench_filter, '--benchmark_format=csv'
    ],
                          timeout=600,
                          stdout=file,
                          stderr=open(os.devnull, 'wb'))


# Run membench executable for each all combinations of lengths/ngpus
# Mode determines how the data will be organized and compared:
#   - Default -> N vs. bw
#   - Weak    -> ngpus (on scaling N) vs. bw -- this will expect lengths and ngpus to match
#   - Strong  -> ngpus (on constant N) vs. bw
def run(lengths, ngpus, trials, executable, log_path, bench_filter, mode):
    total_runs = len(ngpus) if mode == 'weak' else len(lengths) * len(ngpus)
    current_runs = 1

    def execute(n, g, path):
        nonlocal current_runs
        print(
            f'[{current_runs}/{total_runs}] Running {executable} on size {n} x {n}, across {g} GPUs'
        )
        with (open(localpath + f'/log{n}-{g}.csv', 'w')) as file:
            run_membench(n, g, trials, executable, bench_filter, file)
        current_runs += 1

    if mode == 'default':
        # Make a separate graph for each ngpu run
        for g in ngpus:
            localpath = log_path + f'/default/{g}'
            os.makedirs(localpath, dir_perms, exist_ok=True)
            for n in lengths:
                execute(n, g, localpath)

    elif mode == 'weak':
        # Single graph will be made using all run data
        assert len(lengths) == len(
            ngpus
        ), f'Expected same length from --length and --ngpus args for weak scaling'
        lengths.sort()
        ngpus.sort()
        localpath = log_path + f'/weak_scaling'
        os.makedirs(localpath, dir_perms, exist_ok=True)
        for i, g in enumerate(ngpus):
            n = lengths[i]
            execute(n, g, localpath)
    else:
        # Graph per each N in lengths
        for n in lengths:
            localpath = log_path + f'/strong_scaling/{n}'
            os.makedirs(localpath, dir_perms, exist_ok=True)
            for g in ngpus:
                execute(n, g, localpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-GPU Memory Transfer Benchmark',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-n',
        '--lengths',
        type=int,
        dest='lengths',
        nargs='+',
        default=[512, 1024, 2048],
        help=textwrap.dedent(
            'List of power of 2 lengths to run, for 2D matrices.\nEx. "--length 128 256" to run on 128^2 and 256^2 matrices'
        ))
    parser.add_argument(
        '-g',
        '--ngpus',
        type=int,
        dest='ngpus',
        nargs='+',
        default=[1, 2, 4, 8],
        help='List of power of 2 number of gpus to run on. Ex. "--ngpus 2 4 8"'
    )
    parser.add_argument('-t',
                        '--trials',
                        type=int,
                        dest='trials',
                        default=10,
                        help='Number of trials to run per benchmark type')
    parser.add_argument('-x',
                        '--exec',
                        dest='executable',
                        default="./build/membench",
                        help='Path to membench executable')
    parser.add_argument(
        '-b',
        '--build',
        dest='build',
        action='store_true',
        default=False,
        help='Flag to enable building membench from this script')
    parser.add_argument('-l',
                        '--log-path',
                        dest='log_path',
                        default=".",
                        help='Destination path for membench output logs')
    parser.add_argument('-o',
                        '--output-path',
                        dest='out_path',
                        default=".",
                        help='Destination path for graph output')
    parser.add_argument('-m',
                        '--mode',
                        dest='mode',
                        default='default',
                        choices=['default', 'weak', 'strong'],
                        help=textwrap.dedent('''\
            The mode of the benchmark files [default|weak|strong].
            - "default" graphs N vs bandwidth per GPU # from --ngpus
            - "weak" graphs ngpus vs bandwidth for different N per GPU #
                - Note: # of args in --lengths must correspond to N per --ngpus
            - "strong" graphs ngpus vs bandwidth per N
            '''))
    parser.add_argument('-f',
                        '--filter',
                        dest='filter',
                        default='all',
                        help=textwrap.dedent('''\
                                Filter for which benchmarks to run, taken as space-separated list of benchmark names.
                                Refer to output of `./membench -h` for up-to-date list of benchmarks.'''
                                             ))

    args = parser.parse_args()

    lengths = args.lengths
    ngpus = args.ngpus
    is_length_po2 = list(
        accumulate([is_power_of_two(x) for x in lengths], operator.and_))[0]
    is_ngpus_po2 = list(
        accumulate([is_power_of_two(x) for x in ngpus], operator.and_))[0]
    assert is_length_po2, f'Lengths: non power of two length specified in {lengths}'
    assert is_ngpus_po2, f'# of GPUs: non power of two GPU # specified in {ngpus}'
    assert os.path.exists(
        args.executable
    ), f'Membench exec path: {args.executable} is not a valid path'
    assert os.path.exists(
        args.log_path
    ), f'Membench log path: {args.log_path} is not a valid path'
    assert os.path.exists(
        args.out_path
    ), f'Membench output path: {args.out_path} is not a valid path'

    print(f'''Executing membench with the following args:
    Lengths\t= {lengths}
    GPUs\t= {ngpus}
    Executable\t= {args.executable}
    Filter\t= {args.filter}
    Mode\t= {args.mode}
    Log Path\t= {args.log_path}
    Output Path\t= {args.out_path}
          ''')

    print("Starting membench runs to collect data...")
    run(lengths, ngpus, args.trials, args.executable, args.log_path,
        args.filter, args.mode)
    print("Now parsing data and graphing...")
    if (args.mode == 'default'):
        for g in ngpus:
            full_log_path = args.log_path + f'/default/{g}/'
            parse(full_log_path, args.out_path, args.mode)
    elif (args.mode == 'weak'):
        full_log_path = args.log_path + '/weak_scaling/'
        parse(full_log_path, args.out_path, args.mode)
    else:
        for n in lengths:
            full_log_path = args.log_path + f'/strong_scaling/{n}/'
            parse(full_log_path, args.out_path, args.mode)
    print("Done!")

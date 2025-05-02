import os
import argparse
import textwrap
import operator
import subprocess
from itertools import accumulate

# This script is designed to wrap and run membench multiple times for varying
# size/# of gpus, and chart the results using the visualizer.py script.

dir_perms = 0o640  # rw-r-----


def is_power_of_two(x):
    if (x == 0):
        return false
    return (x & (x - 1) == 0)


def run_membench(n, g, trials, executable, bench_filter, file_path):
    proc = subprocess.run(args=[
        executable, '-n',
        str(n), '-g',
        str(g), '-t',
        str(trials), '-r', bench_filter, '--benchmark_out_format=csv',
        f'--benchmark_out={file_path}'
    ],
                          timeout=300,
                          stdout=open(os.devnull, 'wb'),
                          stderr=open(os.devnull, 'wb'))


# Run membench executable for each all combinations of lengths/ngpus
# Mode determines how the data will be organized and compared:
#   - Default -> N vs. bw
#   - Weak    -> ngpus (on scaling N) vs. bw -- this will expect lengths and ngpus to match
#   - Strong  -> ngpus (on constant N) vs. bw
def run(lengths, ngpus, trials, executable, log_path, bench_filter, mode):
    if mode == 'default':
        # Make a separate graph for each ngpu run
        for g in ngpus:
            localpath = log_path + f'/{g}'
            os.makedirs(localpath, dir_perms, exist_ok=True)
            for n in lengths:
                print(f'Running membench on size {n} x {n}, across {g} GPUs')
                run_membench(n, g, trials, executable, bench_filter,
                             localpath + f'/log{n}')

    elif mode == 'weak':
        assert len(lengths) == len(
            ngpus
        ), f'Expected same length from --length and --ngpus args for weak scaling'
        sort(lengths)
        sort(ngpus)
        for i in enumerate(ngpus):
            n = lengths[i]

    else:
        pass


def graph(lengths, ngpus, out_path, mode):
    pass


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
                        default="./",
                        help='Destination path for membench output logs')
    parser.add_argument('-o',
                        '--output-path',
                        dest='out_path',
                        default="./",
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

    run(lengths, ngpus, args.trials, args.executable, args.log_path,
        args.filter, args.mode)
    graph(lengths, ngpus, args.out_path, args.mode)

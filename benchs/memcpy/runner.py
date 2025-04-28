import os
import argparse
import operator
import subprocess
from itertools import accumulate

# This script is designed to run membench multiple times for varying size/# of gpus,
# and chart the results using the visualizer.py script.

dir_perms = 0o660  # rw-rw----


def is_power_of_two(x):
    if (x == 0):
        return false
    return (x & (x - 1) == 0)


def run(lengths, ngpus, executable, log_path, mode):
    # Default -> N vs. bw
    # Weak    -> ngpus (on scaling N) vs. bw
    # Strong  -> ngpus (on constant N) vs. bw
    if mode == default:
        # Make a separate graph for each ngpu
        for g in ngpus:
            localpath = log_path + f'/{g}'
            os.mkdir(localpath)


def graph(lengths, ngpus, out_path, mode):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--lengths',
        type=int,
        dest='lengths',
        nargs='+',
        default=[512, 1024, 2048],
        help=
        'List of power of 2 lengths to run, for 2D matrices. Ex. "--length 128 256" to run on 128^2 and 256^2 matrices'
    )
    parser.add_argument(
        '-g',
        '--ngpus',
        type=int,
        dest='ngpus',
        required=False,
        nargs='+',
        default=[1, 2, 4, 8],
        help='List of power of 2 number of gpus to run on. Ex. "--ngpus 2 4 8"'
    )
    parser.add_argument(
        '-x',
        '--exec',
        dest='executable',
        required=False,
        default="./build/membench",
        help="Path to membench executable, if not located in ./build/membench")
    parser.add_argument('-l',
                        '--log-path',
                        dest='log_path',
                        required=False,
                        default="./",
                        help='Destination path for membench output logs')
    parser.add_argument('-o',
                        '--output-path',
                        dest='out_path',
                        required=False,
                        default="./",
                        help='Destination path for graph output')
    parser.add_argument(
        '-m',
        '--mode',
        dest='mode',
        required=False,
        choices=['default', 'weak', 'strong'],
        help=
        'The mode of the benchmark files [default|weak|strong]. Default graphs N vs throughput for fixed # gpus'
    )

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

    run(lengths, ngpus, args.executable, args.log_path, args.mode)
    graph(lengths, ngpus, args.out_path, args.mode)

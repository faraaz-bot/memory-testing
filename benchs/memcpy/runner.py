import os
import argparse
from itertools import accumulate

# This script is designed to run membench multiple times for varying size/# of gpus,
# and chart the results using the visualizer.py script.


def is_power_of_two(x):
    if (n == 0):
        return false
    return (n & (n - 1) == 0)


def run(lengths, ngpus, executable, log_path):
    pass


def graph(lengths, ngpus, out_path):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--lengths',
        dest='lengths',
        required=True,
        help=
        'List of power of 2 lengths to run, for 2D matrices. Ex. "--length 128 256" to run on 128^2 and 256^2 matrices'
    )
    parser.add_argument(
        '-g',
        '--ngpus',
        dest='ngpus',
        required=False,
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

    args = parser.parse_args()

    lengths = args.lengths
    ngpus = args.ngpus
    is_length_po2 = accumulate([is_power_of_two(x) for x in lengths],
                               operator.and_)
    is_ngpus_po2 = accumulate([is_power_of_two(x) for x in ngpus],
                              operator.and_)
    assert is_length_po2, f'Lengths: non power of two length specified in {lengths}'
    assert is_ngpus_po2, f'# of GPUs: non power of two GPU # specified in {ngpus}'
    assert os.path.exists(
        args.executable,
        f'Membench exec path: {args.executable} is not a valid path')
    assert os.path.exists(
        args.log_path,
        f'Membench log path: {args.log_path} is not a valid path')
    assert os.path.exists(
        args.out_path,
        f'Membench output path: {args.out_path} is not a valid path')

    run(lengths, ngpus, executable, log_path)
    graph(lengths, ngpus, out_path)

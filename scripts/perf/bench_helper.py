#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import re
from io import StringIO


# Plots bench outputs by averaging time and gflops separately
# Currently plots multiple files against each other, expecting batch size as metric
def plot(results, output_path, use_log_x, use_log_y):
    x_axis = sorted(results.keys())  # Batch sizes, sorted

    timings = []
    gflops = []
    normalized_timings = []

    # Gather all timings/gflops for each file/batch run
    for batch, pair in dict(sorted(results.items())).items():
        timings.append(pair[0])
        gflops.append(pair[1])
        normalized_timings.append(pair[0] / batch)

    # Plot batch size vs timings/normalized timings:
    plt.clf()
    plt.grid()

    plt.xlabel('Batch Size (N)')
    plt.scatter(x_axis, timings, s=10, label=f'Raw Timings')
    plt.plot(x_axis, timings)
    plt.scatter(x_axis, normalized_timings, s=10, label=f'Normalized Timings')
    plt.plot(x_axis, normalized_timings)

    if use_log_x:
        plt.xscale('log', base=2)
    if use_log_y:
        plt.yscale('log', base=2)
    plt.ylabel("Time Taken (ms)")
    plt.title("Batch Size vs. Timings")
    plt.legend()
    plt.savefig(output_path + 'timings.png')

    # Plot gflops:
    plt.clf()
    plt.grid()

    plt.xlabel('Batch Size (N)')
    plt.scatter(x_axis, gflops, s=10)
    plt.plot(x_axis, gflops)

    if use_log_x:
        plt.xscale('log', base=2)
    if use_log_y:
        plt.yscale('log', base=2)
    plt.ylabel("Time Taken (ms)")
    plt.title("Batch Size vs. Gflops")
    plt.savefig(output_path + 'gflops.png')


def parse_output(filepath):
    with open(filepath, 'r') as f:
        # Grab gpu time/gflops lines from output, stripping away leading space and 'ms' text
        all_lines = f.readlines()
        if "SKIPPED" in all_lines[-1]:
            return []
        lines = all_lines[-2:]  # Grab last two lines of bench output
        timings_str = lines[0].split(':')[-1][1:-4]
        gflops_str = lines[1].split(':')[-1][3:-1]

    timings = pd.Series(timings_str.split(' '), dtype=float, copy=False)
    gflops = pd.Series(gflops_str.split(' '), dtype=float, copy=False)
    return [timings.mean(), gflops.mean()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='rocfft bench helper')
    parser.add_argument("bench_logs", nargs=argparse.REMAINDER)

    # Note: specify these options before files, since csv_files takes remainder of args
    parser.add_argument('-o',
                        '--output_path',
                        default="./",
                        help='Directory to store output graphs')

    parser.add_argument('-t',
                        '--text_output',
                        action="store_true",
                        help="Flag to enable result output via text")

    parser.add_argument('-g',
                        '--disable_graph_output',
                        default=False,
                        action="store_true",
                        help="Flag to disable output via graph")

    parser.add_argument(
        '--log_x',
        default=False,
        action="store_true",
        help=
        "Flag to enable log base 2 scaling on x-axis (if disabled, then linear)"
    )

    parser.add_argument(
        '--log_y',
        default=False,
        action="store_true",
        help=
        "Flag to enable log base 2 scaling on y-axis (if disabled, then linear)"
    )

    args = parser.parse_args()
    results = {}  # Map batch size to metrics dict
    for file in args.bench_logs:
        # Extract first number found in filename as batch, if present
        nums_in_file = [int(num) for num in re.findall(r'\d+', file)]
        batch = nums_in_file[0]

        # Check for skipped test (ex. due to large memory requirement)
        res = parse_output(file)
        if not res:
            continue
        results[batch] = res

        if args.text_output:
            print(f'Batch Size = {batch}')
            print(
                f'Average time = {results[batch][0]}\nAverage gflops = {results[batch][1]}\n'
            )

    if not args.disable_graph_output:
        plot(results, args.output_path, args.log_x, args.log_y)

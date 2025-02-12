#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import re
from io import StringIO

"""
Tested by running MultEvent profiler with mi300x all .ini file config, with only MALL metrics enabled.
Extract blocks of following format from within output csv (note: script looks for int in filename for x-axis):
'
Data Fabric - MALL0

<metrics for MALL0...>

Data Fabric - MALL1

<metrics for MALL1...>

Data Fabric - MALL31
...
'
"""

# Store key metrics of interest, as well as filename for metric
# Ignore % time metrics, full vs partial hit rate
bw_fields = ["Total Read bandwidth", "Total Write bandwidth"]

hit_rate_fields = [
        "Total Hit Rate",
        "Total Write Hit Rate",
        "Total Read Hit Rate"
        ]

def plot(results, output_path, use_linear_scale_x, use_log_scale_y):
    x_axis = sorted(results.keys()) # Batch sizes

    plt.clf()

    # For each metric, get y-axis data per file given
    for field in bw_fields:
        y_axis = [results[batch][field] for batch in x_axis]
        plt.scatter(x_axis, y_axis, s=10, label=f'{field}')
        plt.plot(x_axis, y_axis)

    plt.grid()
    plt.xlabel('Batch Size (N)')
    if not use_linear_scale_x:
        plt.xscale('log', base=2)
    if use_log_scale_y:
        plt.yscale('log', base=2)
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("MALL Bandwidth Metrics")
    plt.legend()
    plt.savefig(output_path + 'bandwidth.png')

    plt.clf()

    for field in hit_rate_fields:
        y_axis = [results[batch][field] for batch in x_axis]
        plt.scatter(x_axis, y_axis, s=10, label=f'{field}')
        plt.plot(x_axis, y_axis)

    plt.grid()
    plt.xlabel('Batch Size (N)')
    if not use_linear_scale_x:
        plt.xscale('log', base=2)
    if use_log_scale_y:
        plt.yscale('log', base=2)
    plt.ylabel("Hit Rate (%)")
    plt.title("MALL Hit Rate Metrics")
    plt.legend()
    plt.savefig(output_path + 'hit_rate.png')



def parse_csv(filepath, num_columns):
    # Options for debugging
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)


    with open(filepath, 'r') as f:
        lines = [l.split('\n')[0] for l in f] 

    csvs = []
    
    # Extract each of 32 "MALL sections" into their own csv string and dataframe
    start = False
    curr_csv = ''
    num = -1
    for l in lines:
        if 'Data Fabric - MALL' in l:
            if start:
                csvs.append(curr_csv)
                curr_csv = ''
            start = True
            num = int(l.split('L')[-1]) # Track which MALL # we are on
            # Add in column headers, accounting for current MALL section # for easy merging later
            header_strings = [f'col_{num * num_columns + n}' for n in range(32)]
            curr_csv += f'metrics, {", ".join(header_strings)}\n'
            continue
    
        if start:
            if f'MALL{num}' in l:
                # Strip away MALL#
                stripped_str = l.replace(f'MALL{num}','MALL')
                curr_csv += f'{stripped_str}\n'
    csvs.append(curr_csv) # Catch last CSV

    # Extract csvs into their own dataframes, including only N columns (most columns are unused)
    dataframes = []

    for i, csv in enumerate(csvs):
        if i == 0:
            tmp_df = pd.read_csv(StringIO(csv), dtype=str).iloc[:,0:num_columns]
        else:
            tmp_df = pd.read_csv(StringIO(csv), dtype=str).iloc[:,1:num_columns]
        dataframes.append(tmp_df)

    # Merge together dfs row-wise
    # Note: column headers get duplicated, not an actual issue though?
    df = pd.concat(dataframes, axis=1)
    df = df.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace(' ', '0')

    field_data = {}
    # Extract desired metrics
    for field in bw_fields + hit_rate_fields:
        field_df = df[df['metrics'].str.contains(field)]
        field_df.reset_index(drop=True, inplace=True)
        del field_df['metrics']
        field_df = field_df.astype(float)
        field_data[field] = field_df.mean().mean()

    return field_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='metric_parser')
    parser.add_argument("csv_files", nargs=argparse.REMAINDER)
    
    # Note: specify these options before files, since csv_files takes remainder of args
    parser.add_argument('-n', '--num_columns', 
                        default=5, 
                        type=int,
                        help='Number of columns to take out of 32 (default 5 for MI300X)')

    parser.add_argument('-o', '--output_path',
                        default="./",
                        help='Directory to store output graphs')

    parser.add_argument('-t', '--text_output',
                        action="store_true",
                        help="Flag to enable result output via text")

    parser.add_argument('-g', '--disable_graph_output',
                        default=False,
                        action="store_true",
                        help="Flag to disable output via graph")

    parser.add_argument('--linear_scale_x',
                        default=False,
                        action="store_true",
                        help="Flag to enable linear scaling on x-axis (if disabled, then log base 2)")

    parser.add_argument('--log_scale_y',
                        default=False,
                        action="store_true",
                        help="Flag to enable log base 2 scaling on y-axis (if disabled, then linear)")

    args = parser.parse_args()
    results = {} # Map batch size to metrics dict
    for file in args.csv_files:
        # Extract number found in filename as batch, if present
        nums_in_file = [int(num) for num in re.findall(r'\d+', file)]
        batch = nums_in_file[0]

        results[batch] = parse_csv(file, args.num_columns)

        if args.text_output:
            print(f'Batch Size = {batch}')
            for k, v in results[batch].items():
                print(f'  {k}\t{v}')
            print()
    
    if not args.disable_graph_output:
        plot(results, args.output_path, args.linear_scale_x, args.log_scale_y)

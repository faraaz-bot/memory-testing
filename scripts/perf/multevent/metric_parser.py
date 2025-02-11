#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import re
from io import StringIO

"""
Tested by running MultEvent profiler with mi300x all .ini file config, with only MALL metrics enabled.
Extract blocks of following format from within output csv:
'
Data Fabric - MALL0

<metrics for MALL0...>

Data Fabric - MALL1

<metrics for MALL1...>

Data Fabric - MALL31
...
'
"""

# Metrics of interest
# fields = {
#     # Ignore % time metrics, full vs partial hit rate
#     "Total Read bandwidth": {"unit": "GB/s", "label": "Read"},
#     "Total Write bandwidth": {"unit": "GB/s", "label": "Write"},
#     "Total Hit Rate": {"unit": "%", "label": "Total"},
#     "Total Write Hit Rate": {"unit": "%", "label": "Write"},
#     "Total Read Hit Rate": {"unit": "%", "label": "Read"},
# }

# Store key metrics of interest, as well as filename for metric
fields = {
    # Ignore % time metrics, full vs partial hit rate
    "Total Read bandwidth": "read_bandwidth",
    "Total Write bandwidth": "write_bandwidth",
    "Total Hit Rate": "total_hit_rate",
    "Total Write Hit Rate": "total_write_hit_rate",
    "Total Read Hit Rate": "total_read_hit_rate",
}

def plot(results, output_path):
    x_axis = sorted(results.keys()) # Batch sizes

    # For each metric, get y-axis data per file given
    for field in fields.keys():
        y_axis = [results[batch][field] for batch in x_axis]
         
        plt.clf()

        plt.scatter(x_axis, y_axis, color='blue', s=10, zorder=2, label='')
        plt.plot(x_axis, y_axis, color='blue')

        # plt.scatter(x_axis, y_axis, color='red', s=10, zorder=3, label='')
        # plt.plot(x_axis, y_axis, color='red')

        # plt.xscale('log', base=2)
        # plt.yscale('log', base=2)

        plt.xlabel('Batch Size')
        y_label = "GB/s" if "bandwidth" in field else "%"
        plt.ylabel(y_label)
        plt.title(field)
        plt.grid()
        plt.legend()
        plt.savefig(output_path + f'{fields[field]}.png')  # save the figure


def parse_csv(filepath, num_columns):
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
    for k in fields.keys():
        field_df = df[df['metrics'].str.contains(k)]
        field_df.reset_index(drop=True, inplace=True)
        del field_df['metrics']
        field_df = field_df.astype(float)
        field_data[k] = field_df.mean().mean()

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

    parser.add_argument('-g', '--graph_output',
                        default=True,
                        action="store_true",
                        help="Flag to enable disable output via graph")

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
    
    if args.graph_output:
        plot(results, args.output_path)

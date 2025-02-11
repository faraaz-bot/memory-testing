#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from io import StringIO

"""
Intended specifically when run with mi300x all .ini file config, with only MALL metrics enabled.
Extract blocks of format: 
'
Data Fabric - MALL0

<metrics for MALL0...>

Data Fabric - MALL1

<metrics for MALL1...>

Data Fabric - MALL31
...
'
"""

# The interested field and its attributes
fields = {
    # Ignore % time metrics, full vs partial hit rate
    "Total Read bandwidth": {"unit": "GB/s", "label": "Read"},
    "Total Write bandwidth": {"unit": "GB/s", "label": "Write"},
    "Total Hit Rate": {"unit": "%", "label": "Total"},
    "Total Write Hit Rate": {"unit": "%", "label": "Write"},
    "Total Read Hit Rate": {"unit": "%", "label": "Read"},
}

# Group the fields to show details per bank.
# NB: dataclasses may be better.
groups = [
    {
        "title": "Hit Rate Details (%)",
        "items": ["Total Hit Rate", "Total Read Hit Rate", "Total Write Hit Rate"],
        "height": 16,
        "y_lim": 100,
    },
    {
        "title": "Bandwidth Details (GB/s)",
        "items": ["Total Read bandwidth", "Total Write bandwidth"],
        "height": 16,
        "y_lim": None,
    },
]

def plot(dataframes):
    pass

def parse_csv(filepath, num_columns):
    with open(filepath, 'r') as f:
        lines = [l.split('\n')[0] for l in f] 

    csvs = []
    
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
            # Add in column headers
            curr_csv += l + f', GPUDF_0, GPUDF_1, GPUDF_2, GPUDF_3, GPUDF_4, GPUDF_5, GPUDF_6, GPUDF_7, GPUDF_8, GPUDF_9, GPUDF_10, GPUDF_11, GPUDF_12, GPUDF_13, GPUDF_14, GPUDF_15, GPUDF_16, GPUDF_17, GPUDF_18, GPUDF_19, GPUDF_20, GPUDF_21, GPUDF_22, GPUDF_23, GPUDF_24, GPUDF_25, GPUDF_26, GPUDF_27, GPUDF_28, GPUDF_29, GPUDF_30, GPUDF_31\n'
            continue
    
        if start:
            if f'MALL{num}' in l:
                curr_csv += f'{l}\n'
    csvs.append(curr_csv) # Catch last CSV

    # Extract csvs into their own dataframes, including only 5 columns (for MI300X)
    dataframes = []
    print(num_columns)
    for csv in csvs:
        df = pd.read_csv(StringIO(csv)).iloc[:,:num_columns]
        dataframes.append(df)
        # print(df,'\n')
    
    print(dataframes[0], '\n')
    # Adjust df keys to match each other, for merging dfs later


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='metric_parser')
    parser.add_argument("df_file")
    # parser.add_argument('-s',
    #                     '--start_line',
    #                     default=18,
    #                     type=int,
    #                     help='Specify the line start to read.')

    # parser.add_argument('-f, --file',
    #                     type=str,
    #                     help='Path to csv file to read from.',
    #                     default="GPUDF.csv")
    
    parser.add_argument('-n', '--num_columns', 
                        default=5, 
                        type=int, 
                        help='Number of columns to take out of 32 (default 5 for MI300X)')

    args = parser.parse_args()
    parse_csv(args.df_file, args.num_columns)

    """
    TODO:
    - Print average metrics for key values
    - Allow comparison against multiple files (either multiple args, or a filepath to glob)
    """


    # if args.command == 'list':
    #     scprint(set(['function_pool.cpp'] + list_generated_kernels(kernels)))
#
    # if args.command == 'generate':
    #     cpu_functions = generate_kernels(kernels, precisions,
    #                                      args.stockham_gen)
    #     func_files = generate_cpu_function_pool_pieces(cpu_functions,
    #                                                    args.num_files)
    #     for i in range(args.num_files):
    #         write(f'function_pool_init_{i}.cpp', func_files[i], format=False)
    #     write('function_pool.cpp',
    #           generate_cpu_function_pool_main(args.num_files),
    #           format=False)

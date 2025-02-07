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

def parse(filepath):
    starting_lines = []
    dataframes = []
    # Get starting lines to parse MALL metrics from
    with open(filepath, 'r', newline='') as f:
        for line_no, line in enumerate(f):
            if "Data Fabric - MALL" in line:
                starting_lines.append(line_no)
                
    offset = starting_lines[1] - starting_lines[0] # Both *should* exist now 
    
    f = open(filepath, 'r')
    lines = f.readlines() # Exclude '\n' chars

    # Extract each "MALL section", with number of columns and ignoring empty data
    for start in [line+2 for line in starting_lines]: # Exclude section title and blank line
        section_str = ''.join(lines[start:start+offset])
        # print(StringIO(section_str).read())
        df = pd.read_csv(
                StringIO(section_str),
                ).iloc[:,:5].dropna()
        dataframes.append(df)
        print(df, '\n')

    # Merge together dataframes, after adjusting df keys to match each other
    
    # Parse as one large df or as 32 dfs, one per MALL section? Seems easier to take each as a separate df
    # , and then 
    # Read in csv starting from MALL metrics, only keep 5 columns for MI300X, and exclude missing vals
    # df = pd.read_csv(
    #         filepath,
    #         header=starting_line
    #         ).iloc[:,:5].dropna()
    # print(df)


if __name__ == '__main__':
    """Command line interface..."""
    parser = argparse.ArgumentParser(prog='metric_parser')
    parser.add_argument("df_file", nargs=argparse.REMAINDER)
    # parser.add_argument('-s',
    #                     '--start_line',
    #                     default=18,
    #                     type=int,
    #                     help='Specify the line start to read.')

    # parser.add_argument('-f, --file',
    #                     type=str,
    #                     help='Path to csv file to read from.',
    #                     default="GPUDF.csv")
    
    args = parser.parse_args()
    parse(args.df_file[0])

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

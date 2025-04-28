from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import os
import argparse


def graph_default(storage, out_dir):

    num_tests = len(storage[0]['name'])
    ngpus = storage[0]['ngpus']

    x_axis = []
    for i in range(num_tests):
        y_axis = []
        name = ''

        for s in storage:
            if name == '':
                name = s['name'][i]
            if i == 0:
                x_axis.append(s['size'])
            y_axis.append(s['gbps'][i])
        plt.plot(x_axis, y_axis, label=name)

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Dimension Size (N x N)')
    plt.ylabel('Throughput (GB/s)')
    plt.title(f'Throughput for copying data between {ngpus} GPUs')
    plt.legend()
    plt.savefig(output_dir + '/visual.png')


def parse(input_dir, output_dir, mode):
    storage = []
    files = glob(f'{input_dir}/*.csv')

    for f in files:
        temp = {}
        dict = pd.read_csv(os.path.abspath(f),
                           on_bad_lines='skip').to_dict(orient='list')

        temp['size'] = int(dict['Dimension (N x N)'][0])
        temp['ngpus'] = int(dict['Device Count'][0])
        temp['name'] = [name.split('/')[0] for name in dict['name']]
        temp['gbps'] = [float(gbps) for gbps in dict['Throughput (GB/s)']]

        storage.append(temp)

    if mode == 'default':
        storage.sort(key=lambda x: x['size'])
        visualize(storage, output_dir)
    elif mode == 'weak':
        storage.sort(key=lambda x: x['ngpus'])
    else:
        storage.sort(key=lambda x: x['ngpus'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        dest='input_dir',
                        required=True,
                        help='Path to benchmark files')
    parser.add_argument(
        '-o',
        '--output',
        dest='output_dir',
        required=False,
        help=
        'Path to dump png output, defaulting to path specified by -i/--input if not provided'
    )
    parser.add_argument(
        '-m',
        '--mode',
        dest='mode',
        required=False,
        default='default',
        help=
        'The mode of the benchmark files [default|weak|strong]. Default graphs N vs throughput for fixed # gpus'
    )

    args = parser.parse_args()
    valid_modes = ['default', 'weak', 'strong']

    mode = args.mode.lower()
    assert mode in valid_modes, f'Mode: {mode} is not a valid mode from [default|weak|strong]'
    assert os.path.exists(
        args.input_dir), f'Input dir: {args.input_dir} is not a valid path'

    input_dir = args.input_dir
    if (args.output_dir == None):
        output_dir = input_dir
    else:
        output_dir = args.output_dir
    assert os.path.exists(
        output_dir), f'Output dir: {args.output_dir} is not a valid path'

    parse(input_dir, output_dir, mode)

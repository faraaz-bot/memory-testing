from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import os
import argparse


# Graph N vs. bandwidth
def graph_default(storage, out_dir):
    num_tests = len(storage[0]['name'])
    ngpus = storage[0]['ngpus']
    length = storage[0]['size']

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
        print(x_axis, y_axis, name)

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Dimension Size (N x N)')
    plt.ylabel('Throughput (GB/s)')
    plt.title(f'Throughput for copying data between {ngpus} GPUs')
    plt.legend(fontsize=10)
    plt.savefig(out_dir + f'/default{length}-{ngpus}.png')


# Graph for weak or strong scaling (# of GPUs vs bw)
def graph_scaling(storage, out_dir, mode):
    num_tests = len(storage[0]['name'])
    ngpus = storage[0]['ngpus']
    length = storage[0]['size']

    fig, ax = plt.subplots()

    x_axis = []
    for i in range(num_tests):
        y_axis = []
        name = ''

        for s in storage:
            if name == '':
                name = s['name'][i]
            if i == 0:
                x_axis.append(s['ngpus'])
            y_axis.append(s['gbps'][i])
        plt.plot(x_axis, y_axis, label=name, marker='o')

    plt.legend(fontsize=10)
    if mode == 'strong':
        plt.savefig(out_dir + f'/{mode}-{length}.png')
    else:
        plt.savefig(out_dir + f'/{mode}.png')


def parse(input_dir, output_dir, mode):
    storage = []
    files = glob(f'{input_dir}/*.csv', recursive=True)

    for f in files:
        temp = {}
        dict = pd.read_csv(os.path.abspath(f),
                           on_bad_lines='skip').to_dict(orient='list')

        temp['size'] = int(dict['Dimension (N x N)'][0])
        temp['ngpus'] = int(dict['Device Count'][0])
        temp['name'] = [name.split('/')[0] for name in dict['name']]
        temp['gbps'] = [float(gbps) for gbps in dict['Throughput (GB/s)']]

        storage.append(temp)

    print(storage)
    if mode == 'default':
        storage.sort(key=lambda x: x['size'])
        graph_default(storage, output_dir)
    else:
        storage.sort(key=lambda x: x['ngpus'])
        graph_scaling(storage, output_dir, mode)


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

    args = parser.parse_args()

    assert os.path.exists(
        args.input_dir), f'Input dir: {args.input_dir} is not a valid path'

    input_dir = args.input_dir
    if (args.output_dir == None):
        output_dir = input_dir
    else:
        output_dir = args.output_dir
    assert os.path.exists(
        output_dir), f'Output dir: {args.output_dir} is not a valid path'

    parse(input_dir, output_dir, args.mode)

from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import os
import argparse

def visualize(storage):

    num_tests = len(storage[0]['name'])

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
    plt.title('Throughput for copying data between 8 GPUs')
    plt.legend()
    plt.savefig('visual.png')
    


def visualize_csv(input_dir):
    storage = []
    files = glob(f'{input_dir}/*.csv')

    for f in files:
        temp = {}
        dict = pd.read_csv(os.path.abspath(f), on_bad_lines='skip').to_dict(orient='list')

        temp['size'] = int(dict['Dimension (N x N)'][0])
        temp['name'] = [name.split('/')[0] for name in dict['name']]
        temp['gbps'] = [ float(gbps) for gbps in dict['Throughput (GB/s)']]

        storage.append(temp)

    storage.sort(key=lambda x : x['size'])

    visualize(storage)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_dir', required=True,
                        help='Path to benchmark files')
    parser.add_argument('-m', '--mode', dest='mode', required=False, default='csv', 
                        help='The mode of the benchmark files [console|csv|json]. Default: csv')

    args = parser.parse_args()

    assert os.path.exists(args.input_dir), f'{args.input_dir} is not a valid path'

    input_dir = args.input_dir
    mode = args.mode.lower()

    if mode == 'console':
        ...
        #TODO make console visualizer
    elif mode == 'csv':
        visualize_csv(input_dir)
    elif mode == 'json':
        ...
        #TODO make json visualizer
    else:
        raise TypeError(f'{mode} is not a valid mode! Choose one of [console|csv|json]')

    

     

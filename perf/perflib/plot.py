'''Plotting...'''

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import types

from pathlib import Path as path
from plotly.subplots import make_subplots

NS = types.SimpleNamespace

def drop_first(df):
    def _drop_first(grp):
        return grp.iloc[1:]

    return df.groupby(['method', 'n', 'size'], as_index=False).apply(_drop_first).reset_index(drop=True)


def time_vs_size(df):

    df = drop_first(df).groupby(['method', 'n', 'size']).agg([np.mean, np.std])
    df.columns = ["_".join(x) for x in df.columns.ravel()]
    df = df.reset_index()

    times = px.scatter(df, x="n", y="time_mean", color="method", error_y="time_std", log_x=True,
                       labels={
                           'n': '1D length',
                           'time': 'Time (ms)'
                       })
    times.update_layout(yaxis=dict(tickformat=".2e"))
    return times


def throughput_vs_size(df):
    df['throughput'] = df.size / df.time
    times = px.scatter(df, x="n", y="throughput", color="method",
                       labels={
                           'n': '1D length',
                           'throughput': 'Throughput (one-way, bytes/s)'
                       })

    times.update_layout(yaxis=dict(tickformat=".2f"))
    return times


def plot_from_csv_pair(plot_function, fnames):
    roc, cu = path(fnames[0]), path(fnames[1])

    rocdf = pd.read_csv(roc)
    idx = rocdf['method'] == 'hipfft'
    rocdf['method'][idx] = 'rocfft'

    cudf = pd.read_csv(cu)
    idx = cudf['method'] == 'hipfft'
    cudf['method'][idx] = 'cufft'

    df = pd.concat([rocdf, cudf])

    return NS(title=roc.stem, plot=plot_function(df))


def render_plots(plots):
    html = '<html><head><title>Performance explorer</title></head><body>'
    include_js = True
    for p in plots:
        html += '<h2>' + p.title + '</h2>'
        html += p.plot.to_html(full_html=False, include_plotlyjs=include_js)
        include_js = False
    html += '</body></html>'
    return html


def plot(plot_function, rocfft='rocfft', cufft='cufft', htmlfile='timings-plots.html'):
    roc_files = sorted(path('rocfft').glob('*.csv'))
    cu_files   = [ path('cufft') / x.name for x in roc_files ]
    plots = [ plot_from_csv_pair(plot_function, pair) for pair in zip(roc_files, cu_files) ]
    html = render_plots(plots)
    path(htmlfile).write_text(html)

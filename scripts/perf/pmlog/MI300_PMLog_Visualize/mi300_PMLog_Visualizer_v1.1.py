#!/usr/bin/env python3

###############################################################################
#
#  Description:
#    Categorize and visualize PMlog to html report for MI300
#
#  Usage:
#    - Comparable mode
#        python3 mi300_PMLog_Visualizer.py -i pmlog1 pmlog2 ...
#    - Batch mode
#        python3 mi300_PMLog_Visualizer.py -b pmlog_dir
#
#    See more options with -h
#
#  Inputs:
#    PM log files
#
#  Default Signal Config File:
#    mi300a_pmlog_visualize_config.xlsx
#      - sheet "signals": defines keywords of signals to visualize in category
#        of Freq/Power/Temp/BW/Misc
#
#  Outputs:
#    pmlog_analysis_report.html
#
#  Background:
#    MI300A: 4AID+6XCD+3CCD
#
#
#  Author:  William Li (william.li@amd.com)
#  Date:
#  History:
#    02/06/2023: publish version 1.0
#    01/27/2023: initiate this script based on mi300_PMLog_visualizer.ipynb
#
###############################################################################

import os
import re
import sys
import time
import math
import argparse
from datetime import datetime
from pathlib import Path as path

__authors__ = "William Li"
__copyright__ = "Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved."
__version__ = "1.0-Version"
__email__ = "william.li@amd.com"
__language__ = "Python 3.10.7"
__status__ = "Release"

#------------------------------------------------------------------------------
# global parameters

##Bokeh plot
PLOT_WIDTH = 1200
PLOT_HEIGHT = 640
MARKER = [
    'asterisk', 'circle', 'circle_cross', 'circle_dot', 'circle_x', 'circle_y',
    'cross', 'dash', 'diamond', 'diamond_cross', 'diamond_dot', 'dot', 'hex',
    'hex_dot', 'inverted_triangle', 'plus', 'square', 'square_cross',
    'square_dot', 'square_pin', 'square_x', 'star', 'star_dot', 'triangle',
    'triangle_dot', 'triangle_pin'
]
COLOR = [
    'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
    'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue',
    'darkcyan', 'darkgoldenrod', 'darkgreen', 'darkkhaki', 'darkmagenta',
    'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray',
    'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia',
    'gainsboro', 'gold', 'limegreen', 'linen', 'magenta'
]

list_category = ['Freq', 'Power', 'Temp', 'BW', 'Misc']

#------------------------------------------------------------------------------
# helper functions


def genDictFromPMLog(cur_list, cur_csv, cur_header):
    listData = []
    if not cur_list: return [listData, cur_header, 0]

    for cur_cluster in cur_list:
        keyWord = cur_cluster[0]
        dictData = {}
        max_value = 0
        for cur_element in cur_header:
            if re.search(re.compile(keyWord, re.IGNORECASE), cur_element):
                ##print(cur_element)
                dictData[cur_element] = cur_csv[cur_element].to_list()
                max_value = max(max_value, max(dictData[cur_element]))

        # print("cur_cluster[1]", cur_cluster[1], max_value)
        listData.append([dictData, cur_cluster[1], max_value])
        ##remove found items from header list
        for cur_element in dictData.keys():
            cur_header.remove(cur_element)

    return [listData, cur_header]


def drawOnePMLogFigure(cur_cluster,
                       duration,
                       width,
                       height,
                       span=None,
                       y_range_end=None):
    dictData = cur_cluster[0]
    cur_YLabel = cur_cluster[1]
    #print("~~~~~~~~~~~~~~~~~~~~~ ", cur_YLabel, "max", cur_cluster[2])

    sigName = list(cur_cluster[0].keys())
    #print(sigName)
    cur_Data = dictData[sigName[0]]
    #print(max(cur_Data), duration)
    dictData['xData'] = list(
        map(lambda x: x * duration, list(range(0, len(cur_Data)))))
    mdata = ColumnDataSource(data=dictData)

    # print(sigName[0], "\t\t", max(cur_Data), max(dictData['xData']),
    #       min(dictData['xData']))

    TOOLS = "pan,wheel_zoom,box_zoom,poly_select,lasso_select,tap,reset,hover,save"
    p = figure(tools=TOOLS,
               title=cur_YLabel + ' vs Time',
               width=width,
               height=height)

    # left-y-axis only
    legendItem = []
    for cur_Idx, cur_Key in enumerate(sigName):
        cur_Line = p.line(x='xData',
                          y=cur_Key,
                          line_width=0.9,
                          color=COLOR[cur_Idx],
                          source=mdata)
        cur_Scat = p.scatter(x='xData',
                             y=cur_Key,
                             size=1,
                             color=COLOR[cur_Idx],
                             source=mdata,
                             marker=MARKER[cur_Idx])
        legendItem.append(
            LegendItem(label=cur_Key,
                       renderers=[cur_Line, cur_Scat],
                       index=cur_Idx))
    cur_Legend = Legend(items=legendItem,
                        location='top_center',
                        orientation='vertical',
                        label_text_font_size='10px')
    p.add_layout(cur_Legend, 'right')
    p.legend.click_policy = "hide"
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.axis_label = 'time (ms) at ' + str(duration) + ' ms/sample'
    p.yaxis.axis_label = cur_YLabel

    if y_range_end:
        p.y_range.end = y_range_end
    if span:
        p.add_tools(CrosshairTool(overlay=span))

    return p


def up_bound(n):
    '''
    Return a slight bigger number than n. n might be a float or int.
    For better visualization.
    '''
    if n == 0:
        return 0
    else:
        highest_digit = math.floor(math.log10(abs(n)))
        return n + pow(10, highest_digit - 1)


#------------------------------------------------------------------------------
# main entrance

if __name__ == "__main__":

    ### parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--inputs',
        nargs="+",
        help='Name of the input PMlog files to analyze. Appendable.')
    parser.add_argument('-o',
                        '--out_dir',
                        default=path.cwd(),
                        help='Specify the output directory.')
    parser.add_argument(
        '-b',
        '--batch',
        default=None,
        help='The directory of batched PMlog files to handle, no comparsion.')
    parser.add_argument('-s',
                        '--samples',
                        nargs="+",
                        default=[50],
                        help='The PM log sample duration in ms.')
    parser.add_argument(
        '-p',
        '--prefix',
        default="pmlog_analysis_report",
        help='Give a prefix name for the html files instead of the default one.'
    )
    parser.add_argument(
        '-c',
        '--config',
        default="mi300a_PMlog_visualize_config.xlsx",
        help=
        'The signal configuration file, defaut is ./mi300a_PMlog_visualize_config.xlsx.'
    )

    parser.add_argument("--dependency",
                        action="store_true",
                        help="\t\tList the installation dependency.")

    args = parser.parse_args()

    if args.dependency:
        print("pip3 install pandas bokeh openpyxl")
        sys.exit(0)

    start_time = time.time()

    ### Initialization

    import pandas as pd

    #Bokeh libraries and modules
    from bokeh.io import show, output_notebook, output_file, save
    from bokeh.plotting import figure
    from bokeh.models import Range1d, FactorRange, LinearAxis, Legend, Label, ColumnDataSource, Span, LegendItem, CrosshairTool, Tabs, TabPanel
    from bokeh.layouts import gridplot, row, column
    from bokeh.models.widgets import TextInput
    from bokeh.transform import factor_cmap
    from bokeh.palettes import Spectral5, Viridis256, Colorblind, Magma256, Turbo256
    from bokeh.transform import dodge

    # TODO: remove the hack code
    # dfCfg.dropna(inplace=True)
    ticket_name = "ticket_xxx"  # dfCfg['Ticket Name'][2]  ##Title of JIRA ticket
    scorecard_filename = "scorecard_xxx"  # dfCfg['ScoreCard'][2]  ##ScoreCard xlsx

    # read signal keywords
    if not path(args.config).exists:
        sys.exit("The signal configuration file doesn't exist!")

    df = pd.read_excel(args.config, sheet_name='KeyWordDef')
    df.pop('Notes')
    df.dropna(inplace=True)
    dfFigureList = df.sort_values("Figure Index", axis=0, ascending=True)
    ### Initialization done

    category_spans = {}
    for c in list_category:
        category_spans[c] = [
            Span(dimension="width", line_dash="dashed", line_width=2),
            Span(dimension="height", line_dash="dotted", line_width=2)
        ]

    pmlog_files = []
    comparable = True

    if args.batch != None:
        batch_dir = path(args.batch)
        if not batch_dir.exists:
            sys.exit("Batch directory doesn't exist!")
        pmlog_files = list(batch_dir.glob("*.csv"))
        comparable = False
        args.out_dir = batch_dir
    else:
        for f in args.inputs:
            pmlog_files.append(path(f))

    # ignore unvisualized csv file for batch mode
    pmlog_files = [
        x for x in pmlog_files if x.name != "PMLog_Signals_not_visualized.csv"
    ]

    if len(args.samples) > 1 and len(args.samples) != len(pmlog_files):
        print("Sample numbers don't match PMlog numbers!")
        sys.exit(0)
    elif len(args.samples) == 1 and len(pmlog_files) != 1:
        for i in range(1, len(pmlog_files)):
            args.samples.append(args.samples[0])

    gridplot_figs = {}  # only use for comparable plotting

    for cur_var in list_category:
        gridplot_figs[cur_var] = []

    # Generate data dict for each case
    for f in pmlog_files:
        case_name = f.name
        cur_pmlog = pd.read_csv(f)
        pmlog_header = cur_pmlog.columns.tolist()

        # generate dict for plots
        for cur_var in list_category:
            globals()[case_name + 'list' + cur_var] = []

        for test_idx in range(len(dfFigureList)):
            cur_figure = dfFigureList.iloc[test_idx]
            cur_category = cur_figure['Category']
            cur_keyword = cur_figure['KeyWord']
            cur_label = cur_figure['Y-Axis Label']
            for cur_var in list_category:
                if re.search(cur_var, cur_category):
                    globals()[case_name + 'list' + cur_var].append(
                        [cur_keyword, cur_label])
                    break
        ##fetch PMLog data into list
        new_header = pmlog_header
        for cur_var in list_category:
            globals()[case_name + 'list' + cur_var +
                      'Data'], new_header = genDictFromPMLog(
                          globals()[case_name + 'list' + cur_var], cur_pmlog,
                          new_header)

        # save unvisualized signals for reference
        outCSV = pd.DataFrame(new_header)
        fileName = args.out_dir.joinpath("PMLog_Signals_not_visualized.csv")
        outCSV.to_csv(fileName, index=False, header=False)
        print("PMLog signals not visualized are stored in", fileName,
              "; total unvisualized signals=" + str(len(new_header)))

    # For comparable plotting, need to find the max value along y axis
    # for each group of subplots to make better alignment.
    if comparable and len(pmlog_files) > 1:
        for cur_var in list_category:
            base_list_data = globals()[pmlog_files[0].name + 'list' + cur_var +
                                       'Data']
            for f in pmlog_files[1:]:
                case_name = f.name
                cur_list_data = globals()[case_name + 'list' + cur_var +
                                          'Data']
                if cur_list_data:
                    for i in range(0, len(cur_list_data)):
                        # print("~~~", case_name, cur_group[1], "max",
                        #       cur_group[2])
                        base_list_data[i][2] = max(base_list_data[i][2],
                                                   cur_list_data[i][2])

    # Generate plots for Freq, Power, Temp, BW and MISCs in the list_category
    for f, s in zip(pmlog_files, args.samples):
        case_name = f.name
        print("Generating figure for case " + case_name)
        for cur_var in list_category:
            allPlot = []
            cur_list_data = globals()[case_name + 'list' + cur_var + 'Data']
            if not cur_list_data:
                print(case_name + ": No Signals to plot for " + cur_var)
            else:

                if comparable:
                    base_list_data = globals()[pmlog_files[0].name + 'list' +
                                               cur_var + 'Data']

                    for i in range(0, len(cur_list_data)):
                        allPlot.append(
                            drawOnePMLogFigure(cur_list_data[i],
                                               int(s),
                                               width=PLOT_WIDTH,
                                               height=PLOT_HEIGHT,
                                               span=category_spans[cur_var],
                                               y_range_end=up_bound(
                                                   base_list_data[i][2])))
                    print(
                        '    Appended number of figures in ' + cur_var +
                        ' category: ', len(allPlot))

                    # Append only, save reports later
                    gridplot_figs[cur_var].append(allPlot)

                else:

                    for cur_group in cur_list_data:
                        allPlot.append(
                            drawOnePMLogFigure(cur_group,
                                               int(s),
                                               width=PLOT_WIDTH,
                                               height=PLOT_HEIGHT))
                    print(
                        '    Number of figures in ' + cur_var + ' category: ',
                        len(allPlot))

                    html_file = path.joinpath(
                        args.out_dir, args.prefix + "_" + case_name + "_" +
                        cur_var + ".html")
                    if html_file.exists():
                        # backup if current file exists
                        new_html_file = html_file.with_suffix(
                            html_file.suffix + "." +
                            datetime.now().strftime("_%Y%m%d_%H%M%S"))
                        html_file.rename(new_html_file)
                        print("File: " + html_file.name +
                              " exists. Move it to " + new_html_file.name)

                    output_file(html_file,
                                title=cur_var + ' Analysis for ' +
                                ticket_name + ': ' + case_name)
                    save(column(allPlot))

    # Write single report if comparable
    if comparable:
        tabs = Tabs()
        for cur_var in list_category:
            #print(cur_var, ": ", len(gridplot_figs[cur_var]))

            # to make horizontal style, transpose them before save
            #   NB: we might leave the choice for user to choose the style
            gridplot_figs[cur_var] = list(zip(*gridplot_figs[cur_var]))

            # link panning
            #   NB: we might leave the choice for user to choose x or y
            for figs in gridplot_figs[cur_var]:
                if len(figs) > 1:
                    for fig in reversed(figs):
                        fig.x_range = figs[0].x_range
                        fig.y_range = figs[0].y_range

            gp = gridplot(gridplot_figs[cur_var])

            top_labels = []
            for f in pmlog_files:
                top_labels.append(
                    TextInput(title="",
                              value=f.name,
                              disabled=True,
                              width=PLOT_WIDTH))

            layout = column(row(top_labels), gp)
            tabs.tabs.append(TabPanel(child=layout, title=cur_var.upper()))

        html_file = path.joinpath(args.out_dir, args.prefix + ".html")

        if html_file.exists():
            # backup if current file exists
            new_html_file = html_file.with_suffix(
                html_file.suffix + "." +
                datetime.now().strftime("_%Y%m%d_%H%M%S"))
            html_file.rename(new_html_file)
            print("File: " + html_file.name + " exists. Move it to " +
                  new_html_file.name)
        output_file(html_file, title='Analysis for ' + ticket_name)
        save(tabs, filename=html_file)

    print("It takes", "{time:5.0f}".format(time=(time.time() - start_time)),
          "seconds to run this script.")

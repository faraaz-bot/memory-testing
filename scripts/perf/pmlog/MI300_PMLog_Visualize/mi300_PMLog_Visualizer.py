"""
py mi300_pmlog_visualizer.py -i <input.xlsx>, where "-i" is optional if you want to customize mi300a_pmlog_visualize_config.
For example: py mi300_pmlog_diegest.py -i ./mi300a_pmlog_visuarlize_config.xlsx
This script is to generate the html files to visualize PMLog's Frequency, Power, Temperature and Bandwidth signals
Background:
    MI300A: 4AID+6XCD+3CCD
Input:
    input xlsx file,e.g., mi300a_
    sheet "Overview" defines directory of PMLog files
    sheet "signals"  defines keywords of signals to visualize in category of freq/power/temp/bw
Output:
* For each test case, generate _freq.html, _power.html, _temp.html, _bw.html
Reference:
1. ./mi300a_pmlog_visualize_config.xlsx (worksheet for signal analysis)
Author:  William Li (william.li@amd.com)
Date:
History:
    02/06/2023: publish version 1.0
    01/27/2023: initiate this script based on mi300_PMLog_visualizer.ipynb
"""
import os
import re
import time
import shutil
import argparse
from datetime import datetime
import pandas as pd
from pandas import read_csv, ExcelWriter, DataFrame

#Bokeh libraries and modules
from bokeh.io import  show, output_notebook, output_file, save
from bokeh.plotting import figure
from bokeh.models import Range1d, FactorRange, LinearAxis, Legend, Label, ColumnDataSource, Span, LegendItem
from bokeh.layouts import gridplot, row, column
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral5, Viridis256, Colorblind, Magma256, Turbo256
from bokeh.transform import dodge

__authors__   = "William Li"
__copyright__ = "Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved."
__version__   = "1.0-Version"
__email__     = "william.li@amd.com"
__language__  = "Python 3.10.7"  # run in windows powershell: py
__status__    = "Release"

##global parameters
##Bokeh plot
PLOT_WIDTH=1200
PLOT_HEIGHT=640
MARKER=['asterisk', 'circle', 'circle_cross', 'circle_dot', 'circle_x', 'circle_y', 'cross',
        'dash', 'diamond', 'diamond_cross', 'diamond_dot', 'dot', 'hex', 'hex_dot', 'inverted_triangle',
        'plus', 'square', 'square_cross', 'square_dot', 'square_pin', 'square_x', 'star', 'star_dot',
        'triangle', 'triangle_dot', 'triangle_pin']
COLOR= ['blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
        'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray',
        'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey',
        'dodgerblue', 'firebrick',  'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'limegreen', 'linen', 'magenta']

##initialization
start_time=time.time()
parser = argparse.ArgumentParser()

##single input file with all information
defaultInputFile =".\\mi300a_pmlog_visualize_config.xlsx"
parser.add_argument("-i", "--inputfile",  default=defaultInputFile, help="integrated configuration file (optional)")
args = parser.parse_args()
inputFile=args.inputfile
print("Input file: "+inputFile)
##read configuration first
dfCfg = pd.read_excel(inputFile, sheet_name='Overview',header=None).T
dfCfg.columns=dfCfg.iloc[0]

dfCfg.dropna(inplace=True)
ticketName=dfCfg['Ticket Name'][2]                      ##Title of JIRA ticket
scorecardFileName=dfCfg['ScoreCard'][2]                 ##ScoreCard xlsx
pmLogDirName=dfCfg['Directory of PM Log'][2]            ##diretory of PM, search for pm*.csv
pmLogSampleDuration=dfCfg['PMLog Sample Duration'][2]   ##default is 50ms, in order to notate X-axis
##read keywords
df=pd.read_excel(inputFile, sheet_name='KeyWordDef')
df.pop('Notes')
df.dropna(inplace=True)
dfFigureList=df.sort_values("Figure Index", axis=0, ascending=True)
##initialization done

def genDictFromPMLog(thisList, thisCSV, thisHeader):
    listData=[]
    if not thisList: return [listData, thisHeader]

    for thisCluster in thisList:
        keyWord=thisCluster[0]
        dictData={}
        for thisElement in thisHeader:
            if re.search(re.compile(keyWord, re.IGNORECASE), thisElement):
                ##print(thisElement)
                dictData[thisElement] = thisCSV[thisElement].to_list()
        listData.append([dictData, thisCluster[1]])
        ##remove found items from header list
        for thisElement in dictData.keys():
            thisHeader.remove(thisElement)
    return [listData, thisHeader]

def drawOnePMLogFigure(thisCluster, duration, width, height) :
  thisYLabel = thisCluster[1]
  dictData = thisCluster[0]
  sigName  = list(thisCluster[0].keys())
  ##print(sigName)
  thisData = dictData[sigName[0]]
  dictData['xData'] = list(map(lambda x: x*duration, list(range(0, len(thisData)))))
  mdata = ColumnDataSource(data=dictData)

  TOOLS="pan,wheel_zoom,box_zoom,poly_select,lasso_select,tap,reset,hover,save"
  p = figure(tools=TOOLS, title=thisYLabel+' vs Time', width=width, height=height)
  ##   left-y-axis only
  legendItem=[]
  for thisIdx, thisKey in enumerate(sigName) :
    thisLine=p.line(   x='xData', y=thisKey, line_width=0.9, color=COLOR[thisIdx], source=mdata)
    thisScat=p.scatter(x='xData', y=thisKey, size=10,        color=COLOR[thisIdx], source=mdata, marker=MARKER[thisIdx])
    legendItem.append(LegendItem(label=thisKey, renderers=[thisLine, thisScat], index=thisIdx))
  thisLegend=Legend(items=legendItem, location='top_center', orientation='vertical', label_text_font_size='10px')
  p.add_layout(thisLegend, 'right')
  p.legend.click_policy = "hide"
  p.y_range.start = 0
  p.x_range.range_padding = 0.1
  p.xaxis.axis_label = 'time (ms) at '+str(duration)+' ms/sample'
  p.yaxis.axis_label = thisYLabel
  return p

##main
listAll=os.listdir(pmLogDirName)
listPMLogFiles=[thisItem for thisItem in listAll if re.search('pm.*csv', thisItem)]

firstInstance=0
listCategory=['Freq', 'Power', 'Temp', 'BW', 'Misc']
for thisPMLogFileName in listPMLogFiles:
    temp=thisPMLogFileName.split('.'); caseName=temp[0].replace(" ", "")
    thisPMLog = read_csv(pmLogDirName+'\\'+thisPMLogFileName)
    ##reload PM Log signals
    pmlogHeader=thisPMLog.columns.tolist()
    if firstInstance==0: ##fetch list_<cat> for 1st instance only
        ##generate dictionary for plots
        for thisVar in listCategory: globals()['list'+thisVar]=[]
        for thisTestIndex in range(len(dfFigureList)):
            thisFigure=dfFigureList.iloc[thisTestIndex]
            thisCat    =thisFigure['Category']
            thisKeyword=thisFigure['KeyWord']
            thisLabel  =thisFigure['Y-Axis Label']
            for thisVar in listCategory:
                if re.search(thisVar, thisCat):
                    globals()['list'+thisVar].append([thisKeyword, thisLabel])
                    break
        ##fetch PMLog data into list
        ##listFreqData, newHeader=genDictFromPMLog(listFreq,  thisPMLog, newHeader)
        ##listPowerData,newHeader=genDictFromPMLog(listPower, thisPMLog, newHeader)
        ##listTempData, newHeader=genDictFromPMLog(listTemp,  thisPMLog, newHeader)
        ##listBWData,   newHeader=genDictFromPMLog(listBW,    thisPMLog, newHeader)
        ##listMiscData, newHeader=genDictFromPMLog(listMisc,  thisPMLog, newHeader)
        newHeader=pmlogHeader
        for thisVar in listCategory:
            globals()['list'+thisVar+'Data'],newHeader=genDictFromPMLog(globals()['list'+thisVar],  thisPMLog, newHeader)

        ##save unvisualized signals for reference
        outCSV=pd.DataFrame(newHeader)
        fileName="PMLog_Signals_not_visualized.csv"
        outCSV.to_csv(fileName, index=False, header=False)
        print("PMLog signals not visualized are stored in "+fileName + "; total unvisualized signals="+str(len(newHeader)))
        firstInstance=1
    else:
        for thisVar in listCategory:
            globals()['list'+thisVar+'Data'], newHeader=genDictFromPMLog(globals()['list'+thisVar],  thisPMLog, pmlogHeader)

    print("generating html report for case "+caseName)
    # Plots for Freq, Power, Temp, BW and MISCs
    for thisVar in listCategory:
        allPlot=[]
        listData=globals()['list'+thisVar+'Data']
        if not listData: print(caseName+": No Signals to plot for "+thisVar)
        else:
            for thisGroup in listData:
                allPlot.append(drawOnePMLogFigure(thisGroup, pmLogSampleDuration, width=PLOT_WIDTH, height=PLOT_HEIGHT))
            print('Number of figures in '+thisVar+' category: ', len(allPlot))
            fileName="mi300a_analysis_"+caseName+'_'+thisVar+'.html'
            if(os.path.isfile(fileName)):  # backup if this file exists
                srcf = fileName; dest = fileName+datetime.now().strftime("_%Y%m%d_%H%M%S")
                shutil.move(srcf, dest)
                print("File: "+srcf+" EXISTS. Move it to "+dest)
            output_file(fileName, title=thisVar+' Analysis for '+ticketName+': '+caseName)
            save(column(allPlot))
print("It takes",(time.time()-start_time), "seconds to run this script.")

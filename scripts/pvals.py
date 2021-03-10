#!/usr/bin/python3

import csv
import os, sys, getopt
import numpy as np

def decomment(csvfile):
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw: yield raw

def readdata(filename):
    data = []
    with open(filename, newline='') as fp:
        reader = csv.reader(decomment(fp), delimiter="\t")
        for row in reader:
            dimension = int(row[0])
            size = [int(i) for i in row[1:1+dimension]]
            rdata = []
            for val in row[1 + dimension + 1 + 1:]:
                rdata.append(float(val))
            data.append([size, rdata])
    return data
        
            
def main(argv):
    file0 = None
    file1 = None
    
    try:
        opts, args = getopt.getopt(argv,"ha:b:")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print("help")
            exit(0)
        elif opt in ("-a"):
            file0 = os.path.abspath(arg)
        elif opt in ("-b"):
            file1 = os.path.abspath(arg)

    print("file0", file0)
    print("file1", file1)

    if file0 == None or file1 == None:
        print("please specify files")
        sys.exit(1)

    data0 = readdata(file0)
    data1 = readdata(file1)

    pvals = []
    
    import scipy.stats
    for row0 in data0:
        N = row0[0]
        for row1 in data1:
            if row1[0] == N:
                #print(len(row0[1]), len(row1[1]))
                stat, pm, med, tbl = scipy.stats.median_test(row0[1], row1[1], ties="ignore")
                #w, pw = scipy.stats.wilcoxon(row0[1], row1[1])
                stat, pwu = scipy.stats.mannwhitneyu(row0[1], row1[1])
                stat, pkru = scipy.stats.kruskal(row0[1], row1[1])
                #print(scipy.stats.ttest_ind(row0[1], row1[1]))
                median0 = np.median(row0[1])
                median1 = np.median(row1[1])
                pvals.append([N, pm, median0,median1, median0/median1])

    print("size, p_median_test, median0, median1, speedup")
    for vals in pvals:
        print("\t".join(str(x) for x in vals))
        
if __name__ == "__main__":
    main(sys.argv[1:])

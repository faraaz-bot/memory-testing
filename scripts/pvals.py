#!/usr/bin/python3

import csv
import os, sys, getopt
import numpy as np
import scipy

def decomment(csvfile):
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw: yield raw

def readdata(filename):
    data = []
    with open(filename, newline='') as fp:
        reader = csv.reader(decomment(fp), delimiter="\t")
        for row in reader:
            token = row[0]
            nsmaple = int(row[1])
            rdata = []
            for val in row[2:]:
                rdata.append(float(val))
            data.append([token, rdata])
    return data
        
            
def main(argv):
    file0 = None
    file1 = None
    token = None
    
    try:
        opts, args = getopt.getopt(argv,"ha:b:T:")
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
        elif opt in ("-T"):
            token = arg

    print("file0", file0)
    print("file1", file1)
    print("token", token)

    if file0 == None or file1 == None:
        print("please specify files")
        sys.exit(1)

    data0 = readdata(file0)
    data1 = readdata(file1)

    pvals = []

    vals0 = None
    for row in data0:
        if row[0] == token:
            vals0 = row[1][1:]
            break
    #print(vals0)
    
    vals1 = None
    for row in data1:
        if row[0] == token:
            vals1 = row[1][1:]
            break
    #print(vals1)
    

    stat, pm, med, tbl = scipy.stats.median_test(vals0, vals1, ties="ignore")
    print("median test:", pm)
    
    stat, pwu = scipy.stats.mannwhitneyu(vals0, vals1)
    print("mwu test:", pwu)
    
    stat, pkru = scipy.stats.kruskal(vals0, vals1)
    print("kruskal test:", pkru)
    
    stat, pks = scipy.stats.ks_2samp(vals0, vals1)
    print("ks test:", pks)
    
    stat, pt = scipy.stats.ttest_ind(vals0, vals1)
    print("t-test:", pt)
    
        
if __name__ == "__main__":
    main(sys.argv[1:])

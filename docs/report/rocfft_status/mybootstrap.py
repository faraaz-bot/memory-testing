#!/usr/bin/env python3

import sys
import getopt
import csv
import numpy
import random

def confidence_interval(vals, alpha=0.95, nboot=2000):
    """Compute the alpha-confidence interval for the given values using boot-strap resampling."""
    medians = []
    for iboot in range(nboot):
        resample = []
        for i in range(len(vals)):
            resample.append(vals[random.randrange(len(vals))])
        medians.append(numpy.median(resample))
    medians = sorted(medians)
    low = medians[int(numpy.floor(nboot * 0.5 * (1.0 - alpha)))]
    high = medians[int(numpy.ceil(nboot * (1.0 - 0.5 * (1.0 - alpha))))]
    return low, high

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print("test.py -i <inputfile> -o <outputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("mybootsrap.py -i <inputfile> -o <outputfile>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print("Input file is ", inputfile)
    print("Output file is ", outputfile)

    indat = []
    
    with open(inputfile, newline='') as csvfile:
        datreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        column = []
        for row in datreader:
            label = row[0]
            vals = []
            for idx in range(1, len(row)):
                vals.append(float(row[idx]))
            #print(column)
            indat.append([label, vals])

    outdat = []
    for label, dat in indat:
        low, high = confidence_interval(dat)
        print(label, numpy.median(dat), low, high)
        outdat.append([label, str(numpy.median(dat)), str(low), str(high)])
    
    if outputfile != '':
        with open(outputfile, "w") as outfile:
            for row in outdat:
                outfile.write(" ".join(row))
                outfile.write("\n")
        
        
        
if __name__ == '__main__':
    main(sys.argv[1:])

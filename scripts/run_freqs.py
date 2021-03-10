#!/usr/bin/python3

import specs
import subprocess
import os
import re
import numpy
import tempfile

devicenum = 0


mfreqs = [600, 800, 1000, 1200]
sfreqs = [1000, 1100, 1200, 1300, 1400, 1500]

def get_mclk():
    cmd = ["sudo", "/home/AMD/marobert/atitool", "-clkstatus"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    for line in out.split("\n"):
        searchstr = "    Current Memory Clock      : "
        if line.startswith(searchstr):
            #print(line)
            #print(line[len(searchstr):-3])
            return line[len(searchstr):-3]


data = []
        
for sfreq in sfreqs:
    for mfreq in mfreqs:
        print(sfreq, mfreq)
        fout = tempfile.TemporaryFile(mode="w+")
        ferr = tempfile.TemporaryFile(mode="w+")
        cmd = ["sudo", "/home/AMD/marobert/atitool", "-i=*"  ,"-eng=" + str(sfreq), "-mem=" + str(mfreq)]
        print(" ".join(cmd))
        p =	subprocess.Popen(cmd,stdout=fout, stderr=ferr)
        p.wait()
        #print(p.returncode)

        mclk = get_mclk()
        
        #cmd = ["sudo", "/home/AMD/marobert/atitool", "-i=*","-eng=" + str(sfreq)]
        #p =	subprocess.Popen(cmd,stdout=fout, stderr=ferr)
        #p.wait()
        
        machine_specs = specs.get_machine_specs(devicenum)
        print(machine_specs.sclk, mclk)

        
        
        #get_mclk()
        
        runcmd = ["./clients/staging/rocfft-rider",  "--length", "336", "336", "56", "-N", "10", "--double"]
        p =     subprocess.Popen(runcmd, env=os.environ.copy(), stdout=subprocess.PIPE, universal_newlines=True)
        p.wait()
        cout, cerr = p.communicate()

        
        vals = []
        searchstr = "Execution gpu time: "
        for line in cout.split("\n"):
            if line.startswith(searchstr):
                # Line ends with "ms", so remove that.
                ms_string = line[len(searchstr): -2]
                for val in ms_string.split():
                    vals.append(float(val))
        print(vals)
        # #print(p.returncode)

        datum = [float(machine_specs.sclk[:-3]), float(mclk), numpy.median(vals)]
        print(datum)
        data.append(datum)

print(data)

x= []

print("sclcks:")
for i in range(len(sfreqs)):
    x.append(data[i * len(mfreqs)][0],)
print(x)

y = []
print("mclcks:")
for i in range(len(mfreqs)):
    y.append(data[i][1],)
print(y)

z = []
for i in range(len(data)):
    z.append(data[i][2])
print(z)

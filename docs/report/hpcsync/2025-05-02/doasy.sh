#!/bin/bash

asy -f pdf datagraphs.asy -u'legendlist="single,double";filenames="1024_3_sp.dat,1024_3_dp.dat"' -o m1024.pdf
asy -f pdf datagraphs.asy -u'legendlist="single,double";filenames="2048_3_sp.dat,2048_3_dp.dat"' -o m2048.pdf
asy -f pdf datagraphs.asy -u'legendlist="single,double";filenames="4096_3_sp.dat,4096_3_dp.dat"' -o m4096.pdf
asy 2d.asy

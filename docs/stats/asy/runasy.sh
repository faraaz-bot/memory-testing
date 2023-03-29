#!/bin/bash

asy -f pdf alt.asy
asy -f pdf ran.asy
asy -f pdf sep.asy
asy -f pdf seq.asy

asy fpr.asy -u 'filelist="../data/ixt_hq_107_sep.dat,../data/ixt_hq_107_seq.dat,../data/ixt_hq_107_alt.dat,../data/ixt_hq_107_ran.dat";legendlist="separate,sequential,alternating,random"' -o ixt_hq_107_fpr.pdf -f pdf

asy fpr.asy -u 'filelist="../data/x1000c0s6b0n0_sep.dat,../data/x1000c0s6b0n0_seq.dat,../data/x1000c0s6b0n0_alt.dat,../data/x1000c0s6b0n0_ran.dat";legendlist="separate,sequential,alternating,random"' -o x1000c0s6b0n0_fpr.pdf -f pdf

asy fpr.asy -u 'filelist="../data/x1000c0s5b0n0_sep.dat,../data/x1000c0s5b0n0_seq.dat,../data/x1000c0s5b0n0_alt.dat,../data/x1000c0s5b0n0_ran.dat";legendlist="separate,sequential,alternating,random"' -o x1000c0s5b0n0_fpr.pdf -f pdf

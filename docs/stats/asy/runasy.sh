#!/bin/bash

asy -f pdf alt.asy
asy -f pdf ran.asy
asy -f pdf sep.asy
asy -f pdf seq.asy

: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/ixt-hq-107 -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/ixt-hq-107 -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/ixt-hq-107 -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/ixt-hq-107 -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_ran.dat
'

asy fpr.asy -u 'filelist="../data/ixt-hq-107_sep.dat,../data/ixt-hq-107_seq.dat,../data/ixt-hq-107_alt.dat,../data/ixt-hq-107_ran.dat";legendlist="separate,sequential,alternating,random"' -o ixt_hq_107_fpr.pdf -f pdf

   
: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/x1000c0s6b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s6b0n0_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/x1000c0s6b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s6b0n0_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/x1000c0s6b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s6b0n0_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/x1000c0s6b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s6b0n0_ran.dat
'

asy fpr.asy -u 'filelist="../data/x1000c0s6b0n0_sep.dat,../data/x1000c0s6b0n0_seq.dat,../data/x1000c0s6b0n0_alt.dat,../data/x1000c0s6b0n0_ran.dat";legendlist="separate,sequential,alternating,random"' -o x1000c0s6b0n0_fpr.pdf -f pdf


: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/x1000c0s5b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s5b0n0_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/x1000c0s5b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s5b0n0_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/x1000c0s5b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s5b0n0_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/x1000c0s5b0n0 -o ~/repo/rocfft-misc/docs/stats/data/x1000c0s5b0n0_ran.dat
'

asy fpr.asy -u 'filelist="../data/x1000c0s5b0n0_sep.dat,../data/x1000c0s5b0n0_seq.dat,../data/x1000c0s5b0n0_alt.dat,../data/x1000c0s5b0n0_ran.dat";legendlist="separate,sequential,alternating,random"' -o x1000c0s5b0n0_fpr.pdf -f pdf


: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/crusher -o ~/repo/rocfft-misc/docs/stats/data/crusher_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/crusher -o ~/repo/rocfft-misc/docs/stats/data/crusher_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/crusher -o ~/repo/rocfft-misc/docs/stats/data/crusher_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/crusher -o ~/repo/rocfft-misc/docs/stats/data/crusher_ran.dat
'

asy fpr.asy -u 'filelist="../data/crusher_sep.dat,../data/crusher_seq.dat,../data/crusher_alt.dat,../data/crusher_ran.dat";legendlist="separate,sequential,alternating,random"' -o crusher_fpr.pdf -f pdf



: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/char -o ~/repo/rocfft-misc/docs/stats/data/char_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/char -o ~/repo/rocfft-misc/docs/stats/data/char_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/char -o ~/repo/rocfft-misc/docs/stats/data/char_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/char -o ~/repo/rocfft-misc/docs/stats/data/char_ran.dat
'

asy fpr.asy -u 'filelist="../data/char_sep.dat,../data/char_seq.dat,../data/char_alt.dat,../data/char_ran.dat";legendlist="separate,sequential,alternating,random"' -o char_fpr.pdf -f pdf


: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/rtx2060 -o ~/repo/rocfft-misc/docs/stats/data/rtx2060_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/rtx2060 -o ~/repo/rocfft-misc/docs/stats/data/rtx2060_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/rtx2060 -o ~/repo/rocfft-misc/docs/stats/data/rtx2060_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/rtx2060 -o ~/repo/rocfft-misc/docs/stats/data/rtx2060_ran.dat
'

asy fpr.asy -u 'filelist="../data/rtx2060_sep.dat,../data/rtx2060_seq.dat,../data/rtx2060_alt.dat,../data/rtx2060_ran.dat";legendlist="separate,sequential,alternating,random"' -o rtx2060_fpr.pdf -f pdf



: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/rocher -o ~/repo/rocfft-misc/docs/stats/data/rocher_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/rocher -o ~/repo/rocfft-misc/docs/stats/data/rocher_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/rocher -o ~/repo/rocfft-misc/docs/stats/data/rocher_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/rocher -o ~/repo/rocfft-misc/docs/stats/data/rocher_ran.dat
'

asy fpr.asy -u 'filelist="../data/rocher_sep.dat,../data/rocher_seq.dat,../data/rocher_alt.dat,../data/rocher_ran.dat";legendlist="separate,sequential,alternating,random"' -o rocher_fpr.pdf -f pdf



: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/ixt-hq-107_clk -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_clk_ran.dat
'

asy fpr.asy -u 'filelist="../data/ixt-hq-107_ran.dat,../data/ixt-hq-107_clk_ran.dat";legendlist="dynamic,fixed"' -o ixt-hq-107_clk_fpr.pdf -f pdf



: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/ixt-rack-60 -o ~/repo/rocfft-misc/docs/stats/data/ixt-rack-60_sep.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s seq -b ~/sync/amd/ntrial/ixt-rack-60 -o ~/repo/rocfft-misc/docs/stats/data/ixt-rack-60_seq.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s alt -b ~/sync/amd/ntrial/ixt-rack-60 -o ~/repo/rocfft-misc/docs/stats/data/ixt-rack-60_alt.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s ran -b ~/sync/amd/ntrial/ixt-rack-60 -o ~/repo/rocfft-misc/docs/stats/data/ixt-rack-60_ran.dat
'

asy fpr.asy -u 'filelist="../data/ixt-rack-60_sep.dat,../data/ixt-rack-60_seq.dat,../data/ixt-rack-60_alt.dat,../data/ixt-rack-60_ran.dat";legendlist="separate,sequential,alternating,random"' -o ixt-rack-60_fpr.pdf -f pdf



: '
~/sync/amd/ntrial/ntrials_test.py -m mwu  -B -s ran -b ~/sync/amd/ntrial/ixt-hq-107 -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_ran_mwu.dat
~/sync/amd/ntrial/ntrials_test.py -m ttest  -B -s ran -b ~/sync/amd/ntrial/ixt-hq-107 -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_ran_ttest.dat
'

asy fpr.asy -u 'filelist="../data/ixt-hq-107_ran.dat,../data/ixt-hq-107_ran_mwu.dat,../data/ixt-hq-107_ran_ttest.dat";legendlist="moods,mwu,ttest"' -o ixt-hq-107_fpr_method.pdf -f pdf


: '
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/ixt-hq-107 -a ~/sync/amd/ntrial/crusher -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_crusher_moods.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/ixt-hq-107 -a ~/sync/amd/ntrial/crusher -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_crusher_mwu.dat
~/sync/amd/ntrial/ntrials_test.py -m moods  -B -s sep -b ~/sync/amd/ntrial/ixt-hq-107 -a ~/sync/amd/ntrial/crusher -o ~/repo/rocfft-misc/docs/stats/data/ixt-hq-107_crusher_ttest.dat
'
asy fnr.asy -u 'filelist="../data/ixt-hq-107_crusher_moods.dat,../data/ixt-hq-107_crusher_mwu.dat,../data/ixt-hq-107_crusher_ttest.dat";legendlist="moods,mwu,ttest"' -o ixt-hq-107_crusher_fnr.pdf -f pdf

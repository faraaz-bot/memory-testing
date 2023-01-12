import graph;
import stats;

size(400,200,IgnoreAspect);

scale(Log, Linear);
//scale(Linear, Linear);


import utils;

string filelist = "";

string token = "";

usersetting();

if(filelist == "") {
    filelist = getstring("comma-separated filenames:");
}

string[] filenames = listfromcsv(filelist);
write("filenames: ", filenames);

if(token == "") {
    token = getstring("token:");
}

write("token: ", token);

data[][] inputs = new data[filenames.length][];
int[] tokenidx = new int[filenames.length];
for(int ifile = 0; ifile < filenames.length; ++ifile) {
    inputs[ifile] = readfile(filenames[ifile]);
    tokenidx[ifile] = -1;
    for(int i = 0; i < inputs[ifile].length; ++i) {
        if(inputs[ifile][i].token == token) {
            tokenidx[ifile] = i;
            break;
        }
    }
}

// TODO: the different histograms use different bin boundaries, so we
// should maybe unify that somehow.

// Take the log of the data (maybe it's log-normal?)
if(false) {
    for(int ifile = 0; ifile < filenames.length; ++ifile) {
        int idx = tokenidx[ifile];
        for(int i = 0; i <inputs[ifile][idx].vals.length; ++i) {
            inputs[ifile][idx].vals[i] = exp(inputs[ifile][idx].vals[i]);
        }
    }
}

// Get the bin count and the range to match all the data.
real maxval = -inf;
real minval = +inf;
int Nbins = bins(inputs[0][tokenidx[0]].vals);
for(int ifile = 0; ifile < filenames.length; ++ifile) {
    int idx = tokenidx[ifile];
    Nbins = max(Nbins, bins(inputs[ifile][idx].vals));
    minval = min(min(inputs[ifile][idx].vals), minval);
    maxval = max(max(inputs[ifile][idx].vals), maxval);
}
Nbins *= 4;

write("Nbins: ", Nbins);
write("minval: ", minval);
write("maxval: ", maxval);

//maxval = 0.08;

for(int ifile = 0; ifile < filenames.length; ++ifile) {
    int idx = tokenidx[ifile];
    histogram(inputs[ifile][idx].vals,
              minval,
              maxval,
              Nbins, normalize=true, low=0, Pen(ifile)+opacity(0.5), black, bars=true);
}

xaxis("time (ms)",BottomTop,LeftTicks);
yaxis("count",LeftRight,RightTicks(trailingzero));

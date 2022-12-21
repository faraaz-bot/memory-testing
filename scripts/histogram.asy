import graph;
import stats;

size(400,200,IgnoreAspect);

scale(Log, Linear);

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

int Nbins = 8 * bins(inputs[0][tokenidx[0]].vals);
write("Nbins: ", N);
                         
for(int ifile = 0; ifile < filenames.length; ++ifile) {
    int idx = tokenidx[ifile];
    histogram(inputs[ifile][idx].vals,
              min(inputs[ifile][idx].vals),
              max(inputs[ifile][idx].vals), Nbins, normalize=true, low=0, Pen(ifile)+opacity(0.5), black, bars=true);
}

xaxis("time (ms)",BottomTop,LeftTicks);
yaxis("count",LeftRight,RightTicks(trailingzero));

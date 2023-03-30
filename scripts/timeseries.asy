import graph;
import stats;

size(800,400,IgnoreAspect);

//scale(Log, Linear);
scale(Linear, Linear);


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


for(int ifile = 0; ifile < filenames.length; ++ifile) {
    int idx = tokenidx[ifile];
    int N = inputs[ifile][idx].vals.length;
    pair[] pvals = new pair[N];
    for(int i = 0; i < N; ++i) {
        pvals[i] = (i, inputs[ifile][idx].vals[i]);
        dot(pvals[i]);
    }
    draw(graph(pvals), invisible);
    //write(inputs[ifile][idx].vals);
}

xaxis("iteration",BottomTop,LeftTicks);
yaxis("time (ms)",LeftRight,RightTicks(trailingzero));

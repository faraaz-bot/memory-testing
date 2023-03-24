size(15cm, 12cm, IgnoreAspect);

import utils;

import graph;

scale(Linear,Linear);

import quartiles;

import whiskerplot;

string filelist = "";

string token = "";

string legends = "";
string xaxislabel = "";

//string token = "complex_inverse_len_256_double_ip_batch_1_istride_1_CI_ostride_1_CI_idist_256_odist_256_ioffset_0_0_ooffset_0_0";

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

real[][] plotdata = new real[][];

for(int ifile = 0; ifile < filenames.length; ++ifile) {
    plotdata.push(new real[]);
    int idx = tokenidx[ifile];
    plotdata[ifile] = inputs[ifile][idx].vals;
}

string[] legendlist = new string[];
if(legends != "" ) {
    legendlist = listfromcsv(legends);
}
// legendlist.push("a");
// legendlist.push("b");

//write(abdata);

string ylegend = "time (ms)";

//xaxislabel = texify(token);

whiskerplot(plotdata, legendlist, xaxislabel, ylegend);

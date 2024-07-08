import graph;
import stats;

size(400,200,IgnoreAspect);

//scale(Log, Linear);
scale(Linear, Linear);


struct data{
    string token;
    int dim;
    int[] lengths;
    int batch;
    int nsample;
    real[] vals;
};

data[] readfile(string filename) {
    data[] input;

    file fin = input(filename).line();

    bool moretoread = true;
    while(moretoread) {
        data dat;
        
        string line = fin;

        // Separate the token from the data:
        int pos = find(line, '\t', 0);
        string token = substr(line, 0, pos);
        write("token: ", token);
        dat.token = token;
        string vals = substr(line, pos + 1, -1);
        //write("vals: ", vals);

        // Get the data:
        int lastpos = 0;
        pos = find(vals, '\t', lastpos);
        dat.nsample = (int)substr(vals, lastpos, pos - lastpos);
        lastpos = pos > 0 ? pos + 1 : -1;
        //write("nsample: ", dat.nsample);

        while(lastpos != -1) {
            int pos = find(vals, '\t', lastpos);
            string nsample = substr(vals, lastpos, pos - lastpos);
            real val = (real)substr(vals, lastpos, pos - lastpos);
            //write(val);
            dat.vals.push(val);
            lastpos = pos > 0 ? pos + 1 : -1;
        }


        input.push(dat);
        
        if(eof(fin)) {
	    moretoread = false;
	    break;
        }

    }

    return input;
}


// Create an array from a comma-separated string
string[] listfromcsv(string input)
{
    string list[] = new string[];
    int n = -1;
    bool flag = true;
    int lastpos;
    while(flag) {
        ++n;
        int pos = find(input, ",", lastpos);
        string found;
        if(lastpos == -1) {
            flag = false;
            found = "";
        }
        found = substr(input, lastpos, pos - lastpos);
        if(flag) {
            list.push(found);
            lastpos = pos > 0 ? pos + 1 : -1;
        }
    }
    return list;
}

//import utils;

string filelist = "";

string token = "";

string labelstring = "";

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

for(int ifile = 0; ifile < filenames.length; ++ifile) {
    int idx = tokenidx[ifile];
    //write(inputs[ifile][idx].vals);
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
//Nbins *= 4;

write("Nbins: ", Nbins);
write("minval: ", minval);
write("maxval: ", maxval);

maxval = 1;//0.08;

for(int ifile = 0; ifile < filenames.length; ++ifile) {
    int idx = tokenidx[ifile];
    pen p = ifile == 0 ? red : green;
    histogram(data = inputs[ifile][idx].vals,
              a = minval,
              b = maxval,
              n = Nbins,
	      normalize = false,
	      low = 0,
	      fillpen = p+opacity(0.5),
	      drawpen = black,
	      bars=true);
    //attach(legend());
}


//label(

xaxis("time (ms)", BottomTop, LeftTicks);
yaxis("count",LeftRight,RightTicks(trailingzero));


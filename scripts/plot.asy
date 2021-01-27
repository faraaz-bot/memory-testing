size(15cm, 12cm, IgnoreAspect);

import graph;
import stats;

import whiskerplot;

string file_a = "/home/mal/mnt/char/repo/rocfft/scripts/perf/40vs41/40/radix3_dim1_double_n1_c2c_inplace.dat";
string file_b = "/home/mal/mnt/char/repo/rocfft/scripts/perf/40vs41/41/radix3_dim1_double_n1_c2c_inplace.dat";


usersetting();

struct data{
    int dim;
    int[] lengths;
    int batch;
    int nsample;
    real[] vals;
};

// Read file a.


data[] readfile(string filename) {
        data[] input;
        
        file fin = input(filename);

        fin.line();

        bool go = true;
        while(go) 
        {
            if(eof(fin))
            {
                break;
            }
            data temp;
            temp.dim = fin;
            if(!initialized(temp.dim))
            {
                go = false;
                break;
            }
            //write(temp.dim);
    
            for(int i=0; i < temp.dim; ++i) {
                int val = fin;
                temp.lengths.push(val);
            }
            //write(temp.lengths[0]);
    
            temp.batch = fin;
            temp.nsample = fin;
            for(int i = 0; i < temp.nsample; ++i)
            {
                temp.vals.push(fin);
            }

            input.push(temp);
        } 

        
        return input;
    }

data[] data_a = readfile(file_a);

for(int i = 0; i < data_a.length; ++i)
{
    write(data_a[i].lengths[0]);
}

data[] data_b = readfile(file_b);

for(int i = 0; i < data_b.length; ++i)
{
    write(data_b[i].lengths[0]);
}

int mylength = 1594323;

int idxa = 0;
int idxb = 0;

for(int i = 0; i < data_a.length; ++i)
{
    if(data_a[i].lengths[0] == mylength)
        idxa = i;
}


for(int i = 0; i < data_b.length; ++i)
{
    if(data_b[i].lengths[0] == mylength)
        idxb = i;
}


real[][] abdata = new real[][];

abdata.push(new real[]);
for(int i = 0; i < data_a[idxa].vals.length; ++i)
{
    abdata[0].push(data_a[idxa].vals[i]);
}

//write(abdata[0]);

abdata.push(new real[]);
for(int i = 0; i < data_b[idxb].vals.length; ++i)
{
    abdata[1].push(data_b[idxb].vals[i]);
}

//write(abdata[1]);

string[] legendlist = new string[];
legendlist.push("4.0");
legendlist.push("4.1");

//write(abdata);

string ylegend = "time (s)";

whiskerplot(abdata, legendlist, "1D c2c length 1594323", ylegend);

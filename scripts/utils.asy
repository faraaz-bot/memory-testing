
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

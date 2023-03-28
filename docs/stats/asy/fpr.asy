import graph;

size(200, 150, IgnoreAspect);

scale(Log,Linear);

string filelist = "";
string legendlist = "";

usersetting();

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

string[] filenames;
if(filelist != "") {
    filenames = listfromcsv(filelist);
}

string[] legends;
if(legendlist != "") {
    legends = listfromcsv(legendlist);
}

for(int i = 0; i < filenames.length; ++i) {
    write(filenames[i]);
    file in=input(filenames[i]).line();
    real[][] a=in;
    a=transpose(a);

    real[] x=a[0];
    real[] y=a[1];
    
    draw(graph(x,y), Pen(i), texify(legends[i]));
}

xaxis("$N$",BottomTop,LeftTicks);
yaxis("false positive rate",LeftRight,RightTicks);


attach(legend(), point(E), 30*E);

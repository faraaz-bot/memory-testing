import graph;
import utils;

//asy datagraphs -u "xlabel=\"\$\\bm{u}\cdot\\bm{B}/uB$\"" -u "doyticks=false" -u "ylabel=\"\"" -u "legendlist=\"a,b\""

bool legendNW=false;

texpreamble("\usepackage{bm}");

size(300,175,IgnoreAspect);

scale(Log,Linear);
//scale(Log,Log);

bool dolegend=true;

string filenames = "";
string filename;
string legendlist="";
int n=-1;
bool flag=true;
int lastpos;

bool doxticks=true;
bool doyticks=true;
string xlabel="Number of GPUs";
string ylabel="time (ms)";

usersetting();

if(filenames == "" )
  filenames=getstring("filenames");

bool myleg=((legendlist == "") ? false: true);
string[] legends=set_legends(legendlist);

while(flag) {
  ++n;
  int pos=find(filenames,",",lastpos);
  if(lastpos == -1) {filename=""; flag=false;}
  filename=substr(filenames,lastpos,pos-lastpos);

  if(flag) {
    lastpos=pos > 0 ? pos+1 : -1;

    file fin = input(filename).line();
    real[][] a = fin.dimension(0,0);
    a = transpose(a);
    real[] x = a[0];
    real[] y = a[1];
    real[] lo = a[2];
    real[] hi = a[3];
    pen p = Pen(n);
    if(n == 1) p += dashed;
    if(n == 2) p=darkgreen+Dotted;
    
    marker mark1 = marker(scale(0.6mm) * polygon(3 + n), Draw(p + solid));
    draw(graph(x, y), p, myleg ? legends[n] : texify(filename), mark1);

    real[] dpx;
    real[] dmx;
    real[] dpy;
    real[] dmy;
    
    for(int idx = 0; idx < y.length; ++idx) {
        dpx.push(0);
        dmx.push(0);
        dpy.push(-y[idx] + hi[idx]);
        dmy.push(y[idx] - hi[idx]);
    }
    errorbars(x, y, dpx, dpy, dmx, dmy, p + solid);

  }
}

if(doxticks)
    xaxis(xlabel,BottomTop,LeftTicks(DefaultFormat, new real[] {8, 16, 32, 64, 128, 256}));
else
  xaxis(xlabel);
if(doyticks)
  yaxis(ylabel,LeftRight,RightTicks);
else
  yaxis(ylabel,LeftRight);
if(dolegend) {
  if(legendNW) 
    attach(legend(),point(plain.NW),10S + 10E);
  else
    attach(legend(),point(plain.SE),60N + 40W);
}

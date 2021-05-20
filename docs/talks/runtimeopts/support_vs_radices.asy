import graph;

size(200,150,IgnoreAspect);

//scale(Linear,Log);

string filename;
filename=getstring("external data:");
file fin=input(filename).line();
real[][] a=fin.dimension(0,0);
a=transpose(a);
real[] x=a[0];
real[] y=a[1];
draw(graph(x,y),texify("data file"));

xaxis("Number of Radices",BottomTop,LeftTicks);
yaxis("Supported Sizes",LeftRight,RightTicks);

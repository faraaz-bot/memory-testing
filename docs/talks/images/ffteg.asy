import graph;

picture pic;
real xsize = 200, ysize = 140;
size(pic, xsize, ysize, IgnoreAspect);
real f0(real x) { return 0.15cos(x);}
real f1(real x) { return 0.25*cos(2x);}
real f2(real x) { return -0.3*cos(3x);}
real f3(real x) { return 0.1*cos(4x);}

real f(real x) { return f0(x) + f1(x) + f2(x) + f3(x);}

draw(pic, graph(pic, f, 0, 2pi));

string[] zeropitwopi = {"$0$","$\pi$","$2\pi$"};
     
xaxis(pic, Bottom, LeftTicks(new string(real x) { return zeropitwopi[round(x / pi)];}, new real[] {0, pi, 2pi}));
// xaxis(pic, BottomTop,LeftTicks(DefaultFormat,
// 			       new real[] {0,pi,2pi}));
yaxis(pic,LeftRight);

scale(pic, true);

picture pic2;
size(pic2, xsize, ysize, IgnoreAspect);

int ipen = 0;
draw(pic2, graph(pic2, f0, 0, 2pi), Pen(++ipen));
draw(pic2, graph(pic2, f1, 0, 2pi), Pen(++ipen));
draw(pic2, graph(pic2, f2, 0, 2pi), Pen(++ipen));
draw(pic2, graph(pic2, f3, 0, 2pi), Pen(++ipen));


xaxis(pic2, Bottom, LeftTicks(new string(real x) { return zeropitwopi[round(x / pi)];}, new real[] {0, pi, 2pi}));
//labelx(pic2, "$\ldots$",1);
scale(pic2,true);

// Fit pic to W of origin:
add(pic.fit(), (0,0), W);

draw((0,0)--(10mm,0), Arrows, L="$\mathcal{F}$");

// Fit pic2 to E of (5mm,0):
add(pic2.fit(), (10mm,0), E);

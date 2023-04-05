size(300, 300);


import graph;

scale(false);

real itmax = 3.0;

real f(real x) {return max(0, x/itmax);}
draw(graph(f, 0.0, itmax), invisible);



int nbox = 4;
real[] p={0.5, 0.3, 0.2, 0};

for(int ibox = 0; ibox < nbox; ++ibox) {
    real lside = ibox * itmax / nbox;
    real rside = (ibox + 1) * itmax / nbox;
    fill(box((lside, 0), (rside, p[ibox])), red + opacity(0.5));
    fill(box((lside, p[ibox]), (rside, 1)), blue + opacity(0.5));
}

labely("$0$",0);
labely("$1$",1);

xlimits(0,itmax);
ylimits(0,1);


xaxis("iteration",Bottom, Arrow, xmax=itmax+0.2);
yaxis("random weight", RightTicks(1), ymin=0, ymax=1);

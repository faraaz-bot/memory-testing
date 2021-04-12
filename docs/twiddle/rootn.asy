import graph;
size(150,0);

real f(real x) {return exp(x);}
pair F(real x) {return (x,f(x));}

//draw(graph(f,-4,2,operator ..),red);

real bound = 1.1;
xaxis(-bound, bound);
yaxis(-bound, bound);

real N = 5;
for(int k = 0; k < N; ++k)
{
    real arg = k*2*pi/N;
    dot((cos(arg), sin(arg)));
}

real x(real t) {return cos(2pi*t);}
real y(real t) {return sin(2pi*t);}

draw(graph(x,y,0,1));

//labely(1,E);
label("$i$",(0,1),SE);
label("$-i$",(0,-1),SE);
label("$1$",(1,0),SE);
label("$-1$",(-1,0),SE);

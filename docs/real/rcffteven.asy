//size(300,200,IgnoreAspect);
size(800,400);

usepackage("amsfonts");
usepackage("amsfonts");
usepackage("physics");

// N must be even
int N = 16;

real boxw = 1.0;
real boxh = 0.5;

pen arrowPen = grey;
pen labelPen = red;

void draw_array(pair pos, real boxw, real boxh, int N) 
{
    draw(pos--pos+(N*boxw,0));
    for(int i = 0; i <= N; ++i) {
        pair offset = (i * boxw, 0);
        pair down = (0, -boxh);
        draw(pos+offset -- pos+offset+down);
    }
    pos = pos + (0, -boxh);
    draw(pos--pos+(N * boxw,0));
}

pair Rmiddle = (0.5*boxw, -0.5*boxh);
pair Cmiddle = (boxw, -0.5*boxh);
pair Ctop = (boxw, 0);
pair Cbottom = (boxw, -boxh);
pair CmiddleL = (0.5*boxw, -0.5*boxh);
pair CmiddleR = (1.5*boxw, -0.5*boxh);
pair top = (0,0.5*boxh);
pair bottom = (0,-0.5*boxh);

int halfN = floor(N/2);
int Ncomplex = ceil(N/2) + 1;

pair bx = (0,0);

draw_array(bx, boxw, boxh, N);

pair bz = bx + (0,-1);

draw_array(bz, 2*boxw, boxh, halfN);

for(int i = 0; i < halfN; ++i){
    pair Cioff = i * (2*boxw,0);
    
    pair Rieven = 2* i * (boxw,0);
    label("$x_{"+(string)(2*i)+"}$" ,bx + Rieven + Rmiddle);
    draw(bx+Rieven+Rmiddle+bottom--bz+Cioff + CmiddleL + top, arrowPen,
         EndArrow);
    pair Riodd = (2* i + 1) * (boxw,0);
    label("$x_{"+(string)(2*i+1)+"}$" ,bx + Riodd + Rmiddle);
    draw(bx+Riodd+Rmiddle+bottom--bz+Cioff + CmiddleR + top, arrowPen,
         EndArrow);
    
    label("$z_{"+(string)(i)+"}$" ,bz + Cioff + Cmiddle);
}

pair bZ = bz + (0,-2);

draw_array(bZ, 2*boxw, boxh, halfN);

real ampl=0.5;
draw(brace(bz +(N*boxw, -boxh),bz +(0, -boxh),ampl), arrowPen);
draw(bz +(0.5*N*boxw, -boxh-ampl)--bZ +(0.5*N*boxw, ampl), arrowPen,
     L=Label("$\mathcal{F}$", E, position=MidPoint, labelPen), 
     EndArrow);
draw(brace(bZ +(N*boxw, 0), bZ, -ampl), arrowPen);

pair bX = bZ + (0,-4);

draw_array(bX, 2*boxw, boxh, Ncomplex);

real loffset=0.28;

label("$Z_0$" ,bZ + Cmiddle);
for(int i = 1; i < halfN; ++i){
    pair Cioff = i * (2*boxw,0);
    label("$Z_{"+(string)(i)+"}$" ,bZ + Cioff + Cmiddle);
    pair Cion = (halfN - i) * (2*boxw,0);
    pair p0, p1;
    p0 = bZ + Cbottom + Cioff;
    p1 = bX + Ctop + Cioff;
    draw(p0 -- p1, arrowPen,
         L=rotate(90)*Label("$Z_{"+(string)(i)
                            + "}\left(1-\omega_N^"+(string)i+"\right)/2$",
                            (0,0), position=loffset, labelPen)
         , EndArrow);
    p0 = bZ+Cbottom + Cion;
    p1 = bX + Ctop + Cioff;
    write(p1 - p0);
    write(atan2((p1-p0).x,(p1-p0).y));
    real angle = degrees(atan2((p1-p0).y,(p1-p0).x));

    if(angle < -90)
        angle += 180;
    
    write(angle);
       
    draw(p0 -- p1, arrowPen,
         L=rotate(angle)*Label("$Z_{"+(string)(halfN - i)
                            + "}\left(1+\omega_N^"+(string)i+"\right)/2$",
                               (0,0), position=1.0-loffset, labelPen)
         , EndArrow);
}

draw(bZ+Cbottom--bX+Ctop, arrowPen,
     L=rotate(90)*Label("$\Re{Z_0} + \Im{Z_0}$", (0,0), position=loffset, labelPen),
     EndArrow);
draw(bZ+Cbottom--bX+Ctop+(halfN*2*boxw), arrowPen,
     L=Label("$\Re{Z_0} - \Im{Z_0}$", NE, position=0.9, labelPen),
     EndArrow);

for(int i = 0; i < Ncomplex; ++i){
    pair Cioff = i * (2*boxw,0);
    label("$X_{"+(string)(i)+"}$" ,bX + Cioff + Cmiddle);
}

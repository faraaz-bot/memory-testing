//size(300,200,IgnoreAspect);
size(1000,400);
texpreamble("\usepackage{physics}");


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


int N = 8;
int Nhalf = floor(N/2);
int Ncomplex = floor(N/2) + 1;

real boxw = 1.0;
real boxh = 0.5;
real yoffset = 2 + Ncomplex * 2 * boxw;

pair Rmiddle = (0.5*boxw, -0.5*boxh);
pair Cmiddle = (boxw, -0.5*boxh);
pair CmiddleL = (0.5*boxw, -0.5*boxh);
pair CmiddleR = (1.5*boxw, -0.5*boxh);
pair Top = (0,0.5*boxh);
pair Bottom = (0,-0.5*boxh);

pen arrowPen = grey;
pen labelPen = red;

pair bX = (0,0);
draw_array(bX, 2* boxw, boxh, Ncomplex);

pair bY = (yoffset,0);
draw_array(bY, 2*boxw, boxh, Ncomplex);
for(int i = 0; i < Ncomplex; ++i){
  pair Cioff = i * (2*boxw,0);
  string si = (string)(i);
  label("$X_{"+si+"}$" ,bX + Cioff + Cmiddle);
  label("$Y_{"+si+"}$" ,bY + Cioff + Cmiddle);
}

pair bZ = bX + (0,-2);
draw_array(bZ, 2*boxw, boxh, N);
pair C0off = 0 * (2*boxw,0);
string s0 = (string)(0);
label("$Z_{"+s0+"} = X_{"+s0+"} + i Y_{" +s0+ "}$" ,bZ + C0off + Cmiddle);
draw(bX + Cmiddle + Bottom--bZ + Cmiddle + Top,arrowPen, EndArrow);
draw(bY + Cmiddle + Bottom--bZ + Cmiddle + Top,arrowPen, EndArrow);

bool even = (N % 2) == 0;
int stop = even ? floor(N/2) : floor(N/2) + 1;
for(int i = 1; i < stop; ++i){
  pair Cioff = i * (2*boxw,0);
  string si = (string)(i);
  label("$Z_{"+si+"} = X_{"+si+"} + i Y_{" +si+ "}$" ,bZ + Cioff + Cmiddle);
  draw(bX +Cioff + Cmiddle + Bottom--bZ + Cioff + Cmiddle + Top,arrowPen,
       EndArrow);
  draw(bY +Cioff + Cmiddle + Bottom--bZ + Cioff + Cmiddle + Top,arrowPen,
       EndArrow);
  
  pair CNioff = (N-i) * (2*boxw,0);
  string sNi = (string)(N-i);
  label("$Z_{"+sNi+"} = X_{"+si+"}^* + i Y_{" +si+ "}^*$" ,bZ+ CNioff+ Cmiddle);
  draw(bX +Cioff + Cmiddle + Bottom--bZ + CNioff + Cmiddle + Top,arrowPen,
       EndArrow);
  draw(bY +Cioff + Cmiddle + Bottom--bZ + CNioff + Cmiddle + Top,arrowPen,
       EndArrow);
  write(i, N-i);
}

if(even) {
  pair Cioff = floor(N/2) * (2*boxw,0);
  draw(bX + Cioff + Cmiddle + Bottom--bZ + Cioff + Cmiddle + Top, arrowPen,
       EndArrow);
  draw(bY + Cioff + Cmiddle + Bottom--bZ + Cioff + Cmiddle + Top, arrowPen,
       EndArrow);
  string si = (string)(floor(N/2));
  label("$Z_{"+si+"} = X_{"+si+"}^* + i Y_{" +si+ "}^*$" ,bZ+ Cioff+ Cmiddle);
} 

pair bz = bZ + (0,-2);
draw_array(bz, 2*boxw, boxh, N);

real ampl=0.5;
draw(brace(bZ +(N*2*boxw, -boxh),bZ +(0, -boxh),ampl), arrowPen);

draw(bZ +(N*boxw, -boxh-ampl)--bz +(N*boxw, ampl),arrowPen,
     L=Label("$\mathcal{F}^{-1}$", E, position=MidPoint, labelPen),
     EndArrow);

draw(brace(bz +(N*2*boxw, 0),bz,-ampl), arrowPen);

for(int i = 0; i < N; ++i){
  pair Cioff = i * (2*boxw,0);
  label("$z_{"+(string)(i)+"}$" ,bz + Cioff + Cmiddle);
}

pair bx = bz + (0, -2);
draw_array(bx, boxw, boxh, N);

pair by = bx + (yoffset, 0);
draw_array(by, boxw, boxh, N);

for(int i = 0; i < N; ++i) {
  pair Cioff = i * (2 * boxw,0);
  pair Rioff = i * (boxw,0);
  string si = (string)(i);
  draw(bz + Cioff + CmiddleL + Bottom--bx + Rioff + Rmiddle + Top, arrowPen,
       EndArrow);
  draw(bz + Cioff + CmiddleR + Bottom--by + Rioff + Rmiddle + Top, arrowPen,
       EndArrow);

  label("$x_{"+si+"}$" ,bx+ Rioff+ Rmiddle);
  label("$y_{"+si+"}$" ,by+ Rioff+ Rmiddle);
}

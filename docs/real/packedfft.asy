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


real boxw = 1.0;
real boxh = 0.5;
int N = 9;
int Ncomplex = ceil(N/2) + 1;
real yoffset = 2 + Ncomplex * 2 * boxw;


pen arrowPen = grey;
pen labelPen = red;

pair bx = (0,0);

draw_array(bx, boxw, boxh, N);

pair by = (yoffset,0);
draw_array(by, boxw, boxh, N);

pair bz = (0,-2);
draw_array(bz, 2*boxw, boxh, N);


pair Rmiddle = (0.5*boxw, -0.5*boxh);
pair Cmiddle = (boxw, -0.5*boxh);
pair CmiddleL = (0.5*boxw, -0.5*boxh);
pair CmiddleR = (1.5*boxw, -0.5*boxh);
pair Top = (0,0.5*boxh);
pair Bottom = (0,-0.5*boxh);

for(int i = 0; i < N; ++i){
  pair Cioff = i * (2*boxw,0);
  string si = (string)(i);
  label("$z_{"+si+"} = x_{"+si+"} + i y_{" +si+ "}$" ,bz + Cioff + Cmiddle);
}

for(int i = 0; i < N; ++i) {
    pair Rioff = i * (boxw,0);
    pair Cioff = i * (2*boxw,0);
    //dot(bx + Rioff,blue);
    string si = (string)(i);
    label("$x_{"+si+"}$" ,bx + Rioff + Rmiddle);
    label("$y_{"+si+"}$" ,by + Rioff + Rmiddle);
    draw(bx+Rioff+Rmiddle+Bottom--bz+Cioff + CmiddleL + Top,arrowPen, EndArrow);
    draw(by+Rioff+Rmiddle+Bottom--bz+Cioff + CmiddleR + Top,arrowPen, EndArrow);
}
pair bZ = bz + (0,-2);
draw_array(bZ, 2*boxw, boxh, N);

for(int i = 0; i < N; ++i){
  pair Cioff = i * (2*boxw,0);
  label("$Z_{"+(string)(i)+"}$" ,bZ + Cioff + Cmiddle);
}

real ampl=0.5;
draw(brace(bz +(N*2*boxw, -boxh),bz +(0, -boxh),ampl), arrowPen);

draw(bz +(N*boxw, -boxh-ampl)--bZ +(N*boxw, ampl),arrowPen,
     L=Label("$\mathcal{F}$", E, position=MidPoint, labelPen),
     EndArrow);

draw(brace(bZ +(N*2*boxw, 0),bZ,-ampl), arrowPen);

pair bX = bZ + (0, -3);
draw_array(bX, 2*boxw, boxh, Ncomplex);

pair bY = bX + (yoffset, 0);
draw_array(bY, 2*boxw, boxh, Ncomplex);

for(int i = 0; i < Ncomplex; ++i) {
    pair Cioff = i * (2*boxw,0);
    pair CNioff = (N-i) * (2*boxw,0);
    draw(bZ + Cioff +Cmiddle + Bottom -- bX + Cioff+Cmiddle+Top,
	 arrowPen, EndArrow);
    if(i != 0)
      draw(bZ + CNioff +Cmiddle + Bottom -- bX + Cioff+Cmiddle+Top,
	   arrowPen, EndArrow);
    
    draw(bZ + Cioff +Cmiddle +Bottom-- bY + Cioff+Cmiddle+Top,
	 arrowPen, EndArrow);
    if(i != 0)
      draw(bZ + CNioff +Cmiddle +Bottom -- bY + Cioff+Cmiddle+Top,
	   arrowPen, EndArrow);
    string si = (string)(i);
    string sNi = (string)(N-i);
    string xrhs = (i == 0) ? "\Re{Z_0}$" :  "\frac{Z_{"+si+"} + Z_{"+sNi+"}^*}{2}$";
    string yrhs = (i == 0) ? "\Im{Z_0}$" :  "\frac{Z_{"+si+"} - Z_{"+sNi+"}^*}{2i}$";
    label("$X_{"+si+"}="+ xrhs,
	  bX + Cioff + Cmiddle);
    label("$Y_{"+si+"}=" + yrhs,
	  bY + Cioff + Cmiddle);
}

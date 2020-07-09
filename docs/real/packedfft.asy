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
int Ncomplex = floor(N/2) + 1;
real yoffset = Ncomplex * 2 * boxw;


pen arrowPen = grey;
pen labelPen = red;

pair bx = (0,0);

draw_array(bx, boxw, boxh, N);

pair by = (yoffset,0);
draw_array(by, boxw, boxh, N);


pair Rmiddle = (0.5*boxw, -0.5*boxh);
pair Cmiddle = (boxw, -0.5*boxh);
pair CmiddleL = (0.5*boxw, -0.5*boxh);
pair CmiddleR = (1.5*boxw, -0.5*boxh);
pair Top = (0,0.5*boxh);
pair Bottom = (0,-0.5*boxh);

// for(int i = 0; i < N; ++i){
//   pair Cioff = i * (2*boxw,0);
//   string si = (string)(i);
//   label("$z_{"+si+"} = x_{"+si+"} + i y_{" +si+ "}$" ,bz + Cioff + Cmiddle);
// }

for(int i = 0; i < N; ++i) {
    pair Rioff = i * (boxw,0);
    pair Cioff = i * (2*boxw,0);
    //dot(bx + Rioff,blue);
    string si = (string)(i);
    label("$x_{"+si+"}$" ,bx + Rioff + Rmiddle);
    label("$y_{"+si+"}$" ,by + Rioff + Rmiddle);
    // draw(bx+Rioff+Rmiddle+Bottom--bz+Cioff + CmiddleL + Top,arrowPen, EndArrow);
    // draw(by+Rioff+Rmiddle+Bottom--bz+Cioff + CmiddleR + Top,arrowPen, EndArrow);
}

pair bZre = bx + (0,-2);
draw_array(bZre, boxw, boxh, N);

pair bZim = bZre + (yoffset, 0); 
draw_array(bZim, boxw, boxh, N);

for(int i = 0; i < N; ++i) {
    pair Rioff = i * (boxw,0);
    pair Cioff = i * (2*boxw,0);
    //dot(bx + Rioff,blue);
    string si = (string)(i);
    label("$\Re{Z_{"+si+"}}$" ,bZre + Rioff + Rmiddle);
    label("$\Im{Z_{"+si+"}}$" ,bZim + Rioff + Rmiddle);
}

real ampl=0.5;
draw(brace(by +(N*boxw, -boxh), bx +(0, -boxh),ampl), arrowPen);

real xarrowoffset = 0.5*(yoffset - N * boxw);
draw(bx +(xarrowoffset + N*boxw, -boxh-ampl)--bZre +(xarrowoffset + N*boxw, ampl),arrowPen,
     L=Label("$\mathcal{F}$", E, position=MidPoint, labelPen),
     EndArrow);

draw(brace(bZim +(N*boxw, 0), bZre,-ampl), arrowPen);

pair bX = bx + (0, -4);
draw_array(bX, 2*boxw, boxh, Ncomplex);

pair bY = bX + (yoffset, 0);
draw_array(bY, 2*boxw, boxh, Ncomplex);



label("$X_0=\Re{Z_0}$", bX + Cmiddle);
draw(bZre + Rmiddle + Bottom -- bX + Cmiddle + Top, arrowPen, EndArrow);
label("$Y_0=\Im{Z_0}$", bY + Cmiddle);
draw(bZim + Rmiddle + Bottom -- bY + Cmiddle + Top, arrowPen, EndArrow);


for(int i = 1; i < Ncomplex; ++i) {
    pair ioff = i * (boxw,0);
    pair Nioff = (N - i) * (boxw,0);
    
    draw(bZre + ioff + Rmiddle + Bottom -- bX + 2* ioff + Cmiddle+Top,
	 arrowPen, EndArrow);

    draw(bZre + Nioff + Rmiddle + Bottom -- bX + 2* ioff + Cmiddle+Top,
	 arrowPen, EndArrow);

    draw(bZim + ioff + Rmiddle + Bottom -- bX + 2* ioff + Cmiddle+Top,
	 arrowPen, EndArrow);

    draw(bZim + Nioff + Rmiddle + Bottom -- bX + 2* ioff + Cmiddle+Top,
	 arrowPen, EndArrow);

    
    draw(bZim + ioff + Rmiddle +Bottom-- bY + 2*ioff + Cmiddle + Top,
    	 arrowPen, EndArrow);

    draw(bZim + Nioff + Rmiddle +Bottom -- bY + 2*ioff + Cmiddle+Top,
    	 arrowPen, EndArrow);
    
    draw(bZre + ioff + Rmiddle +Bottom-- bY + 2*ioff + Cmiddle + Top,
    	 arrowPen, EndArrow);

    draw(bZre + Nioff + Rmiddle +Bottom -- bY + 2*ioff + Cmiddle+Top,
    	 arrowPen, EndArrow);
    
    string si = (string)(i);
    string sNi = (string)(N-i);
    string xrhs = "\frac{Z_{"+si+"} + Z_{"+sNi+"}^*}{2}$";
    string yrhs = "\frac{Z_{"+si+"} - Z_{"+sNi+"}^*}{2i}$";
    label("$X_{"+si+"}="+ xrhs,
	  bX + 2 * ioff + Cmiddle);
    label("$Y_{"+si+"}=" + yrhs,
	  bY + 2 * ioff + Cmiddle);
}

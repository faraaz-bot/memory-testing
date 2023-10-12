size(800, 800);

int P = 4;
int N = 16;

real dw = 0.05;

real dskip = 0.4;

real w = 1.0  + (P-1) * dw;

real ypos = 0.0;
real xpos = 0.0;


for(int i = 0; i < P; ++i) {
  real y0 = i * dw + i * w / P;
  real y1 = y0 + w / P;

  label("GPU"+(string)i, (0, y0 + 0.5 * w / P), W);
  
  draw((0, y0) -- (1, y0) -- (1, y1) -- (0, y1) -- cycle);
}


xpos += w + dskip;
//ypos -= w + dskip;

for(int i = 0; i < P; ++i) {
  real y0 = i * dw + (i + 0.5 ) * w / P;
  pair p0 = (1, y0);
  pair p1 = (xpos, y0);
  draw("$\mathcal{F}_x$", p0--p1,EndArrow);
}

for(int i = 0; i < P; ++i) {
  real y0 = i * dw + i * w / P;
  real y1 = y0 + w / P;

  //label("GPU"+(string)i, (xpos, y0 + 0.5 * w / P), W);

  draw((xpos, y0) -- (xpos + 1, y0) -- (xpos + 1, y1) -- (xpos, y1) -- cycle);

  for(int j = 1; j < P; ++j) {
    real x0 = j / P + xpos;
    draw((x0, y0) -- (x0, y1), dashed);
  }
}

xpos += w + dskip;
ypos -= w + dskip;
// ypos -= w + dskip;
// xpos = 0.0;

for(int i = 0; i < P; ++i) {
  for(int j = 0; j < P; ++j) {
    // if(P - i -1 == j)
    //   continue;
    
    real x0 = xpos - w - dskip + (i + 0.5) / P; // add i dep
    real y0 = j * dw + (j + 0.5 ) * w / P; // add i dep
    pair p0 = (x0, y0);
    //dot(p0);

    real x1 = xpos + (j + 0.5) / P;
    real y1 = ypos +  i * dw + (i + 0.5 ) * w / P;
    pair p1 = (x1, y1);
    //dot(p1);

    draw(p0 -- p1, EndArrow);
  }
}



for(int i = 0; i < P; ++i) {

  //label("GPU"+(string)i, (xpos, y0 + 0.5 * w / P), W);

  real y0 = ypos;
  real y1 = y0  + 1;

  real x0 = xpos + i * dw + i * w / P;
  real x1 = x0 + w / P;
  draw((x0, y0) -- (x0, y1) -- (x1, y1) -- (x1, y0) -- cycle);
  //draw((xpos, y0) -- (xpos + 1, y0) -- (xpos + 1, y1) -- (xpos, y1) -- cycle);

  for(int j = 1; j < P; ++j) {
    real x0 = j / P + xpos; // FIXME: dashed in wrong direction
    draw((x0, y0) -- (x0, y1), dashed);
  }
}

// for(int i = 0; i < P; ++i) {
//   real x0 = i * dw + i * w / P + xpos;
//   real x1 = x0 + w / P;

//   //label("GPU"+(string)i, (x0 + 0.5 * w / P, ypos + 1), N);
  
//   draw((x0, ypos) -- (x1, ypos) -- (x1, ypos + 1) -- (x0, ypos + 1) -- cycle);

//   for(int i = 1; i < P; ++i) {
//     real y0 = i / P + ypos;
//     draw((x0, y0 + ypos) -- (x1, y0 + ypos), dashed);
//   }
// }

for(int i = 0; i < P; ++i) {
  real y0 = ypos;
  real y1 = y0 - w - dskip + 1;
  real x0 = xpos + i * dw + (i + 0.5 ) * w / P;
  pair p0 = (x0, y0);
  //dot(p0);
  pair p1 = (x0, y1);
  dot(p1);

  draw("$\mathcal{F}_y$", p0--p1,EndArrow);
}

// xpos += w + dskip;
ypos -= w + dskip;

// for(int i = 0; i < P; ++i) {
//   real x0 = i * dw + i * w / P + xpos;
//   real x1 = x0 + w / P;

//   //label("GPU"+(string)i, (x0 + 0.5 * w / P, ypos + 1), N);
  
//   draw((x0, ypos) -- (x1, ypos) -- (x1, ypos + 1) -- (x0, ypos + 1) -- cycle);

// }

for(int i = 0; i < P; ++i) {
  // real y0 = ypos + i * dw + i * w / P;
  // real y1 = y0 + w / P;
  real y0 = ypos;
  real y1 = y0  + 1;

  real x0 = xpos + i * dw + i * w / P;
  real x1 = x0 + w / P;

  
  //label("GPU"+(string)i, (xpos, y0 + 0.5 * w / P), W);

  draw((x0, y0) -- (x0, y1) -- (x1, y1) -- (x1, y0) -- cycle);
}

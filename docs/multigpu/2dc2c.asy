size(400, 400);

int P = 4;

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
  draw(p0--p1,EndArrow);
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

ypos -= w + dskip;
xpos = 0.0;

for(int i = 0; i < P; ++i) {
  real x0 = i * dw + i * w / P;
  real x1 = x0 + w / P;

  label("GPU"+(string)i, (xpos + x0 + 0.5 * w / P, ypos + 1), N);
  
  draw((x0, ypos) -- (x1, ypos) -- (x1, ypos + 1) -- (x0, ypos + 1) -- cycle);

  for(int i = 1; i < P; ++i) {
    real y0 = i / P + xpos;
    draw((x0, y0 + ypos) -- (x1, y0 + ypos), dashed);
  }
}

xpos += w + dskip;
//ypos -= w + dskip;

for(int i = 0; i < P; ++i) {
  real x0 = i * dw + i * w / P + xpos;
  real x1 = x0 + w / P;

  label("GPU"+(string)i, (x0 + 0.5 * w / P, ypos + 1), N);
  
  draw((x0, ypos) -- (x1, ypos) -- (x1, ypos + 1) -- (x0, ypos + 1) -- cycle);

}

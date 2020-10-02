size(800,400);

usepackage("amsfonts");
usepackage("amsfonts");
usepackage("physics");

// N must be even.
int N = 16;

real boxw = 1.0;
real boxh = 0.5;
real boxdh = 1.2;

pen arrowPen = grey;
pen labelPen = red;

int log2N = round(log(N) / log(2.0));

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



// Butterflies
for(int s = 0; s <= log2N; ++s)
{
    real yh =-s * boxdh;
    pair pos = (0, yh);
    draw_array(pos, boxw, boxh, N);
    int a = 2^s;
    int b = N # a;
    for(int l = 0; l < b # 2; ++l)
    {
        for(int k = 0; k < a; ++k)
        {
            int p = l + k * b;
            int q = p + b # 2;
            pair pp = (boxw * p + 0.5 * boxw, yh - 0.5 * boxh);
            pair qq = (boxw * q + 0.5 * boxw, yh - 0.5 * boxh);

            draw(pp -- pp - (0, boxdh), arrowPen, EndArrow);
            draw(qq -- qq - (0, boxdh), arrowPen, EndArrow);
            draw(pp -- qq - (0, boxdh), arrowPen, EndArrow);
            draw(qq -- pp - (0, boxdh), arrowPen, EndArrow);
        }
    }
}

real yh = -(log2N + 1) * boxdh;
pair pos = (0, yh);
draw_array(pos, boxw, boxh, N);
for(int p = 0; p < N; ++p)
{
    int q = bitreverse(p, log2N);
    pair pp = (boxw * p + 0.5 * boxw, yh + boxdh -  0.5 * boxh);
    pair qq = (boxw * q + 0.5 * boxw, yh - 0.5 * boxh);
    draw(pp -- qq, arrowPen, EndArrow);
}

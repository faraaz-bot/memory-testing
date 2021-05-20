size(800,400);

usepackage("amsfonts");
usepackage("amsfonts");
usepackage("physics");

// N must be even.
int N = 16;

int L = 7;

real boxw = 1.0;
real boxh = 0.5;
real boxdh = 1.5;

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

real yh = 0;


// Butterflies
for(int q = 1; q <= log2N; ++q)
{
    pair pos = (0, yh);
    draw_array(pos, boxw, boxh, N);
    if(q == 1)
    {
        for(int i = L; i < N; ++i)
        {
            pair p = ((i + 0.5) * boxw, -0.5*boxh);
            //dot(p,blue);
            label("$0$", p, red);
        }
    }

    int L = 2^q;
    int r = N # L;
    int Lstar = L # 2;
    for(int k = 0; k < r; ++k)
    {
        for(int j = 0; j < Lstar; ++j)
        {
            int xa = k * L + j;
            int xb = k * L + Lstar + j;

            int ya = k * Lstar + j;
            int yb = (k + r) * Lstar + j;

            pair pxa = (xa * boxw + 0.5 * boxw , yh - boxh );
            pair pxb = (xb * boxw + 0.5 * boxw , yh - boxh );
            
            pair pya = (ya * boxw + 0.5 * boxw , yh  );
            pair pyb = (yb * boxw + 0.5 * boxw , yh  );
            
            // dot(pxa, blue);
            // dot(pxb, red);
            // dot(pxa, blue);
            // dot(pxb, red);

            draw(pxa -- pya - (0, boxdh), arrowPen, EndArrow);
            draw(pxb -- pyb - (0, boxdh), arrowPen, EndArrow);
            draw(pxa -- pyb - (0, boxdh), arrowPen, EndArrow);
            draw(pxb -- pya - (0, boxdh), arrowPen, EndArrow);
        }
    }
    yh -= boxdh;
}


pair pos = (0, yh);
draw_array(pos, boxw, boxh, N);

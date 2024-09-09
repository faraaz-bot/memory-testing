size(300,300);

int N = 16;

draw(box((0,0),(N,N)));

draw((N # 2, 0) -- (N # 2, N), grey);
draw((0, N # 2) -- (N, N # 2), grey);

pair f(pair p) {
    return (N, N) - p;
}

void dodraw(pair p) {
    dot(p, red);
    dot(f(p), blue);
    draw(p -- f(p), Arrows);
}

pair p = (10, 1);

dodraw(p);

dodraw((12, 5));
dodraw((2, 15));
dodraw((9, 9));
dodraw((12, 15));

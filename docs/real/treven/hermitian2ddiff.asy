size(300,300);

int N = 16;

pair shift = (0,0);

pair p0 = (0,0);
pair p1 = (N,N);

for(int i = 0; i < 2; ++ i) {

    if(i == 1) {
        shift = (1.5 N, 0);
    }
    draw(box(shift + p0, shift + p1));

    draw(shift + (N # 2, 0) -- shift + (N # 2, N), grey);
    draw(shift + (0, N # 2) -- shift + (N, N # 2), grey);

    if(i == 0) {
        fill(shift + (0,0)--(0,N)--(N#2, N)--(N#2, 0)--cycle, lightgrey);
    } else {
        fill(shift + (0,0)--shift+(0,N#2)--shift+(N, N#2)--shift+(N, 0)--cycle, lightgrey);
    }
        
    
    pair f(pair p, pair shift) {
        return shift + (N, N) - p;
    }

    void dodraw(pair p, pair shift) {
        dot(shift + p, red);
        dot(f(p, shift), blue);
        draw(p + shift -- f(p, shift), Arrows);
    }

    pair p = (10, 1);

    dodraw(p, shift);

    dodraw((12, 5), shift);
    dodraw((2, 15), shift);
    dodraw((9, 9), shift);
    dodraw((12, 15), shift);

    
}

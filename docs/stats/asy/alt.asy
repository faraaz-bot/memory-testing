size(350,0);


real width = 1.0;
real height = 0.3;


int Nbox = 8;

for(int ibox = 0; ibox < Nbox; ++ibox) {
    pair a = (2*ibox * width, 0);
    pair b = ((2*ibox + 1) * width, -height);
    filldraw(box(a,b),red);
}

real offset = width;
for(int ibox = 0; ibox < Nbox; ++ibox) {
    pair a = (offset + 2*ibox * width, 0);
    pair b = (offset + (2*ibox + 1) * width, -height);
    filldraw(box(a,b),blue);
}

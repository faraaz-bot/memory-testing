size(350,0);


real width = 1.0;
real height = 0.3;

real sep = 2.0;

int Nbox = 8;

for(int ibox = 0; ibox < Nbox; ++ibox) {
    pair a = (ibox * width, 0);
    pair b = ((ibox + 1) * width, -height);
    filldraw(box(a,b),red);
}

real offset = Nbox * width + sep;
for(int ibox = 0; ibox < Nbox; ++ibox) {
    pair a = (offset + ibox * width, 0);
    pair b = (offset + (ibox + 1) * width, -height);
    filldraw(box(a,b),blue);
}

size(350,0);


real width = 1.0;
real height = 0.3;


int Nbox = 8;

int Nred = 0;
int Nblue = 0;
int ibox = 0;
while(Nred < Nbox && Nred < Nbox) {
    pair a = (ibox * width, 0);
    pair b = ((ibox + 1) * width, -height);
    bool usered = rand() % 2 == 0;
    if(usered)
        Nred += 1;
    else
        Nblue += 1;
    filldraw(box(a,b),usered ? red : blue);
    ibox += 1;
}

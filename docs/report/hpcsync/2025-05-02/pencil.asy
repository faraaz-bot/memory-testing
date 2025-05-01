import three;

int N = 16;
int P = 2;
real w = 0.1;

size(200,0);

pen surfpen=lightgreen+opacity(0.5);
//draw(surface(g2), lightblue);

triple[] pblock = new triple[];
pblock.push((0, 0, 0));
pblock.push((0, (N # P) * w, 0));
pblock.push((0, (N # P) * w, N * w));
pblock.push((0, 0, N * w));
for(int idx = 0; idx < 4; ++ idx) {
    pblock.push(pblock[idx] + (w * (N # P), 0, 0));
}

for(int idx = 0; idx < pblock.length; ++idx) {
    //dot((string)idx, pblock[idx]);
}

path3[] sblock = new path3[];
sblock.append(pblock[0]--pblock[1]--pblock[2]--pblock[3]--cycle);
sblock.append(pblock[4]--pblock[5]--pblock[6]--pblock[7]--cycle);
sblock.append(pblock[0]--pblock[1]--pblock[5]--pblock[4]--cycle);
sblock.append(pblock[0]--pblock[1]--pblock[5]--pblock[4]--cycle);
sblock.append(pblock[0]--pblock[4]--pblock[7]--pblock[3]--cycle);
sblock.append(pblock[2]--pblock[3]--pblock[7]--pblock[6]--cycle);
sblock.append(pblock[1]--pblock[2]--pblock[6]--pblock[5]--cycle);

for(int pxidx = 0; pxidx < P; ++pxidx) {
    for(int pyidx = 0; pyidx < P; ++pyidx) {
        int pidx = pyidx + P * pxidx;
        for(int idx = 0; idx < sblock.length; ++idx) {
            triple offset  = (pxidx * (1 + N # P) * w,
                              pyidx * (1 + N # P) * w ,
                              0);


            draw(surface(shift(offset) * sblock[idx]), Pen(pidx));
        }
    }
}


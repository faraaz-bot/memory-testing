size(35cm,0);

texpreamble("\usepackage{amsmath}");
texpreamble("\usepackage{amssymb}");


path plabel(pair a, pair b) { return a + 0.5 * b;}

pair a = (0,0);
real dx = 0.7;
real dy = 0.3;
real dboxx = 0.5;
real dboxy = 0.1;
pair dbox = (dboxx, dboxy);
pair rbox = (dboxx, 0.5*dboxy);
pair lbox = (dx, 0.5*dboxy);

draw(box(a, a + dbox), red);
label("$x_n$", plabel(a, dbox));
draw((a + rbox)..(a + lbox), EndArrow);

a += (dx, 0);

draw(box(a, a + dbox), blue);
label("$x_n \omega_{2N}^{n^2}$ ", plabel(a, dbox));
draw((a + rbox)..(a + lbox), EndArrow);

pair b = a - (0, dy);
draw(box(b, b + dbox), blue);
label("$\omega_{2N}^{-k^2}$", plabel(b, dbox));
draw((b + rbox)..(b + lbox), EndArrow);

a += (dx, 0);

draw(box(a, a + dbox), blue);
label("$\mathcal{F}\left(x_n \omega_{2N}^{n^2}\right)$ ", plabel(a, dbox));
draw((a + rbox)..(a + lbox), EndArrow);

b += (dx, 0);
draw(box(b, b + dbox), blue);
label("$\mathcal{F}\left(\omega_{2N}^{-k^2}\right)$", plabel(b, dbox));
draw((b + rbox)..(a + lbox), EndArrow);

a += (dx, 0);

draw(box(a, a + dbox), blue);
label("$\mathcal{F}\left(x_n \omega_{2N}^{n^2}\right)\mathcal{F}\left(\omega_{2N}^{-k^2}\right)$ ", plabel(a, dbox));
draw((a + rbox)..(a + lbox), EndArrow);

a += (dx, 0);

draw(box(a, a + dbox), blue);
label("$x_n \omega_{2N}^{n^2} * \omega_{2N}^{-k^2}$ ", plabel(a, dbox));
draw((a + rbox)..(a + lbox), EndArrow);

a += (dx, 0);
draw(box(a, a + dbox), red);
label("$\mathcal{F}\left(x_n\right)$", plabel(a, dbox));

size(10cm,0);

texpreamble("\usepackage{amsmath}");
texpreamble("\usepackage{amssymb}");


path plabel(pair a, pair b) { return a + 0.5 * b;}

pair a = (0,0);
real dx = 0.6;
real dy = 0.2;
real dboxx = 0.5;
real dboxy = 0.1;
pair dbox = (dboxx, dboxy);
real short_fact = 0.4;
pair dbox_short = (short_fact*dboxx, dboxy);

pair rbox = (dboxx, 0.5*dboxy);
pair lbox = (dx, 0.5*dboxy);
pair tbox = (0.5*dboxx,-dy + dboxy);
pair bbox = (0.5*dboxx, 0);

pair tbox_short = (0.5*dboxx * short_fact,-dy + dboxy);
pair bbox_short = (0.5*dboxx * short_fact, 0);

draw(box(a, a + dbox_short), red);
label("$x_n$", plabel(a, dbox_short));
draw((a + bbox_short)..(a + tbox), EndArrow);

a += (0, -dy);

draw(box(a, a + dbox), blue);
label("$x_n \omega_{2N}^{n^2}$ ", plabel(a, dbox));
draw((a + bbox)..(a + tbox), EndArrow);


pair b = a + (dx, 0);
draw(box(b, b + dbox), blue);
label("$\omega_{2N}^{-k^2}$", plabel(b, dbox));
draw((b + bbox)..(b + tbox), EndArrow);

a += (0, -dy);

draw(box(a, a + dbox), blue);
label("$\mathcal{F}\left(x_n \omega_{2N}^{n^2}\right)$ ", plabel(a, dbox));
draw((a + bbox)..(a + tbox), EndArrow);

b += (0, -dy);
draw(box(b, b + dbox), blue);
label("$\mathcal{F}\left(\omega_{2N}^{-k^2}\right)$", plabel(b, dbox));
draw((b + bbox)..(a + tbox), EndArrow);

a += (0, -dy);

draw(box(a, a + dbox), blue);
label("$\mathcal{F}\left(x_n \omega_{2N}^{n^2}\right)\mathcal{F}\left(\omega_{2N}^{-k^2}\right)$ ", plabel(a, dbox));
draw((a + bbox)..(a + tbox), EndArrow);


a += (0, -dy);

draw(box(a, a + dbox), blue);
label("$x_n \omega_{2N}^{n^2} * \omega_{2N}^{-k^2}$ ", plabel(a, dbox));
draw((a + bbox)..(a + tbox_short), EndArrow);

a += (0, -dy);
draw(box(a, a + dbox_short), red);
label("$\mathcal{F}\left(x_n\right)$", plabel(a, dbox_short));

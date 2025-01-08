size(0, 15cm);


texpreamble("\usepackage{amsmath}");
texpreamble("\usepackage{amssymb}");

pair a = (0,0);

path plabel(pair a, pair b) { return a + 0.5 * b;}



real dboxx = 0.5;
real dboxy = 0.5;
pair dbox = (dboxx, dboxy);
pair hbox = (0.5*dboxx, dboxy);

real dy = 0.2 + dboxy;
pair bbox = (0.5*dboxx, 0);
pair tbox = (0.5*dboxx, -dy + dboxy);

pair bhbox = (0.25*dboxx, 0);
pair thbox = (0.25*dboxx, -dy + dboxy);

draw(box(a, a + dbox), red);
label("$\mathbb{R}$ ", plabel(a, dbox));
draw((a + bbox)..(a + tbox), EndArrow, L="Complex embedding");

a += (0, -dy);

draw(box(a, a + dbox), red);
label("$\mathbb{C}$ ", plabel(a, dbox));
draw((a + bbox)..(a + tbox), EndArrow, L = "$\mathcal{F}_{x,\mathbb{C}\rightarrow\mathbb{C}}$");

a += (0, -dy);

draw(box(a, a + dbox), red);
label("$\mathbb{C}$ ", plabel(a, dbox));
draw((a + bbox)..(a + tbox), EndArrow, L =  "$\mathcal{F}_{y,\mathbb{C}\rightarrow\mathbb{C}}$");

a += (0, -dy); 

label("$\mathbb{C}$ ", plabel(a, dbox));
draw(box(a, a + dbox), blue);
draw((a + bbox)..(a + thbox), EndArrow, L = "Remove redundant data");

a += (0, -dy); 

label("$\mathbb{C}$ ", plabel(a, hbox));
draw(box(a, a + hbox), blue);

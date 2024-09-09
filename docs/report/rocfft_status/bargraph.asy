import graph;
import utils;

import palette;
import colormap;

struct bardata {
  string label;
  real y;
  real ylow;
  real yhigh;
}


// Input data:
string filenames = "";
string secondary_filenames = "";
string legendlist = "";

// Graph formatting
string xlabel = "Problem size type";
string ylabel = "Time [s]";
bool dolabel=true;
bool dospeedups = false;


string primaryaxis = "time";
string secondaryaxis = "speedup";
bool inverse = true;

usersetting();

pen penpicker(int n) {
  return tab10.palette()[n];
  if(n == 0)
    return RGB(179, 0, 0);
  if(n ==1)
    return RGB(124, 17, 88);
  if(n==2)
    return RGB(68, 33, 175);
  if(n==3)
    return RGB(26, 83, 255);
  if(n==4)
    return RGB(13, 136, 230);
  if(n==5)
    return RGB(0, 183, 199);
  if(n==6)
    return RGB(90, 212, 90);
  if(n==7)
    return RGB(139, 224, 78);
  if(n==8)
    return RGB(235, 220, 120);
  return Pen(n);
}


// TODO: make inverses an option.
// TODO: line-up the data with the release number so that we don't have to
// have the exact same version range.

void readbarfiles(string[] filelist, bardata[][] data)
{
    for(int n = 0; n < filelist.length; ++n)
    {
      data.push(new bardata[]);
      
      string filename = filelist[n];
//      write(filename);
      
      file fin = input(filename).line().word();
      //string[] hdr = fin;

      while(!eof(fin)) {
     
	bardata dat;
	dat.label = fin;
	dat.y = (real)fin;
	dat.ylow = (real)fin;
	dat.yhigh = (real)fin;

	if(inverse) {
	  dat.y = 1.0 / dat.y;
	  real temp = dat.ylow;
	  dat.ylow = 1.0 / dat.yhigh;
	  dat.yhigh = 1.0 / dat.ylow;
	}
	
	write(dat.label, dat.y, dat.ylow, dat.yhigh);

	data[n].push(dat);

      }
    }
//    write("done reading");
}

void drawbargraph(bardata[][] data, string[] legs, string[] otherlegs) {
  // Assumption: same number of data points.

  int ncase = data.length;

  string[] barkeys;
  // FIXME: try and figure out which is the first label.
  // What happens if the labels are all unique?  I guess we default
  // to file ordering to break these kinds of ties.
  
  // Start with the first file.  If the first key is first or not present
  // in the other files, then this is the first barekey.  If it's second in
  // the other files, then re-start the search looking for the key before
  // the first key.
  // When we are happy that we've found a key, pop it from the other files.
  // FIXME: what about (a,b) (b,a)?
  string[][] allkeys;
  for(int icase = 0; icase < ncase; ++icase) {
    allkeys.push(new string[]);
    for(int idx = 0; idx < data[icase].length; ++idx) {
      allkeys[icase].push(data[icase][idx].label);
    }
  }
  
  //write("allkeys:");
  //write(allkeys);
  if(ncase == 0)
    return;

  while(allkeys.length > 0) {
    int mycase = 0;
    string key = allkeys[mycase][0];
    //write(key);
    bool pushkey = true;
    do {
      for(int icase = 0; icase < allkeys.length; ++icase) {
	if(key != allkeys[icase][0]) {
	  // See if the key shows up later:
	  for(int idx = 1; idx < allkeys[icase].length; ++idx) {
	    if(key == allkeys[icase][idx]) {
	      key = allkeys[icase][idx];
	      pushkey = false;
	      break;
	    }
	  }
	}
      }
      // TODO: if the current key shows up later in another case,
      // switch to the first key in the other case.
    } while(!pushkey);

    barkeys.push(key);
    for(int icase = 0; icase < allkeys.length; ++icase) {
      allkeys[icase].delete(0);
    }

    int nncase = allkeys.length;
    for(int icase = nncase-1; icase >= 0; --icase ) {
      if(allkeys[icase].length == 0) {
	allkeys.delete(icase);
      }
    }
  }
  //write("barkeys");
  //write(barkeys);

  
  real width = 1.0 / ncase;
  real skip = 0.5;
  
  // Loop through all the data sets.
  for(int icase = 0; icase < ncase; ++icase) {
    pen p = penpicker(icase); // + opacity(0.5);
    if(icase == 2)
      p = deepgreen;

    int len = data[icase].length;

    int nbar = barkeys.length;
    
    // Set up the left and right sides of the bars.
    real[] left = new real[nbar];
    real[] right = new real[nbar];
    left[0] = icase * width;
    for(int i = 1; i < nbar; ++i) {
      left[i] = left[i - 1] + ncase * width + skip;
    }
    for(int i = 0; i < nbar; ++i) {
      right[i] = left[i] + width;
    }
    
    // Draw an invisible graph to set up the axes.
    real[] fakex = new real[nbar];
    fakex[0] = left[0];
    for(int i = 1; i < fakex.length; ++i) {
      fakex[i] = right[i];
    }
    real[] yvals = new real[len];
    real maxy = -infinity;
    for(int i = 0; i < len; ++i) {
      if(maxy < data[icase][i].y) {
	maxy = data[icase][i].y;
      }
    }

    for(int ibar = 0; ibar < nbar; ++ibar) {
      bool found = false;
      real yval = -infinity;
      for(int i = 0; i < len; ++i) {
	if(barkeys[ibar] == data[icase][i].label) {
	  found = true;
	  yval = data[icase][i].y;
	  break;
	}
      }
      yvals[ibar] = found ? yval : maxy;
	
    }
    
    {
      // FIXME: work this out.  Max of the cases?
      draw(graph(left, yvals), invisible, legend = Label(otherlegs[icase], p));
      //draw(graph(left, yvals), invisible); 
    }
    
    // TODO: in log plots, compute a better bottom.
    real bottom = 0.0;
    
    // Draw the bars
    for(int ibar = 0; ibar < barkeys.length; ++ibar) {
      string key = barkeys[ibar];
      //write(icase);
      //write(key);
      for(int idx = 0; idx < data[icase].length; ++idx) {
	if(key == data[icase][idx].label) {
	  pair p0 = Scale((left[ibar], data[icase][idx].y));
	  pair p1 = Scale((right[ibar], data[icase][idx].y));
	  pair p2 = Scale((right[ibar], bottom));
	  pair p3 = Scale((left[ibar], bottom));
	  filldraw(p0--p1--p2--p3--cycle, p, black);
	}
      }
    }
   
    if(false)
      {
	// Draw the bounds:
	for(int i = 0; i < data[icase].length; ++i) {
	  real xval = 0.5 * (left[i] + right[i]);
	  pair plow = (xval, data[icase][i].ylow);
	  dot(plow);
	  pair phigh = (xval, data[icase][i].yhigh);
	  dot(phigh);
	  draw(plow--phigh);
	  draw(plow-(0.25*width)--plow+(0.25*width));
	  draw(phigh-(0.25*width)--phigh+(0.25*width));
	}
      }
        
    // This is there the legends go
    if(icase == ncase - 1) {
      for(int i = 0; i <  nbar; ++i) {
	pair p = (0.5 * ncase * width + i * (skip + ncase * width), 0);
	// 	//label(rotate(90) * Label(xleg[i]), align=S, p);
	label(Label(barkeys[i]), align=S, p);
      }
    }
    
  }
}

texpreamble("\usepackage{bm}");

size(400, 300, IgnoreAspect);

if(primaryaxis == "gflops") {
    ylabel = "GFLOP/s";
}

//write("filenames:\"", filenames+"\"");
if(filenames == "") {
    filenames = getstring("filenames");
}
    
if (legendlist == "") {
    legendlist = filenames;
}

// TODO: the first column will eventually be text.
string[] testlist = listfromcsv(filenames);

// Data containers:
pair[][] xyval = new real[testlist.length][];
pair[][] ylowhigh = new real[testlist.length][];


bardata[][] data;

readbarfiles(testlist, data);

// for(int n = 0; n < data.length; ++n) {
//   for(int i = 0; i < data[n].length; ++i) {
//     write(data[n][i].label, data[n][i].y, data[n][i].ylow, data[n][i].yhigh);
//   }
// }

//write(ylowhigh);

//write(xyval);

// Generate bar legends.
string[] legs = {};
for(int i = 0; i < xyval[0].length; ++i) {
  legs.push(string(xyval[0][i].x));
}

bool bargraph = true;

real[] speedups;
for(int didx = 0; didx < data.length; ++didx) {
  int dlength = data[didx].length;
  real speedup = data[didx][0].y / data[didx][dlength-1].y;
  if(inverse)
    speedup = 1.0 / speedup;
  speedups.push(speedup);
}
write(speedups);

bool myleg = ((legendlist == "") ? false : true);
string[] legends = set_legends(legendlist);
if(dospeedups) {
  for (int i = 0; i < legends.length; ++i) {
    legends[i] = texify(legends[i] + " speedup: " + string(speedups[i],4));
  }
}

if(bargraph) {
  drawbargraph(data, legs, legends);
  xaxis(BottomTop);
} else {
  // line graph:
  scale(Linear,Log);

  string[] label;
  
  pair[][] yvals;
  for(int didx = 0; didx < data.length; ++didx) {
      pen graphpen = penpicker(didx);
      guide g = scale(0.5mm) * unitcircle;
      marker mark = marker(g, Draw(graphpen + solid));

    yvals.push(new pair[]);
    for(int idx = 0; idx < data[didx].length; ++idx) {
      if(didx == 0) {
	label.push(data[didx][idx].label);
      }
      yvals[didx].push((idx,data[didx][idx].y));
    }
    //"asdf", //testlist[didx] + "adsf " +  (string)speedups[didx],
    draw(graph(yvals[didx]),
	 graphpen,
	 legends[didx],
	 mark);
  }
  //write(label);
  //write(yvals);
  xaxis(BottomTop, LeftTicks(new string(real x) {
	return label[round(x)];}));
}

if(inverse) {
  yaxis("Transforms per ms", LeftRight, RightTicks);
} else {
  yaxis(ylabel, LeftRight, RightTicks);
}

if(dolabel) {
  attach(legend(),point(plain.E),  20*plain.E);
}

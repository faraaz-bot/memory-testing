import graph;
import stats;

void whiskerplot(real[][] data,
		 string[] legend,
		 string xaxislabel,
		 string yaxislabel)
{
  real[] x = sequence(data.length);

  real barwidth = 0.5;

  real dmin = min(data);
  real dmax = max(data);
  
  real[] miny = new real[data.length];
  real[] maxy = new real[data.length];
  for(int i = 0; i < data.length; ++i) {
    miny[i] = dmin;
    maxy[i] = dmax;
  }
  real[] xgraph = new real[];
  xgraph.push(x[0] - barwidth);
  for(int i = 1; i < x.length; ++i) {
    xgraph.push(x[i] + barwidth);
  }
  draw(graph(xgraph, maxy), invisible);
  draw(graph(xgraph, miny), invisible);

  int nsample = data.length;
  
  ticks xticks = LeftTicks(rotate(45)*"$%f$",
			   new string(real x) {return legend[round(x % nsample)];}, 
			   sequence(nsample));
  xaxis(xaxislabel, BottomTop, xticks);
  yaxis(yaxislabel, LeftRight, RightTicks);

  for(int i = 0; i < data.length; ++i) {
    // scatter plot:
    if(true) {
      srand(1);
      pen scatterpen = blue + opacity(0.5);
      real scatterwidth = 0.25;
      int ilen = data[i].length;
      for(int j = 0; j < ilen; ++j) {
	real d = 4.0 * scatterwidth * j * (ilen -1 - j) / (ilen * ilen);
	pair p = (x[i] + d * 2.0 * (unitrand() - 0.5), data[i][j]);
	dot(Scale(p), scatterpen);
      }
    }
    
    // whisker plot:
    if(true) {
      pen whiskerpen = black;
      pair bard = ( 0.5 * barwidth,0);

      real[] sdata = new real[data[i].length];
      for(int j = 0; j < data[i].length; ++j) {
	sdata[j] = data[i][j];
      }
      sdata = sort(sdata);
      
      real median = median(sdata);
      real q1 = firstq(sdata);
      real q3 = lastq(sdata);
      write(q1, median, q3);
     
      pair p = (x[i], median);
      p = Scale(p);
      draw(p-bard--p+bard, whiskerpen);
      
      pair pl = (x[i], q1);
      pl = Scale(pl);
      pair ph = (x[i], q3);
      ph = Scale(ph);
      draw((pl-bard)--(pl+bard)--(ph+bard)--(ph-bard)--cycle,whiskerpen);

      pair pmin =  (x[i], min(sdata));
      pmin = Scale(pmin);
      pair pmax = (x[i], max(sdata));
      pmax = Scale(pmax);
      draw(pl--pmin, whiskerpen);
      draw(ph--pmax, whiskerpen);
     
      // if(drawmean) {
      //   pair p = (x[i], means[i]);
      //   draw((p-bard)..(p+bard),red);
      // }
    }
     
  }
}

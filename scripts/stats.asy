real median(real[] vals)
{
  // We assume that vals is sorted
  int n = vals.length;
  
  if (n == 0)
    return 0.0;
  if (n == 1)
    return vals[0];
  
  vals = sort(vals);
  int halfn = n # 2;
  if (n % 2 == 0) {
    return 0.5 * (vals[halfn] + vals[halfn - 1]);
  }
  return vals[halfn];
}

real firstq(real[] vals)
{
  // We assume that vals is sorted
  int n = vals.length;
  if(n == 0)
    return 0.0;
  if(n == 1)
    return vals[0];
  if(n == 2)
    return vals[0];

  int qn = (n) # 4;
  if(n % 4 == 0) {
      return vals[qn - 1];
  }
  if(n % 2 == 0) {
    return 0.5 * (vals[qn - 1] + vals[qn]);
  }
  return vals[qn];
}

real lastq(real[] vals)
{
  // We assume that vals is sorted
  int n = vals.length;
  if(n == 0)
    return 0.0;
  if(n == 1)
    return vals[0];
  if(n == 2)
    return vals[1];

  int qn = n # 4 * 3;
  
  if(n%4 == 0) {
    return vals[qn];
  }
  if(n%2 == 0) {
    return 0.5 * (vals[qn] + vals[qn + 1]);
  }
  return vals[qn];
}

pair medianbounds(real[] vals)
{
  int n = vals.length;

  real vmin;
  real vmax;
  
  if (n == 0) {
    vmin = 0.0;
    vmax = 0.0;
  }
  if (n == 1) {
    vmin = vals[0];
    vmax = vals[0];
  }
  if (n == 2) {
    vmin = min(vals[0], vals[1]);
    vmax = max(vals[0], vals[1]);
  }
  if (n > 2) {
    // Desired confidence interval:
    real alpha = 0.9;
      
    // Number of bootstrap resamples:
    int nboot = 5000;

    real[] medians = new real[nboot];

    medians[0] = median(vals);
    
    srand(seconds());
    real[] tvals = new real[n];
    
    for(int boot = 1; boot < nboot; ++boot) {
      for(int i = 0; i < n; ++i) {
        tvals[i] = vals[rand() %  n];
      }
      medians[boot] = median(tvals);
    }

    real offset = 0.5 * (1.0 - alpha);

    medians = sort(medians);
    vmin = medians[(int)floor(offset * nboot)];
    vmax = medians[(int)ceil((1.0 - offset ) * nboot)];
  }

  return (vmin, vmax);
}

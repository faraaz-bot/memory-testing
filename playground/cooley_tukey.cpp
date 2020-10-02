//
// Cooley-Tukey playground.
//
// Simple 1d, complex to complex, power of 2.
//

#include<cmath>
#include<iostream>
#include<memory>
#include<random>
#include<utility>
#include<vector>

#include<hip/hip_runtime.h>
#include<fftw3.h>

#define PI 3.141592653589793238462643383279502884L

using namespace std;

//
// Random inputs
//
vector<fftw_complex> random_vector(size_t n)
{
  vector<fftw_complex> x(n);
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<double> dis(0.0, 1.0);
  for (auto xi : x) {
    *xi = dis(gen);
  }
  return x;
}

//
// Naive DFT
//
vector<fftw_complex> fft_naive(vector<fftw_complex>& x)
{
  auto const N = x.size();

  vector<fftw_complex> z(N);
  for (size_t n = 0; n < N; ++n) {
    z[n][0] = 0.0;
    z[n][1] = 0.0;
  }

  for (size_t k = 0; k < N; ++k) {
    for (size_t n = 0; n < N; ++n) {
      double theta = -2 * PI * n * k / N;
      z[k][0] += cos(theta) * x[n][0] - sin(theta) * x[n][1];
      z[k][1] += cos(theta) * x[n][1] + sin(theta) * x[n][0];
    }
  }
  return z;
}

//
// FFTW backed FFT
//
vector<fftw_complex> fft_fftw(vector<fftw_complex>& x)
{
  vector<fftw_complex> z(x.size());
  auto p = fftw_plan_dft_1d(x.size(), x.data(), z.data(), FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  return z;
}

//
// Cooley-Tukey FFT
//

size_t bitreverse(size_t x, size_t n) {
  size_t r = 0;
  for (size_t i=0; i < n; ++i) {
    r <<= 1;
    r |= x & 1;
    x >>= 1;
  }
  return r;
}

vector<fftw_complex> fft_ict(vector<fftw_complex>& x)
{
  auto const N = x.size();
  auto const log2N = (size_t) log2(N);

  vector<fftw_complex> z(N);
  for (size_t n = 0; n < N; ++n) {
    z[n][0] = x[n][0];
    z[n][1] = x[n][1];
  }

  for (size_t s = 0; s < log2N; ++s) {
    auto a = (size_t) pow(2, s);
    auto b = N / a;
    auto halfb = b / 2;
    for (size_t l = 0; l < halfb; ++l) {
      for (size_t k = 0; k < a; ++k) {
        auto p = l + k * b;
        auto q = p + halfb;
        double theta = -2 * PI * l / b;
        fftw_complex xp = { z[p][0], z[p][1] };
        fftw_complex xq = { z[q][0], z[q][1] };
        z[p][0] = xp[0] + xq[0];
        z[p][1] = xp[1] + xq[1];
        z[q][0] = cos(theta) * (xp[0] - xq[0]) - sin(theta) * (xp[1] - xq[1]);
        z[q][1] = cos(theta) * (xp[1] - xq[1]) + sin(theta) * (xp[0] - xq[0]);
      }
    }
  }

  for (size_t p=0; p < N; ++p) {
    auto q = bitreverse(p, log2N);
    if (p > q) swap(z[p], z[q]);
  }

  return z;
}

//
// Quick compare
//
void compare(vector<fftw_complex> const & z1, vector<fftw_complex> const & z2)
{
  double d = 0.0;
  double r = 0.0;
  for (size_t n = 0; n < z1.size(); ++n) {
    double dx = z1[n][0] - z2[n][0];
    double dy = z1[n][1] - z2[n][1];
    //    cout << n << " " << z1[n][0] << " " << z2[n][0] << endl;
    d += sqrt(dx*dx + dy*dy);
    r += sqrt(z1[n][0] * z1[n][0] + z1[n][1] * z1[n][1]);
  }
  cout << "rel diff: " << d / r << endl;
}

void test_ct()
{
  size_t const n = 4096;
  auto x = random_vector(n);
  auto z1 = fft_fftw(x);
  auto z2 = fft_naive(x);
  auto z3 = fft_ict(x);
  //  auto z4 = fft_gpu(x);
  compare(z1, z2);
  compare(z1, z3);
  //  compare(z1, z4);
}

int main(int argc, char* argv[])
{
  test_ct();
}

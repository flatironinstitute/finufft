// this is all you must include for the finufft lib...
#include <complex>
#include <finufft.h>

// also needed for this example...
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

static const double PI = 3.141592653589793238462643383279502884;

int main() {

  /* Simple 2D type-1 example of calling the FINUFFT library from C++, using plain
     arrays of C++ complex numbers, with a math test. Double precision version.
     To compile, see README. Usage:  ./simple2d1
  */

  // generate random non-uniform points on (x,y) and complex strengths (c),
  // each strength component uniform random in [-1,1]:
  // docs-start: simple2d1-data
  int M = 1e7; // number of nonuniform points
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);
  generate(x.begin(), x.end(), [&] { return upi(rng); }); // uniform random in [-pi, pi)
  generate(y.begin(), y.end(), [&] { return upi(rng); }); // uniform random in [-pi, pi)
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  // docs-end: simple2d1-data

  // choose numbers of output Fourier coefficients in each dimension,
  // and allocate the output array for the Fourier modes
  // docs-start: simple2d1-alloc
  int N1 = 1000;
  int N2 = 2000;
  vector<complex<double>> F(N1 * N2);
  // docs-end: simple2d1-alloc

  // call the NUFFT (with iflag += 1): note passing in pointers...
  // docs-start: simple2d1-call
  double tol = 1e-6; // desired accuracy
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.upsampfac = 1.25;
  int ier = finufft2d1(M, x.data(), y.data(), c.data(), 1, tol, N1, N2, F.data(), &opts);
  // docs-end: simple2d1-call

  int k1  = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  int k2  = round(-0.35 * N2);

  complex<double> I(0.0, 1.0); // the imaginary unit
  complex<double> Ftest(0, 0);
  for (int j = 0; j < M; j++)
    Ftest += c[j] * exp(I * (double(k1) * x[j] + double(k2) * y[j]));

  // compute inf norm of F
  double Fmax = 0.0;
  for (int m = 0; m < N1 * N2; m++) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }

  // indices in output array for this frequency pair (k1,k2)
  int k1out    = k1 + N1 / 2;
  int k2out    = k2 + N2 / 2;
  int indexOut = k1out + k2out * N1;

  // compute relative error
  double err   = abs(F[indexOut] - Ftest) / Fmax;
  cout << "2D type-1 NUFFT done. ier=" << ier << ", err in F[" << indexOut
       << "] rel to max(F) is " << setprecision(2) << err << endl;
  if (!(err <= 10 * tol)) return 1;
  return ier;
}

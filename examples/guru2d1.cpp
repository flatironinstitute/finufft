// docs-start: guru2d1
#include <finufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

int main() {
  /* 2D type 1 guru interface example of calling the FINUFFT library from C++,
     using STL double complex vectors, with a math test. Similar to simple2d1
     except illustrates the guru interface.
     To compile, see README.  Usage: ./guru2d1
  */
  constexpr int M      = 1e6;  // number of nonuniform points
  constexpr int N      = 1e6;  // approximate total number of modes (N1*N2)
  constexpr double tol = 1e-6; // desired accuracy
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.upsampfac = 1.25;
  complex<double> I(0.0, 1.0); // the imaginary unit

  // generate random non-uniform points on (x,y) and complex strengths (c):
  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // choose numbers of output Fourier coefficients in each dimension
  const int N1 = round(2.0 * sqrt(N));
  const int N2 = round(N / N1);

  // output array for the Fourier modes
  vector<complex<double>> F(N1 * N2);

  constexpr int type = 1, dim = 2, ntrans = 1; // you could also do ntrans>1
  const int64_t Ns[] = {N1, N2};               // N1,N2 as 64-bit int array
  // step 1: make a plan...
  finufft_plan plan;
  int ier = finufft_makeplan(type, dim, Ns, +1, ntrans, tol, &plan, &opts);
  if (ier) return ier;
  // step 2: send in M nonuniform points (just x, y in this case)...
  ier = finufft_setpts(plan, M, &x[0], &y[0], NULL, 0, NULL, NULL, NULL);
  if (ier) return ier;
  // step 3: do the planned transform to the c strength data, output to F...
  ier = finufft_execute(plan, &c[0], &F[0]);
  // ... you could now send in new points, and/or do transforms with new c data
  // ...
  // step 4: free the memory used by the plan...
  finufft_destroy(plan);
  // docs-end: guru2d1
  if (ier) return ier;

  const int k1 = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2 = round(-0.35 * N2);

  vector<complex<double>> terms(M);
  transform(x.begin(), x.end(), y.begin(), terms.begin(), [&](auto xj, auto yj) {
    return exp(I * (double(k1) * xj + double(k2) * yj));
  });
  transform(terms.begin(), terms.end(), c.begin(), terms.begin(),
            [](auto tj, auto cj) { return tj * cj; });
  const auto Ftest = reduce(terms.begin(), terms.end());

  const auto Fmax = abs(
      *max_element(F.begin(), F.end(), [](auto a, auto b) { return abs(a) < abs(b); }));

  // indices in output array for this frequency pair (k1,k2)
  const int k1out    = k1 + (int)N1 / 2;
  const int k2out    = k2 + (int)N2 / 2;
  const int indexOut = k1out + k2out * (N1);

  // compute relative error
  const auto err     = abs(F[indexOut] - Ftest) / Fmax;
  cout << "2D type-1 NUFFT done. ier=" << ier << ", err in F[" << indexOut
       << "] rel to max(F) is " << setprecision(2) << err << endl;
  return !(err < 10 * tol);
}

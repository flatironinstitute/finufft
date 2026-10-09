#include <finufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

int main() {
  /* 2D demo of computing the *adjoint* of the planned transform, needing the
     guru interface.
     We plan a type 2, and then perform its adjoint (which is a type 1 with the
     opposite isign).
     We call the FINUFFT library from C++,
     using STL double complex vectors, with a math test.
     Computes an identical transform to guru2d1 except using the execute_adjoint
     feature. Barbone and Barnett, June 2025.
     To compile, see README.  Usage: ./guru2d1_adjoint
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
  // each component of c uniform random in [-1,1]
  generate(x.begin(), x.end(), [&] { return upi(rng); });
  generate(y.begin(), y.end(), [&] { return upi(rng); });
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // choose numbers of output Fourier coefficients in each dimension
  const int N1 = round(2.0 * sqrt(N));
  const int N2 = round(N / N1);

  // output array for the Fourier modes
  vector<complex<double>> F(N1 * N2);

  constexpr int type = 2, dim = 2, ntrans = 1; // you could also do ntrans>1
  const int64_t Ns[] = {N1, N2};               // N1,N2 as 64-bit int array

  // step 1: make a plan... note we choose isign=-1 for this type 2 plan
  finufft_plan plan;
  int ier = finufft_makeplan(type, dim, Ns, -1, ntrans, tol, &plan, &opts);
  if (ier) return ier;
  // step 2: send in M nonuniform points (just x, y in this case)...
  ier = finufft_setpts(plan, M, &x[0], &y[0], NULL, 0, NULL, NULL, NULL);
  if (ier) return ier;
  // step 3: do the adjoint of the planned transform. This maps
  // c strength data, to F output, and is identical to the type 1 with isign=+1.
  ier = finufft_execute_adjoint(plan, &c[0], &F[0]);
  // ... you could now send in new points, and/or do transforms or their adjoints.
  // ...
  // step 4: free the memory used by the plan...
  finufft_destroy(plan);
  if (ier) return ier;

  const int k1 = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2 = round(-0.35 * N2);

  complex<double> Ftest(0.0, 0.0);
  for (int j = 0; j < M; ++j)
    Ftest += c[j] * exp(I * (double(k1) * x[j] + double(k2) * y[j]));

  double Fmax = 0.0; // compute inf norm of F
  bool Ffin   = true;
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    Ffin &= std::isfinite(a);
    if (a > Fmax) Fmax = a;
  }

  // indices in output array for this frequency pair (k1,k2)
  const int k1out    = k1 + (int)N1 / 2;
  const int k2out    = k2 + (int)N2 / 2;
  const int indexOut = k1out + k2out * (N1);

  // compute relative error
  const auto err     = abs(F[indexOut] - Ftest) / Fmax;
  if (!(Ffin && Fmax > 0.0 && err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}

/* 2D type 1 guru interface example of calling the FINUFFT library from C++,
   using STL double complex vectors, with a math test. Similar to simple2d1
   except illustrates the guru interface.
   The code between the docs-start/docs-end markers is shown in ../docs/cex.rst.
*/
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <finufft.h>
#include <vector>
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
  int M = 1e7; // number of nonuniform points
  vector<double> x(M), y(M);
  vector<complex<double>> c(M);
  complex<double> I(0.0, 1.0); // the imaginary unit
  for (int j = 0; j < M; ++j) {
    x[j] = PI * (2 * (double)rand() / RAND_MAX - 1);
    y[j] = PI * (2 * (double)rand() / RAND_MAX - 1);
    c[j] = 2 * (double)rand() / RAND_MAX - 1 + I * (2 * (double)rand() / RAND_MAX - 1);
  }
  int N1 = 1000, N2 = 2000;
  vector<complex<double>> F(N1 * N2);

  // docs-start: guru2d1
  // (assume x, y, c are filled, and F allocated, as in the 2D code above...)
  int type = 1, dim = 2, ntrans = 1;
  int64_t Ns[] = {1000, 2000}; // N1,N2 as 64-bit int array
  // step 1: make a plan...
  finufft_plan plan;
  int ier = finufft_makeplan(type, dim, Ns, +1, ntrans, 1e-6, &plan, NULL);
  // step 2: send in pointers to M nonuniform points (just x, y in this case)...
  finufft_setpts(plan, M, &x[0], &y[0], NULL, 0, NULL, NULL, NULL);
  // (user should not change x, y nonuniform point arrays here!)
  // step 3: do the planned transform to the c strength data, output to F...
  finufft_execute(plan, &c[0], &F[0]);
  // ... you could now send in new points, and/or do transforms with new c data
  // ... or even adjoint transforms with the same points but now mapping F to c.
  // ...
  // step 4: when done, free the memory used by the plan...
  finufft_destroy(plan);
  // docs-end: guru2d1
  if (ier) return ier;

  // math test, after the documented code:
  constexpr double tol = 1e-6;             // the tolerance requested above
  const int k1         = round(0.45 * N1); // check the answer for mode frequency (k1,k2)
  const int k2         = round(-0.35 * N2);
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
  // index in output array for this frequency pair (k1,k2)
  const auto err = abs(F[k1 + N1 / 2 + (k2 + N2 / 2) * N1] - Ftest) / Fmax;
  if (!(Ffin && Fmax > 0.0 && err < 10 * tol)) {
    fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}

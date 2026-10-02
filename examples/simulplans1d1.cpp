/* Demo two simultaneous FINUFFT plans (A,B) being handled in C++ without
   interacting (or at least without crashing; note that FFTW initialization
   is the only global state of FINUFFT library).
   Using STL double complex vectors, with a math test.
   To compile, see README in this directory. Also see ../docs/cex.rst
   Edited from guru1d1, Barnett 2/15/22
   Usage: ./simulplans1d1
*/

// this is all you must include for the finufft lib...
#include <finufft.h>

// also used in this example...
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

constexpr double pi = 3.14159265358979323846;

double chk1d1(int n, vector<double> &x, vector<complex<double>> &c,
              vector<complex<double>> &F)
// return error in output array F, for n'th mode only, rel to ||F||_inf
{
  const int N = F.size();
  if (n >= N / 2 || n < -N / 2) {
    printf("n out of bounds!\n");
    return NAN;
  }
  vector<complex<double>> terms(x.size());
  transform(c.begin(), c.end(), x.begin(), terms.begin(),
            [&](auto cj, auto xj) { return cj * exp(1i * double(n) * xj); });
  const complex<double> Ftest = reduce(terms.begin(), terms.end());
  const int nout    = n + N / 2; // index in output array for freq mode n
  const auto Fmax   = abs(
      *max_element(F.begin(), F.end(), [](auto a, auto b) { return abs(a) < abs(b); }));
  return abs(F[nout] - Ftest) / Fmax;
}

int main() {
  constexpr double tol = 1e-9;     // desired accuracy for both plans
  constexpr int type = 1, dim = 1; // 1d1
  int64_t Ns[3];                   // guru describes mode array by vector [N1,N2..]
  constexpr int ntransf = 1;       // we want to do a single transform at a time

  constexpr int MA      = 3e6;     // number of nonuniform points    PLAN A
  constexpr int NA      = 1e6;     // number of modes
  constexpr int MB      = 2e6;     // number of nonuniform points    PLAN B, diff sizes
  constexpr int NB      = 1e5;     // number of modes

  finufft_plan planA, planB;       // creates plan structs
  Ns[0]   = NA;
  int ier = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &planA, NULL);
  if (ier) return ier;
  Ns[0] = NB;
  ier   = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &planB, NULL);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);

  // generate some random nonuniform points
  vector<double> xA(MA), xB(MB);
  generate(xA.begin(), xA.end(), [&] { return upi(rng); });
  generate(xB.begin(), xB.end(), [&] { return upi(rng); });

  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  ier = finufft_setpts(planA, MA, &xA[0], NULL, NULL, 0, NULL, NULL, NULL);
  if (ier) return ier;
  ier = finufft_setpts(planB, MB, &xB[0], NULL, NULL, 0, NULL, NULL, NULL);
  if (ier) return ier;

  // generate some complex strengths
  vector<complex<double>> cA(MA), cB(MB);
  generate(cA.begin(), cA.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  generate(cB.begin(), cB.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // allocate output arrays for the Fourier modes...
  vector<complex<double>> FA(NA), FB(NB);
  ier = finufft_execute(planA, &cA[0], &FA[0]);
  if (ier) return ier;
  ier = finufft_execute(planB, &cB[0], &FB[0]);
  if (ier) return ier;

  // change strengths and exec again for fun...
  generate(cA.begin(), cA.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  generate(cB.begin(), cB.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  ier = finufft_execute(planA, &cA[0], &FA[0]);
  if (ier) return ier;
  ier = finufft_execute(planB, &cB[0], &FB[0]);
  if (ier) return ier;
  finufft_destroy(planA);
  finufft_destroy(planB);

  // math checking and reporting...
  constexpr int nA = 116354, nB = 27152;
  const double errA = chk1d1(nA, xA, cA, FA);
  printf("planA: 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n",
         ier, nA, errA);
  const double errB = chk1d1(nB, xB, cB, FB);
  printf("planB: 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n",
         ier, nB, errB);

  return !(errA < 10 * tol && errB < 10 * tol);
}

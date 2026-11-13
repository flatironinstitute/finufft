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
#include <cassert>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;

static const double PI = 3.141592653589793238462643383279502884;

mt19937 rng(12345);
uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);

void strengths(vector<complex<double>> &c) { // fill random complex array
  generate(c.begin(), c.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
}

double chk1d1(int n, vector<double> &x, vector<complex<double>> &c,
              vector<complex<double>> &F)
// return error in output array F, for n'th mode only, rel to ||F||_inf
{
  int N = F.size();
  if (n >= N / 2 || n < -N / 2) {
    printf("n out of bounds!\n");
    return NAN;
  }
  complex<double> Ftest = complex<double>(0, 0);
  for (long unsigned int j = 0; j < x.size(); ++j)
    Ftest += c[j] * exp(1i * double(n) * x[j]);
  int nout    = n + N / 2; // index in output array for freq mode n
  double Fmax = 0.0;       // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  return abs(F[nout] - Ftest) / Fmax;
}

int main() {
  double tol = 1e-9;         // desired accuracy for both plans
  int type = 1, dim = 1;     // 1d1
  int64_t Ns[3];             // guru describes mode array by vector [N1,N2..]
  int ntransf = 1;           // we want to do a single transform at a time

  int MA      = 3e6;         // number of nonuniform points    PLAN A
  int NA      = 1e6;         // number of modes
  int MB      = 2e6;         // number of nonuniform points    PLAN B, diff sizes
  int NB      = 1e5;         // number of modes

  finufft_plan planA, planB; // creates plan structs
  Ns[0]   = NA;
  int ier = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &planA, nullptr);
  if (ier) return ier;
  Ns[0] = NB;
  ier   = finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &planB, nullptr);
  if (ier) return ier;

  // generate some random nonuniform points
  vector<double> xA(MA), xB(MB);
  generate(xA.begin(), xA.end(), [&] { return upi(rng); }); // uniform random in [-pi,pi)
  generate(xB.begin(), xB.end(), [&] { return upi(rng); }); // uniform random in [-pi,pi)

  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  ier = finufft_setpts(planA, MA, xA.data(), nullptr, nullptr, 0, nullptr, nullptr,
                       nullptr);
  if (ier) return ier;
  ier = finufft_setpts(planB, MB, xB.data(), nullptr, nullptr, 0, nullptr, nullptr,
                       nullptr);
  if (ier) return ier;

  // generate some complex strengths
  vector<complex<double>> cA(MA), cB(MB);
  strengths(cA);
  strengths(cB);

  // allocate output arrays for the Fourier modes...
  vector<complex<double>> FA(NA), FB(NB);
  int ierA = finufft_execute(planA, cA.data(), FA.data());
  int ierB = finufft_execute(planB, cB.data(), FB.data());
  if (ierA || ierB) return ierA + ierB;

  // change strengths and exec again for fun...
  strengths(cA);
  strengths(cB);
  ierA = finufft_execute(planA, cA.data(), FA.data());
  ierB = finufft_execute(planB, cB.data(), FB.data());
  finufft_destroy(planA);
  finufft_destroy(planB);
  if (ierA || ierB) return ierA + ierB;

  // math checking and reporting...
  int n       = 116354;
  double errA = chk1d1(n, xA, cA, FA);
  printf("planA: 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n",
         ierA, n, errA);
  n           = 27152;
  double errB = chk1d1(n, xB, cB, FB);
  printf("planB: 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n",
         ierB, n, errB);

  if (!(errA <= 10 * tol && errB <= 10 * tol)) return 1;
  return ierA + ierB;
}

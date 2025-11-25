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
#include <cassert>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

static const double PI = 3.141592653589793238462643383279502884;

void strengths(std::vector<std::complex<double>> &c) { // fill random complex array
  const std::complex<double> I(0.0, 1.0);
  for (long unsigned int j = 0; j < c.size(); ++j)
    c[j] = 2 * ((double)std::rand() / RAND_MAX) - 1 +
           I * (2 * ((double)std::rand() / RAND_MAX) - 1);
}

double chk1d1(int n, std::vector<double> &x, std::vector<std::complex<double>> &c,
              std::vector<std::complex<double>> &F)
// return error in output array F, for n'th mode only, rel to ||F||_inf
{
  int N = F.size();
  if (n >= N / 2 || n < -N / 2) {
    printf("n out of bounds!\n");
    return NAN;
  }
  std::complex<double> Ftest = std::complex<double>(0, 0);
  const std::complex<double> I(0.0, 1.0);
  for (long unsigned int j = 0; j < x.size(); ++j)
    Ftest += c[j] * std::exp(I * (double)n * x[j]);
  int nout    = n + N / 2; // index in output array for freq mode n
  double Fmax = 0.0;       // compute inf norm of F
  for (int m = 0; m < N; ++m) {
    double aF = std::abs(F[m]);
    if (aF > Fmax) Fmax = aF;
  }
  return std::abs(F[nout] - Ftest) / Fmax;
}

int main(int argc, char *argv[]) {
  double tol = 1e-9;         // desired accuracy for both plans
  int type = 1, dim = 1;     // 1d1
  int64_t Ns[3];             // guru describes mode array by vector [N1,N2..]
  int ntransf = 1;           // we want to do a single transform at a time

  int MA = 3e6;              // number of nonuniform points    PLAN A
  int NA = 1e6;              // number of modes
  int MB = 2e6;              // number of nonuniform points    PLAN B, diff sizes
  int NB = 1e5;              // number of modes

  finufft_plan planA, planB; // creates plan structs
  Ns[0] = NA;
  finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &planA, nullptr);
  Ns[0] = NB;
  finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &planB, nullptr);

  // generate some random nonuniform points
  std::vector<double> xA(MA), xB(MB);
  for (int j = 0; j < MA; ++j)
    xA[j] = PI * (2 * ((double)std::rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)
  for (int j = 0; j < MB; ++j)
    xB[j] = PI * (2 * ((double)std::rand() / RAND_MAX) - 1); // uniform random in [-pi,pi)

  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(planA, MA, &xA[0], nullptr, nullptr, 0, nullptr, nullptr, nullptr);
  finufft_setpts(planB, MB, &xB[0], nullptr, nullptr, 0, nullptr, nullptr, nullptr);

  // generate some complex strengths
  std::vector<std::complex<double>> cA(MA), cB(MB);
  strengths(cA);
  strengths(cB);

  // allocate output arrays for the Fourier modes...
  std::vector<std::complex<double>> FA(NA), FB(NB);
  int ierA = finufft_execute(planA, &cA[0], &FA[0]);
  int ierB = finufft_execute(planB, &cB[0], &FB[0]);

  // change strengths and exec again for fun...
  strengths(cA);
  strengths(cB);
  ierA = finufft_execute(planA, &cA[0], &FA[0]);
  ierB = finufft_execute(planB, &cB[0], &FB[0]);
  finufft_destroy(planA);
  finufft_destroy(planB);

  // math checking and reporting...
  int n       = 116354;
  double errA = chk1d1(n, xA, cA, FA);
  printf("planA: 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n",
         ierA, n, errA);
  n           = 27152;
  double errB = chk1d1(n, xB, cB, FB);
  printf("planB: 1D type-1 double-prec NUFFT done. ier=%d, rel err in F[%d] is %.3g\n",
         ierB, n, errB);

  return ierA + ierB;
}

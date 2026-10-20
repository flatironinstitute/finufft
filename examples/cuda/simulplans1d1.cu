/* Example of calling the cuFINUFFT library from C++, using thrust device
   vectors, with a math test. GPU version of examples/simulplans1d1.cpp.
   Demo of two simultaneous plans (A,B) handled without interacting.
   Edited from guru1d1.cu, Barbone 11/12/26. To compile, see README in this
   directory. Usage: ./simulplans1d1
*/
#include <cufinufft.h>

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <thrust/device_vector.h>
#include <vector>
using namespace std;

constexpr double PI = 3.141592653589793238462643383279502884;

double chk1d1(int n, vector<double> &x, vector<complex<double>> &c,
              vector<complex<double>> &F)
// return error in output array F, for n'th mode only, rel to ||F||_inf.
{
  const int N = F.size();
  const complex<double> I(0.0, 1.0);
  complex<double> Ftest(0.0, 0.0);
  for (size_t j = 0; j < x.size(); ++j) Ftest += c[j] * exp(I * double(n) * x[j]);
  const int nout = n + N / 2; // index in output array for freq mode n
  double Fmax    = 0.0;       // compute inf norm of F
  for (const auto &Fm : F) {
    const double a = abs(Fm);
    if (a > Fmax) Fmax = a;
  }
  return abs(F[nout] - Ftest) / Fmax;
}

int main() {
  constexpr double tol = 1e-9;     // desired accuracy for both plans
  constexpr int type = 1, dim = 1; // 1d1
  constexpr int ntransf = 1;       // we want to do a single transform at a time

  constexpr int MA      = 3e6;     // number of nonuniform points    PLAN A
  constexpr int NA      = 1e6;     // number of modes
  constexpr int MB      = 2e6;     // number of nonuniform points    PLAN B, diff sizes
  constexpr int NB      = 1e5;     // number of modes

  // guru describes mode array by vector [N1,N2..]
  const int64_t NsA[3] = {NA, 0, 0}, NsB[3] = {NB, 0, 0};
  cufinufft_plan planA, planB; // creates plan structs
  int ier = cufinufft_makeplan(type, dim, NsA, +1, ntransf, tol, &planA, nullptr);
  if (ier) return ier;
  ier = cufinufft_makeplan(type, dim, NsB, +1, ntransf, tol, &planB, nullptr);
  if (ier) return ier;

  mt19937 rng(12345);
  uniform_real_distribution<double> upi(-PI, PI), u1(-1.0, 1.0);

  // generate some random nonuniform points
  vector<double> xA(MA), xB(MB);
  generate(xA.begin(), xA.end(), [&] { return upi(rng); });
  generate(xB.begin(), xB.end(), [&] { return upi(rng); });

  thrust::device_vector<double> d_xA = xA, d_xB = xB; // copy x to the GPU
  ier = cufinufft_setpts(planA, MA, d_xA.data().get(), nullptr, nullptr, 0, nullptr,
                         nullptr, nullptr);
  if (ier) return ier;
  ier = cufinufft_setpts(planB, MB, d_xB.data().get(), nullptr, nullptr, 0, nullptr,
                         nullptr, nullptr);
  if (ier) return ier;

  // generate some complex strengths
  vector<complex<double>> cA(MA), cB(MB);
  generate(cA.begin(), cA.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  generate(cB.begin(), cB.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });

  // allocate output arrays for the Fourier modes...
  vector<complex<double>> FA(NA), FB(NB);
  thrust::device_vector<complex<double>> d_cA = cA, d_cB = cB; // copy strengths to GPU
  thrust::device_vector<complex<double>> d_FA = FA, d_FB = FB; // room for the modes
  auto *dcA = reinterpret_cast<cuDoubleComplex *>(d_cA.data().get()); // same layout
  auto *dcB = reinterpret_cast<cuDoubleComplex *>(d_cB.data().get());
  auto *dFA = reinterpret_cast<cuDoubleComplex *>(d_FA.data().get());
  auto *dFB = reinterpret_cast<cuDoubleComplex *>(d_FB.data().get());
  ier       = cufinufft_execute(planA, dcA, dFA);
  if (ier) return ier;
  ier = cufinufft_execute(planB, dcB, dFB);
  if (ier) return ier;

  // change strengths and exec again for fun...
  generate(cA.begin(), cA.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  generate(cB.begin(), cB.end(), [&] { return complex<double>{u1(rng), u1(rng)}; });
  d_cA = cA;
  d_cB = cB;
  ier  = cufinufft_execute(planA, dcA, dFA);
  if (ier) return ier;
  ier = cufinufft_execute(planB, dcB, dFB);
  if (ier) return ier;
  thrust::copy(d_FA.begin(), d_FA.end(), FA.begin()); // copy FA back to the CPU
  thrust::copy(d_FB.begin(), d_FB.end(), FB.begin()); // copy FB back to the CPU
  cufinufft_destroy(planA);
  cufinufft_destroy(planB);

  // math checking and reporting, for the n'th mode of each plan...
  constexpr int nA = 116354, nB = 27152;
  const double errA = chk1d1(nA, xA, cA, FA);
  const double errB = chk1d1(nB, xB, cB, FB);
  const double err  = errA > errB ? errA : errB;
  if (!(errA <= 10 * tol && errB <= 10 * tol)) { // written so a NaN error fails
    fprintf(stderr, "FAILED: rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  printf("rel err %.3g\n", err);
  return 0;
}

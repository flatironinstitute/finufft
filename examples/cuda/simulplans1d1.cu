/* Demo two simultaneous CUFINUFFT plans (A,B) on the GPU, mirroring
   examples/simulplans1d1.cpp. Barnett 2/15/22; GPU mirror 10/1/26.
   Usage: ./simulplans1d1
*/

#include <cufinufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

constexpr double pi = 3.14159265358979323846;

int main() {
  constexpr double tol = 1e-9;     // desired accuracy for both plans
  constexpr int type = 1, dim = 1; // 1d1
  constexpr int ntransf = 1;       // we want to do a single transform at a time
  constexpr std::complex<double> I(0.0, 1.0); // the imaginary unit
  constexpr int MA     = 3e6;                 // number of nonuniform points    PLAN A
  constexpr int NA     = 1e6;                 // number of modes
  constexpr int MB     = 2e6; // number of nonuniform points    PLAN B, diff sizes
  constexpr int NB     = 1e5; // number of modes
  const int64_t NsA[3] = {NA, 0, 0}, NsB[3] = {NB, 0, 0}; // guru mode arrays [N1,N2..]
  cufinufft_plan planA, planB;                            // creates plan structs
  int ier = cufinufft_makeplan(type, dim, NsA, +1, ntransf, tol, &planA, nullptr);
  if (ier > 1) return ier;
  ier = cufinufft_makeplan(type, dim, NsB, +1, ntransf, tol, &planB, nullptr);
  if (ier > 1) return ier;
  std::mt19937 rng(12345); // fixed seed, so the run is deterministic
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  // generate some random nonuniform points in [-pi,pi]
  std::vector<double> xA(MA), xB(MB);
  std::generate(xA.begin(), xA.end(), [&] { return upi(rng); });
  std::generate(xB.begin(), xB.end(), [&] { return upi(rng); });
  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  const thrust::device_vector<double> d_xA = xA, d_xB = xB;
  thrust::device_vector<cuDoubleComplex> d_cA(MA, thrust::no_init),
      d_cB(MB, thrust::no_init), d_FA(NA, thrust::no_init), d_FB(NB, thrust::no_init);
  ier = cufinufft_setpts(planA, MA, thrust::raw_pointer_cast(d_xA.data()), nullptr,
                         nullptr, 0, nullptr, nullptr, nullptr);
  if (ier > 1) return ier;
  ier = cufinufft_setpts(planB, MB, thrust::raw_pointer_cast(d_xB.data()), nullptr,
                         nullptr, 0, nullptr, nullptr, nullptr);
  if (ier > 1) return ier;
  // random complex strengths, and output arrays for the Fourier modes...
  std::vector<std::complex<double>> cA(MA), cB(MB), FA(NA), FB(NB);
  const auto *h_cA = reinterpret_cast<const cuDoubleComplex *>(cA.data());
  const auto *h_cB = reinterpret_cast<const cuDoubleComplex *>(cB.data());
  std::generate(cA.begin(), cA.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });
  std::generate(cB.begin(), cB.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });
  thrust::copy(h_cA, h_cA + MA, d_cA.begin());
  thrust::copy(h_cB, h_cB + MB, d_cB.begin());
  ier = cufinufft_execute(planA, thrust::raw_pointer_cast(d_cA.data()),
                          thrust::raw_pointer_cast(d_FA.data()));
  if (ier > 1) return ier;
  ier = cufinufft_execute(planB, thrust::raw_pointer_cast(d_cB.data()),
                          thrust::raw_pointer_cast(d_FB.data()));
  if (ier > 1) return ier;
  // change strengths and exec again for fun...
  std::generate(cA.begin(), cA.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });
  std::generate(cB.begin(), cB.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });
  thrust::copy(h_cA, h_cA + MA, d_cA.begin());
  thrust::copy(h_cB, h_cB + MB, d_cB.begin());
  ier = cufinufft_execute(planA, thrust::raw_pointer_cast(d_cA.data()),
                          thrust::raw_pointer_cast(d_FA.data()));
  if (ier > 1) return ier;
  ier = cufinufft_execute(planB, thrust::raw_pointer_cast(d_cB.data()),
                          thrust::raw_pointer_cast(d_FB.data()));
  if (ier > 1 || cudaDeviceSynchronize() != cudaSuccess) return ier > 1 ? ier : 1;
  thrust::copy(d_FA.begin(), d_FA.end(), reinterpret_cast<cuDoubleComplex *>(FA.data()));
  thrust::copy(d_FB.begin(), d_FB.end(), reinterpret_cast<cuDoubleComplex *>(FB.data()));
  cufinufft_destroy(planA);
  cufinufft_destroy(planB);
  // math checking and reporting, for the n'th mode of each plan...
  constexpr int nA = 116354, nB = 27152;
  std::vector<std::complex<double>> termsA(MA);
  std::transform(cA.begin(), cA.end(), xA.begin(), termsA.begin(),
                 [&](auto cj, auto xj) { return cj * std::exp(I * double(nA) * xj); });
  const auto FtestA = std::reduce(termsA.begin(), termsA.end());
  std::vector<std::complex<double>> termsB(MB);
  std::transform(cB.begin(), cB.end(), xB.begin(), termsB.begin(),
                 [&](auto cj, auto xj) { return cj * std::exp(I * double(nB) * xj); });
  const auto FtestB = std::reduce(termsB.begin(), termsB.end());
  const auto FmaxA = std::abs(*std::max_element(
      FA.begin(), FA.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
  const auto FmaxB = std::abs(*std::max_element(
      FB.begin(), FB.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
  const auto errA  = std::abs(FA[nA + NA / 2] - FtestA) / FmaxA; // nA + NA/2: mode nA
  const auto errB  = std::abs(FB[nB + NB / 2] - FtestB) / FmaxB;
  printf(
      "planA: 1D type-1 double-prec NUFFT done (GPU). ier=%d, rel err in F[%d] is %.3g\n",
      ier, nA, errA);
  printf(
      "planB: 1D type-1 double-prec NUFFT done (GPU). ier=%d, rel err in F[%d] is %.3g\n",
      ier, nB, errB);
  return !(errA < 10 * tol && errB < 10 * tol);
}

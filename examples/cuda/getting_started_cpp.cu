// Getting-started example of the 1D type-1 cufinufft transform, using thrust
// device vectors for all GPU memory (C++17). Double precision.
#include <cufinufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

constexpr double pi = 3.14159265358979323846;

int main() {
  const int M    = 100000;                // number of nonuniform points
  const int N    = 10000;                 // number of output modes
  const int type = 1, isign = 1;          // 1D type-1 transform, + sign
  const double tol = 1e-9;                // requested tolerance
  const std::complex<double> I(0.0, 1.0); // the imaginary unit

  // docs-start: getting-started-cpp
  // generate nonuniform points (x) and complex strengths (c)...
  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M), F(N);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // device arrays free themselves; a failed allocation or copy throws (exit != 0)
  thrust::device_vector<double> d_x = x;
  const auto *h_c                   = reinterpret_cast<const cuDoubleComplex *>(c.data());
  thrust::device_vector<cuDoubleComplex> d_c(h_c, h_c + M);
  thrust::device_vector<cuDoubleComplex> d_f(N);
  // docs-end: getting-started-cpp

  // make the cufinufft plan for a 1D type-1 transform at tolerance tol.
  // Any ier above 1 is an error; 1 is a warning and the result is still usable.
  // docs-start: getting-started-cpp-call
  int64_t modes[1] = {N}; // grid size as an array
  cufinufft_plan plan;    // store cufinufft plan
  int ier = cufinufft_makeplan(type, 1, modes, isign, 1, tol, &plan, nullptr);
  if (ier > 1) return ier;

  // set the frequencies of the nonuniform points
  ier = cufinufft_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()), nullptr, nullptr,
                         0, nullptr, nullptr, nullptr);
  if (ier > 1) return ier;

  // execute the plan on the strengths, writing the modes into d_f
  ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_f.data()));
  if (ier > 1) return ier;

  // copy the result back onto the host and destroy the plan
  // docs-end: getting-started-cpp-call
  thrust::copy(d_f.begin(), d_f.end(), reinterpret_cast<cuDoubleComplex *>(F.data()));
  cufinufft_destroy(plan);

  // check the answer at one mode against a direct (slow) sum on the host...
  constexpr int k = 1425; // mode frequency to check
  std::vector<std::complex<double>> terms(M);
  std::transform(c.begin(), c.end(), x.begin(), terms.begin(),
                 [&](auto cj, auto xj) { return cj * std::exp(I * double(k) * xj); });
  const auto Ftest = std::reduce(terms.begin(), terms.end());
  double Fmax      = 0.0; // compute inf norm of F (a NaN entry makes Fmax NaN)
  for (const auto &v : F) {
    const double a = std::abs(v);
    if (a > Fmax || !std::isfinite(a)) Fmax = a;
  }
  const auto err = std::abs(F[k + N / 2] - Ftest) / Fmax; // k + N/2: index of mode k

  if (!std::isfinite(Fmax) || !std::isfinite(err) || err > 10 * tol) {
    std::fprintf(stderr, "FAILED: rel err %.3g exceeds 10*tol %.3g\n", err, 10 * tol);
    return 1;
  }
  std::printf("getting_started_cpp: rel err in F[%d] is %.3g\n", k, err);
  return 0;
}

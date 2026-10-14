#include <cufinufft.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

int main() {
  const int M = 100000, N = 1000; // nonuniform points, modes
  const double tol = 1e-9, pi = 3.14159265358979323846;
  const std::complex<double> I(0.0, 1.0);

  std::mt19937 rng(12345); // fixed seed: reproducible example output
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M), F(N);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  // cuDoubleComplex and std::complex<double> share layout, so the casts copy
  // the host vectors with no element-wise kernel.
  const auto *c_p = reinterpret_cast<const cuDoubleComplex *>(c.data());
  thrust::device_vector<double> d_x(x.begin(), x.end());
  thrust::device_vector<cuDoubleComplex> d_c(c_p, c_p + M);
  thrust::device_vector<cuDoubleComplex> d_F(N);

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  cufinufft_plan plan;
  const int64_t n_modes[1] = {N};
  int ier                  = cufinufft_makeplan(1, 1, n_modes, +1, 1, tol, &plan, &opts);
  if (ier == 0) {
    ier = cufinufft_setpts(plan, M, thrust::raw_pointer_cast(d_x.data()), nullptr,
                           nullptr, 0, nullptr, nullptr, nullptr);
    if (ier == 0)
      ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                              thrust::raw_pointer_cast(d_F.data()));
    const int dier = cufinufft_destroy(plan);
    if (ier == 0) ier = dier;
  }
  if (ier != 0) {
    std::fprintf(stderr, "cufinufft failed, ier=%d\n", ier);
    return 1;
  }
  auto *F_p = reinterpret_cast<cuDoubleComplex *>(F.data());
  thrust::copy(d_F.begin(), d_F.end(), F_p);

  const int k = N / 3; // check the answer just for this mode frequency
  std::complex<double> Fk(0.0, 0.0);
  for (int j = 0; j < M; ++j) Fk += c[j] * std::exp(I * (double)k * x[j]);
  double Fmax = 0.0; // inf norm of F for the relative error
  bool Ffin   = true;
  for (const auto &Fm : F) {
    const double a = std::abs(Fm);
    Ffin &= std::isfinite(a);
    Fmax = std::max(Fmax, a);
  }
  const double err = std::abs(F[k + N / 2] - Fk) / Fmax;
  if (!(Ffin && Fmax > 0.0 && err < 10 * tol)) {
    std::fprintf(stderr, "FAILED: F non-finite or rel err %.3g > %.3g\n", err, 10 * tol);
    return 1;
  }
  std::printf("cufinufft 1d1: rel err in mode %d is %.3g\n", k, err);
  return 0;
}

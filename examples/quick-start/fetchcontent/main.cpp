#include <finufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  const int M = 100000, N = 1000;
  const double tol = 1e-9, pi = 3.14159265358979323846;
  const std::complex<double> I(0.0, 1.0);

  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M), F(N);
  for (int j = 0; j < M; ++j) {
    x[j] = pi * (2 * (double)rand() / RAND_MAX - 1);
    c[j] = {2 * (double)rand() / RAND_MAX - 1, 2 * (double)rand() / RAND_MAX - 1};
  }

  finufft_opts opts;
  finufft_default_opts(&opts);
  const int ier = finufft1d1(M, x.data(), c.data(), +1, tol, N, F.data(), &opts);
  if (ier != 0) {
    std::printf("finufft1d1 failed, ier=%d\n", ier);
    return 1;
  }

  const int k = N / 3;
  std::complex<double> Fk(0.0, 0.0);
  for (int j = 0; j < M; ++j) Fk += c[j] * std::exp(I * (double)k * x[j]);
  double Fmax = 0.0;
  for (int m = 0; m < N; ++m) Fmax = std::max(Fmax, std::abs(F[m]));

  const double err = std::abs(F[k + N / 2] - Fk) / Fmax;
  std::printf("finufft1d1: ier=%d, rel err in mode %d is %.3g\n", ier, k, err);
  return std::isfinite(err) && err < 1e-7 ? 0 : 1;
}

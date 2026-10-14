#include <finufft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

int main() {
  constexpr int M = 100000, N = 1000; // nonuniform points, modes
  constexpr double tol = 1e-9, pi = 3.14159265358979323846;
  const std::complex<double> I(0.0, 1.0);

  std::mt19937 rng(12345); // fixed seed: reproducible example output
  std::uniform_real_distribution<double> upi(-pi, pi), u1(-1.0, 1.0);
  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M), F(N);
  std::generate(x.begin(), x.end(), [&] { return upi(rng); });
  std::generate(c.begin(), c.end(),
                [&] { return std::complex<double>{u1(rng), u1(rng)}; });

  finufft_opts opts;
  finufft_default_opts(&opts);
  const int ier = finufft1d1(M, x.data(), c.data(), +1, tol, N, F.data(), &opts);
  if (ier) return ier;

  constexpr int k = N / 3; // check the answer just for this mode frequency
  std::complex<double> Fk(0.0, 0.0);
  for (int j = 0; j < M; ++j) Fk += c[j] * std::exp(I * double(k) * x[j]);
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
  std::printf("finufft1d1: ier=%d, rel err in mode %d is %.3g\n", ier, k, err);
  return 0;
}

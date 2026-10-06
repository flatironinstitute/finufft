#include <finufft.h>

#include <complex>
#include <vector>

int main() {
  int N = 16;
  std::vector<double> x(N);
  std::vector<std::complex<double>> c(N), F(N);
  for (int j = 0; j < N; ++j) {
    x[j] = 0.1 * j;
    c[j] = {1.0, 0.0};
  }
  finufft_opts opts;
  finufft_default_opts(&opts);
  return finufft1d1(N, x.data(), c.data(), +1, 1e-6, N, F.data(), &opts) == 0 ? 0 : 1;
}

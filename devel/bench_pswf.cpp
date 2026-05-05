#include <chrono>
#include <cmath>
#include <cstdio>
#include <finufft_common/pswf.h>

int main() {
  using namespace finufft::common;
  using clock = std::chrono::steady_clock;

  const double c = 30.0;
  const int N    = 10'000'000;

  // Warm up / trigger precomputation (not timed)
  volatile double warmup = pswf(c, 0.0);
  (void)warmup;

  // Evaluate pswf at N uniformly-spaced points in [-1, 1]
  auto t0    = clock::now();
  double sum = 0.0;
  for (int i = 0; i < N; ++i) {
    double x = -1.0 + 2.0 * i / (N - 1);
    sum += pswf(c, x);
  }
  auto t1 = clock::now();

  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  std::printf("N = %d points, c = %.1f\n", N, c);
  std::printf("Elapsed: %.4f s\n", elapsed);
  std::printf("Throughput: %.3f Meval/s\n", N / elapsed / 1e6);
  std::printf("Checksum (prevent optimization): %.6f\n", sum);
}

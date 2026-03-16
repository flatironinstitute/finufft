/* Regression test for thread-safe concurrent execute() on the same plan.

   Creates a single 1D type-1 plan and then runs finufft_execute from
   multiple threads simultaneously, each into its own output array.
   Correctness is verified against a direct (slow) Fourier transform.
   This catches data races in internal scratch workspace allocation.

   Usage: ./threadsafe_execute     (exit 0 = pass, >0 = fail)
   Barbone, Mar 2026.
*/

#include <finufft.h>
#include <finufft_common/constants.h>
#include <finufft_opts.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "utils/dirft1d.hpp"
#include "utils/norms.hpp"

int main() {
  constexpr int nthreads = 4;
  constexpr int nreps    = 16;
  constexpr int M        = 400;
  constexpr int64_t N1   = 2048;
  constexpr double tol   = 1e-12;

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.nthreads = 1; // crucial: parallelism is across concurrent plan executes
  opts.debug    = 0;

  std::vector<double> x(M);
  std::vector<std::complex<double>> c(M), ref(N1);
  for (int j = 0; j < M; ++j) {
    double t = static_cast<double>(j) / M;
    x[j]     = -finufft::common::PI + 2.0 * finufft::common::PI * t;
    c[j]     = std::complex<double>(0.5 * std::cos(13.0 * t) + 0.25 * std::sin(7.0 * t),
                                    0.75 * std::sin(11.0 * t) - 0.2 * std::cos(5.0 * t));
  }

  int64_t Ns[3] = {N1, 1, 1};
  finufft_plan plan;
  int ier = finufft_makeplan(1, 1, Ns, +1, 1, tol, &plan, &opts);
  if (ier != 0) {
    std::fprintf(stderr, "finufft_makeplan failed: ier=%d\n", ier);
    return ier;
  }
  ier = finufft_setpts(plan, M, x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr);
  if (ier != 0) {
    std::fprintf(stderr, "finufft_setpts failed: ier=%d\n", ier);
    finufft_destroy(plan);
    return ier;
  }

  dirft1d1<int64_t>(M, x, c, +1, N1, ref);

  std::vector<int> failures(nthreads, 0);

  std::vector<std::thread> workers;
  workers.reserve(nthreads);
  for (int tid = 0; tid < nthreads; ++tid) {
    workers.emplace_back([&, tid]() {
      std::vector<std::complex<double>> out(N1);
      for (int rep = 0; rep < nreps; ++rep) {
        int local_ier = finufft_execute(plan, c.data(), out.data());
        double relerr = relerrtwonorm(N1, ref.data(), out.data());
        if (local_ier != 0 || relerr > 10.0 * tol) {
          failures[tid] = 1;
          std::fprintf(stderr, "thread %d rep %d failed: ier=%d relerr=%.3g\n", tid, rep,
                       local_ier, relerr);
          return;
        }
      }
    });
  }

  for (auto &worker : workers) worker.join();

  finufft_destroy(plan);
  return *std::max_element(failures.begin(), failures.end());
}

#include <benchmark/benchmark.h>

#include <fftw3.h>
#include <finufft.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

using cd = std::complex<double>;

static void BM_finufft_guru_type1_realinput_plan_and_execute(benchmark::State &state) {
  constexpr int type  = 1;
  constexpr int dim   = 1;
  constexpr int iflag = +1;
  constexpr int ntr   = 1;

  constexpr int64_t M  = 3000; // nonuniform points
  constexpr int64_t N1 = 1000; // number of modes
  int64_t nmodes[1]    = {N1};

  constexpr double eps = 1e-9;

  // Fixed inputs for all iterations (plan will reference x during execute).
  std::vector<double> x(M);
  std::vector<cd> c(M);
  std::vector<cd> f(N1);

  // Linspace-style coordinates in [-pi, pi) and deterministic real-valued strengths.
  // Avoid including +pi endpoint to keep within the canonical fold interval.
  const double x0 = -M_PI;
  const double x1 = M_PI;
  const double dx = (x1 - x0) / static_cast<double>(M); // endpoint excluded

  for (int64_t j = 0; j < M; ++j) {
    const double t   = x0 + dx * static_cast<double>(j);
    x[j]             = t;
    const double val = std::sin(3.0 * t) + 0.25 * std::cos(11.0 * t);
    c[j]             = cd(val, 0.0); // "real-input": imag=0
  }

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.upsampfac         = 1.25;          // requested
  opts.fftw              = FFTW_ESTIMATE; // requested
  opts.spread_kerformula = state.range(0);

  for (auto _ : state) {
    finufft_plan plan = nullptr;

    int ier = finufft_makeplan(type, dim, nmodes, iflag, ntr, eps, &plan, &opts);
    if (ier > 1) {
      state.SkipWithError("finufft_makeplan failed");
      break;
    }

    ier =
        finufft_setpts(plan, M, x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr);
    if (ier > 1) {
      finufft_destroy(plan);
      state.SkipWithError("finufft_setpts failed");
      break;
    }

    ier = finufft_execute(plan, c.data(), f.data());
    if (ier > 1) {
      finufft_destroy(plan);
      state.SkipWithError("finufft_execute failed");
      break;
    }

    finufft_destroy(plan);

    benchmark::DoNotOptimize(f.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_finufft_guru_type1_realinput_plan_and_execute)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(0)  // default (prolate/PSWF, formula 8)
    ->Arg(1)  // ES (exponential of semicircle, v2.4.1 kernel)
    ->Arg(3)  // Kaiser-Bessel
    ->Arg(4); // deplinthed Kaiser-Bessel

BENCHMARK_MAIN();

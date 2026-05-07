#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <typeinfo>

#include <benchmark/benchmark.h>
#include <finufft.h>
#include <random>
#ifndef FINUFFT_USE_DUCC0
#include <fftw3.h>
#endif

static const double PI       = 3.141592653589793238462643383279502884;
static const auto BENCH_NAME = "perftest/benchmarks/test_benchmark::FINUFFT";

struct Params {
  std::tuple<int64_t, int64_t, int64_t> Nd;
  int ntransf;
  int64_t M;
  double tol;

  Params(std::tuple<int64_t, int64_t, int64_t> Nd, int ntransf, int64_t M, double tol)
      : Nd(Nd), ntransf(ntransf), M(M), tol(tol) {}

  friend std::ostream &operator<<(std::ostream &outs, const Params &params) {
    return outs << " N1 = " << std::get<0>(params.Nd)
                << " N2 = " << std::get<1>(params.Nd)
                << " N3 = " << std::get<2>(params.Nd) << " ntransf = " << params.ntransf
                << " M = " << params.M << " tol = " << params.tol;
  }
};

static int types[]     = {1, 2, 3};
static double sigmas[] = {1.25, 2.00};

static Params float_benchmarks[] = {
    {std::make_tuple(10000, 1, 1), 1, (int64_t)1e7, 1e-4},
    {std::make_tuple(320, 320, 1), 1, (int64_t)1e7, 1e-5},
};
static Params double_benchmarks[] = {
    {std::make_tuple(10000, 1, 1), 1, (int64_t)1e7, 1e-9},
    {std::make_tuple(320, 320, 1), 1, (int64_t)1e7, 1e-9},
};

template<typename T> void register_benchmark(Params test_opts) {
  for (auto &type : types) {
    for (auto &sigma : sigmas) {
      std::stringstream benchmark_name;
      benchmark_name << BENCH_NAME << test_opts << " type = " << type
                     << " sigma = " << sigma << " prec = " << typeid(T).name();
      benchmark::RegisterBenchmark(benchmark_name.str(), [=](benchmark::State &state) {
        const int ntransf    = test_opts.ntransf;
        const int64_t M      = test_opts.M;
        const long int Nd[3] = {std::get<0>(test_opts.Nd), std::get<1>(test_opts.Nd),
                                std::get<2>(test_opts.Nd)};
        const long N         = Nd[0] * Nd[1] * Nd[2];
        const int dim        = Nd[2] > 1 ? 3 : Nd[1] > 1 ? 2 : 1;
        constexpr int iflag  = 1;
        double tol           = test_opts.tol;

        std::vector<T> x(M * ntransf), y(M * ntransf), z(M * ntransf);
        std::vector<T> s(N * ntransf), t(N * ntransf), u(N * ntransf);
        std::vector<std::complex<T>> c(M * ntransf), fk(N * ntransf);

        std::default_random_engine eng{42};
        std::uniform_real_distribution<T> dist11(-1, 1);
        auto randm11 = [&eng, &dist11]() {
          return dist11(eng);
        };

        for (int64_t i = 0; i < M; i++) {
          x[i] = PI * randm11();
          y[i] = PI * randm11();
          z[i] = PI * randm11();
        }
        for (int64_t i = M; i < M * ntransf; ++i) {
          int64_t j = i % M;
          x[i]      = x[j];
          y[i]      = y[j];
          z[i]      = z[j];
        }

        if (type == 1) {
          for (int i = 0; i < M * ntransf; i++) {
            c[i].real(randm11());
            c[i].imag(randm11());
          }
        } else if (type == 2) {
          for (int i = 0; i < N * ntransf; i++) {
            fk[i].real(randm11());
            fk[i].imag(randm11());
          }
        } else if (type == 3) {
          for (int i = 0; i < M * ntransf; i++) {
            c[i].real(randm11());
            c[i].imag(randm11());
          }
          for (int i = 0; i < N * ntransf; i++) {
            s[i] = PI * randm11();
            t[i] = PI * randm11();
            u[i] = PI * randm11();
          }
        }

        T *x_p = dim >= 1 ? x.data() : nullptr;
        T *y_p = dim >= 2 ? y.data() : nullptr;
        T *z_p = dim == 3 ? z.data() : nullptr;
        T *s_p = type == 3 && dim >= 1 ? s.data() : nullptr;
        T *t_p = type == 3 && dim >= 2 ? t.data() : nullptr;
        T *u_p = type == 3 && dim == 3 ? u.data() : nullptr;
        finufft_opts opts;
        finufft_default_opts(&opts);
        opts.upsampfac = sigma;
        opts.nthreads  = 1;
        opts.showwarn  = 0;
        for (auto _ : state) {
          if constexpr (std::is_same_v<T, double>) {
            finufft_plan_s *plan{nullptr};
            finufft_makeplan(type, dim, Nd, iflag, ntransf, tol, &plan, &opts);
            finufft_setpts(plan, M, x_p, y_p, z_p, N, s_p, t_p, u_p);
            finufft_execute(plan, c.data(), fk.data());
            finufft_destroy(plan);
            benchmark::ClobberMemory();
          } else if constexpr (std::is_same_v<T, float>) {
            finufftf_plan_s *plan{nullptr};
            finufftf_makeplan(type, dim, Nd, iflag, ntransf, tol, &plan, &opts);
            finufftf_setpts(plan, M, x_p, y_p, z_p, N, s_p, t_p, u_p);
            finufftf_execute(plan, c.data(), fk.data());
            finufftf_destroy(plan);
            benchmark::ClobberMemory();
          }
          state.SetItemsProcessed(N + M);
        }
      });
    }
  }
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);

  for (auto &test_opts : float_benchmarks) {
    register_benchmark<float>(test_opts);
  }
  for (auto &test_opts : double_benchmarks) {
    register_benchmark<double>(test_opts);
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}

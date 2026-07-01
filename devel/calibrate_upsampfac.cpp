// Calibration data generator for the complexity-based upsampfac selector
// (include/finufft/heuristics.hpp). For each (prec,dim,type,tol,N,M,threads) config
// it sweeps the forced upsampfac sigma over the feasible window
// [analytic_upsampfac, MAX_AUTO_UPSAMPFAC] (step --sigma_step), times setpts+execute
// (min over --n_runs, same min-time pattern as perftest.cpp), and emits:
//   - a per-sigma detail CSV (stdout or --detail), and
//   - a per-config summary CSV (--summary) comparing the empirical optimum against
//     the production model's choice.
// devel/calibrate_upsampfac.py reads the detail CSV and fits the heuristics.hpp cost
// constants. Links finufft only; compiled with FINUFFT_ARCH_FLAGS so the SIMD widths
// (and hence get_padding) match production.
//
// TIMING-FIDELITY CAVEAT (read before tuning from this tool): the min-over-n_runs
// execute time is noise-sensitive on a shared box. With --n_runs=6 the per-sigma
// minima were unstable enough to invert the sigma ordering (a sigma flagged 13%
// faster here measured flat under perftest's 15-run timing), and absolute times ran
// ~4x below perftest for the same config/sigma. Before baking any constant: use
// --n_runs>=15, pin to --threads=1 (multithread minima on a shared node are far
// noisier), cross-check a few cells against perftest at forced --upsampfact, and
// reconcile the absolute-time gap. This run's data was NOT trustworthy enough to
// bake; the tool is committed as reusable infrastructure for a future clean sweep.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <finufft.h>
#include <finufft/heuristics.hpp>
#include <finufft_common/constants.h>
#include <finufft_common/kernel.h>
#include <finufft_common/utils.h>

static const double MY_PI = 3.141592653589793238462643383279502884;

struct cal_options_t {
  char prec = 'd';
  int dim = 1;
  int type = 1;
  double tol = 1e-9;
  std::int64_t N = 1000000; // total modes (split across dims below)
  std::int64_t M = 2000000; // # NU points
  int threads = 0;       // 0 = auto
  int n_runs = 15;
  double sigma_step = 0.01;
  std::string summary; // append a summary row here if set
  std::string detail;  // detail CSV path ("" = stdout)
};

// High-resolution timer: min over repeated (setpts+execute), reported in ms.
struct MinTimer {
  double min_ms = std::numeric_limits<double>::max();
  std::chrono::time_point<std::chrono::steady_clock> t0;
  void start() { t0 = std::chrono::steady_clock::now(); }
  void stop() {
    const auto dt = std::chrono::steady_clock::now() - t0;
    const double ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count() / 1e6;
    min_ms = std::min(min_ms, ms);
  }
};

// Per-dim mode counts: spread N as evenly as possible over `dim` dims.
static void split_modes(std::int64_t N, int dim, std::int64_t out[3]) {
  out[0] = out[1] = out[2] = 1;
  const auto per = (std::int64_t)std::llround(std::pow((double)N, 1.0 / dim));
  for (int d = 0; d < dim; ++d) out[d] = std::max<std::int64_t>(per, 1);
}

template<typename T> void run_config(const cal_options_t &o) {
  using namespace finufft::common;
  using finufft::utils::arraywidcen;
  constexpr bool is_float = std::is_same_v<T, float>;
  constexpr double eps = std::numeric_limits<T>::epsilon();

  std::int64_t Nd[3];
  split_modes(o.N, o.dim, Nd);
  const std::int64_t Ntot = Nd[0] * Nd[1] * Nd[2];
  const std::int64_t M = o.M;

  // ---- data (mirrors perftest.cpp) ----
  std::vector<T> x(M), y(M), z(M), s(Ntot), t(Ntot), u(Ntot);
  std::vector<std::complex<T>> c(M), fk(Ntot);
  std::default_random_engine eng{42};
  std::uniform_real_distribution<T> d11(-1, 1);
  auto r = [&] {
    return d11(eng);
  };
  for (std::int64_t i = 0; i < M; ++i) {
    x[i] = T(MY_PI) * r();
    y[i] = T(MY_PI) * r();
    z[i] = T(MY_PI) * r();
  }
  if (o.type == 2)
    for (std::int64_t i = 0; i < Ntot; ++i) fk[i] = {r(), r()};
  else
    for (std::int64_t i = 0; i < M; ++i) c[i] = {r(), r()};
  if (o.type == 3)
    for (std::int64_t i = 0; i < Ntot; ++i) {
      s[i] = T(MY_PI) * r();
      t[i] = T(MY_PI) * r();
      u[i] = T(MY_PI) * r();
    }

  T *xp = x.data(), *yp = o.dim >= 2 ? y.data() : nullptr,
    *zp = o.dim == 3 ? z.data() : nullptr;
  T *sp = o.type == 3 ? s.data() : nullptr,
    *tp = o.type == 3 && o.dim >= 2 ? t.data() : nullptr,
    *up = o.type == 3 && o.dim == 3 ? u.data() : nullptr;

  // ---- model prediction (production selector) ----
  const int nthreads_eff = o.threads >= 1 ? o.threads : 1; // selector resolves >=1
  std::array<double, 3> nmodes{(double)Nd[0], (double)Nd[1], (double)Nd[2]};
  std::array<double, 3> X{}, S{}; // type-3 interval half-widths (unused for 1/2)
  double sigma_model;
  if (o.type == 3) {
    for (int d = 0; d < o.dim; ++d) {
      T w, cen;
      arraywidcen((std::int64_t)M,
                  d == 0   ? x.data()
                  : d == 1 ? y.data()
                           : z.data(),
                  &w, &cen);
      X[d] = (double)w;
      arraywidcen((std::int64_t)Ntot,
                  d == 0   ? s.data()
                  : d == 1 ? t.data()
                           : u.data(),
                  &w, &cen);
      S[d] = (double)w;
    }
    sigma_model = finufft::heuristics::best_type3<T>(
        o.tol, o.dim, nthreads_eff, (double)M, X.data(), S.data(), (double)Ntot);
  } else {
    sigma_model = finufft::heuristics::best_type12<T>(o.tol, o.dim, o.type, nthreads_eff,
                                                      nmodes.data(), (double)M)
                      .sigma;
  }
  const auto width_of = [&](double sigma) {
    return finufft::heuristics::kernel_width_at<T>(o.tol, o.dim, o.type, sigma);
  };
  // grid traffic (SIMD-padded, ISA-specific, constant-independent) and FFT fine-grid
  // product G at a given sigma/ns, so the Python fitter can reconstruct the model cost
  // exactly without re-deriving the C++ padding / fine-grid formulas.
  const auto grid_traffic_of = [&](int ns) {
    return std::max<double>(2.0 * ns + finufft::spreadinterp::get_padding<T>(2 * ns),
                            16.0);
  };
  const auto fft_G_of = [&](double sigma, int ns) {
    double G = 1.0;
    if (o.type == 3)
      for (int d = 0; d < o.dim; ++d)
        G *= finufft::heuristics::type3_fine_grid_len(sigma, X[d], S[d], ns);
    else
      for (int d = 0; d < o.dim; ++d) G *= (double)fine_grid_len(sigma, nmodes[d], ns);
    return G;
  };
  const int ns_model = width_of(sigma_model);

  // ---- sweep forced sigma ----
  const double maxN = o.type == 3 ? 1.0 : (double)*std::max_element(Nd, Nd + o.dim);
  const double sigma_min =
      analytic_upsampfac(o.tol, o.dim, o.type, eps, MAX_NSPREAD<T>, is_float, maxN);

  std::int64_t Narr[3] = {Nd[0], Nd[1], Nd[2]};
  constexpr int iflag = 1;

  std::vector<double> sigmas, times;
  for (double sigma = sigma_min; sigma <= MAX_AUTO_UPSAMPFAC + 1e-9;
       sigma += o.sigma_step) {
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.debug = 0;
    opts.upsampfac = sigma;
    opts.nthreads = o.threads;
    opts.allow_eps_too_small = 1;

    MinTimer tm;
    if constexpr (std::is_same_v<T, double>) {
      finufft_plan plan{nullptr};
      if (finufft_makeplan(o.type, o.dim, Narr, iflag, 1, o.tol, &plan, &opts)) continue;
      for (int i = 0; i < o.n_runs; ++i) {
        finufft_setpts(plan, M, xp, yp, zp, Ntot, sp, tp, up);
        tm.start();
        finufft_execute(plan, c.data(), fk.data());
        tm.stop();
      }
      finufft_destroy(plan);
    } else {
      finufftf_plan plan{nullptr};
      if (finufftf_makeplan(o.type, o.dim, Narr, iflag, 1, (float)o.tol, &plan, &opts))
        continue;
      for (int i = 0; i < o.n_runs; ++i) {
        finufftf_setpts(plan, M, xp, yp, zp, Ntot, sp, tp, up);
        tm.start();
        finufftf_execute(plan, c.data(), fk.data());
        tm.stop();
      }
      finufftf_destroy(plan);
    }
    sigmas.push_back(sigma);
    times.push_back(tm.min_ms);
  }
  if (sigmas.empty()) {
    fprintf(stderr, "no feasible sigma for this config; skipping\n");
    return;
  }

  // empirical optimum
  const auto it_opt = std::min_element(times.begin(), times.end());
  const std::size_t k_opt = (std::size_t)(it_opt - times.begin());
  const double sigma_opt = sigmas[k_opt];
  const double t_opt = times[k_opt];
  const int ns_opt = width_of(sigma_opt);

  // model time: time at the swept sigma nearest sigma_model
  std::size_t k_model = 0;
  double best_d = 1e30;
  for (std::size_t i = 0; i < sigmas.size(); ++i) {
    const double dd = std::abs(sigmas[i] - sigma_model);
    if (dd < best_d) {
      best_d = dd;
      k_model = i;
    }
  }
  const double t_model = times[k_model];

  // ---- detail CSV ----
  std::ostream *det = &std::cout;
  std::ofstream detf;
  if (!o.detail.empty()) {
    detf.open(o.detail, std::ios::app);
    det = &detf;
  }
  for (std::size_t i = 0; i < sigmas.size(); ++i) {
    const int ns = width_of(sigmas[i]);
    char buf[320];
    std::snprintf(
        buf, sizeof buf, "%c,%d,%d,%.1e,%lld,%lld,%d,%.4f,%.6f,%d,%d,%d,%.1f,%.6e\n",
        o.prec, o.dim, o.type, o.tol, (long long)Ntot, (long long)M, o.threads, sigmas[i],
        times[i], (int)(i == k_opt), ns, finufft::kernel::max_nc_given_ns(ns),
        grid_traffic_of(ns), fft_G_of(sigmas[i], ns));
    (*det) << buf;
  }

  // ---- summary CSV ----
  if (!o.summary.empty()) {
    std::ofstream sf(o.summary, std::ios::app);
    char buf[320];
    std::snprintf(
        buf, sizeof buf, "%c,%d,%d,%.1e,%lld,%lld,%d,%.4f,%d,%d,%.4f,%d,%.6f,%.6f,%.4f\n",
        o.prec, o.dim, o.type, o.tol, (long long)Ntot, (long long)M, o.threads, sigma_opt,
        ns_opt, finufft::kernel::max_nc_given_ns(ns_opt), sigma_model, ns_model, t_opt,
        t_model, t_model > 0 ? t_model / t_opt : 0.0);
    sf << buf;
  }
}

int main(int argc, char *argv[]) {
  cal_options_t o;
  static struct option lo[] = {
      {"prec", required_argument, 0, 0},       {"dim", required_argument, 0, 0},
      {"type", required_argument, 0, 0},       {"tol", required_argument, 0, 0},
      {"N", required_argument, 0, 0},          {"M", required_argument, 0, 0},
      {"threads", required_argument, 0, 0},    {"n_runs", required_argument, 0, 0},
      {"sigma_step", required_argument, 0, 0}, {"summary", required_argument, 0, 0},
      {"detail", required_argument, 0, 0},     {0, 0, 0, 0}};
  int idx = 0, c;
  while ((c = getopt_long(argc, argv, "", lo, &idx)) != -1) {
    if (c != 0) continue;
    const std::string k = lo[idx].name, v = optarg;
    if (k == "prec")
      o.prec = v[0];
    else if (k == "dim")
      o.dim = std::stoi(v);
    else if (k == "type")
      o.type = std::stoi(v);
    else if (k == "tol")
      o.tol = std::stod(v);
    else if (k == "N")
      o.N = (std::int64_t)std::stod(v);
    else if (k == "M")
      o.M = (std::int64_t)std::stod(v);
    else if (k == "threads")
      o.threads = std::stoi(v);
    else if (k == "n_runs")
      o.n_runs = std::stoi(v);
    else if (k == "sigma_step")
      o.sigma_step = std::stod(v);
    else if (k == "summary")
      o.summary = v;
    else if (k == "detail")
      o.detail = v;
  }
  if (o.prec == 'f')
    run_config<float>(o);
  else
    run_config<double>(o);
  return 0;
}

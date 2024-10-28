#include <cstdint>
#include <getopt.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

#include <chrono>
#include <finufft.h>
#ifndef FINUFFT_USE_DUCC0
#include <fftw3.h>
#endif

std::string get_or(const std::unordered_map<std::string, std::string> &m,
                   const std::string &key, const std::string &default_value) {
  auto it = m.find(key);
  if (it == m.end()) {
    return default_value;
  }
  return it->second;
}

struct test_options_t {
  char prec;
  int type;
  int n_runs;
  std::int64_t N[3];
  int M;
  int ntransf;
  int kerevalmethod;
  int sort;
  int threads;
  double tol;
  double upsampfact;
  double bandwidth;
  int debug;

  test_options_t(int argc, char *argv[]) {
    std::unordered_map<std::string, std::string> options_map;

    while (true) {
      int option_index = 0;

      // clang-format off
            static struct option long_options[] {
                {"prec", required_argument, 0, 0},
                {"type", required_argument, 0, 0},
                {"n_runs", required_argument, 0, 0},
                {"N1", required_argument, 0, 0},
                {"N2", required_argument, 0, 0},
                {"N3", required_argument, 0, 0},
                {"M", required_argument, 0, 0},
                {"ntransf", required_argument, 0, 0},
                {"tol", required_argument, 0, 0},
                {"kerevalmethod", required_argument, 0, 0},
                {"threads", required_argument, 0, 0},
                {"sort", required_argument, 0, 0},
                {"upsampfact", required_argument, 0, 0},
                {"debug", required_argument, 0, 0},
                {"bandwidth", required_argument, 0, 0},
                {0, 0, 0, 0},
            };
      // clang-format on

      int c = getopt_long(argc, argv, "", long_options, &option_index);
      if (c == -1) break;

      switch (c) {
      case 0:
        options_map[long_options[option_index].name] = optarg;
        break;

      default:
        break;
      }
    }

    prec          = get_or(options_map, "prec", "f")[0];
    type          = std::stoi(get_or(options_map, "type", "1"));
    n_runs        = std::stoi(get_or(options_map, "n_runs", "10"));
    N[0]          = std::stof(get_or(options_map, "N1", "1E6"));
    N[1]          = std::stof(get_or(options_map, "N2", "1"));
    N[2]          = std::stof(get_or(options_map, "N3", "1"));
    M             = std::stof(get_or(options_map, "M", "2E6"));
    ntransf       = std::stoi(get_or(options_map, "ntransf", "1"));
    kerevalmethod = std::stoi(get_or(options_map, "kerevalmethod", "1"));
    sort          = std::stoi(get_or(options_map, "sort", "1"));
    threads       = std::stoi(get_or(options_map, "threads", "0"));
    tol           = std::stof(get_or(options_map, "tol", "1E-5"));
    upsampfact    = std::stof(get_or(options_map, "upsampfact", "0"));
    debug         = std::stoi(get_or(options_map, "debug", "0"));
    bandwidth     = std::stof(get_or(options_map, "bandwidth", "1"));
  }

  friend std::ostream &operator<<(std::ostream &outs, const test_options_t &opts) {
    return outs << "# prec = " << opts.prec << "\n"
                << "# type = " << opts.type << "\n"
                << "# n_runs = " << opts.n_runs << "\n"
                << "# N1 = " << opts.N[0] << "\n"
                << "# N2 = " << opts.N[1] << "\n"
                << "# N3 = " << opts.N[2] << "\n"
                << "# M = " << opts.M << "\n"
                << "# ntransf = " << opts.ntransf << "\n"
                << "# kerevalmethod = " << opts.kerevalmethod << "\n"
                << "# sort = " << opts.sort << "\n"
                << "# threads = " << opts.threads << "\n"
                << "# tol = " << opts.tol << "\n"
                << "# upsampfact = " << opts.upsampfact << "\n"
                << "# debug = " << opts.debug << "\n"
                << "# bandwidth = " << opts.bandwidth << "\n";
  }
};

struct Timer {

  void start() { start_.emplace_back(std::chrono::steady_clock::now()); }

  void stop() { stop_.emplace_back(std::chrono::steady_clock::now()); }

  float mean() { return this->tot() / start_.size(); }

  float min() {
    float min_dt = std::numeric_limits<float>::max();
    for (size_t i = 0; i < start_.size(); ++i) {
      auto dt = float(
          std::chrono::duration_cast<std::chrono::milliseconds>(stop_[i] - start_[i])
              .count());
      if (dt < min_dt) {
        min_dt = dt;
      }
    }
    return min_dt;
  }

  float std() {
    float avg  = this->mean();
    double var = 0.0;
    for (size_t i = 0; i < start_.size(); ++i) {
      auto dt = float(
          std::chrono::duration_cast<std::chrono::milliseconds>(stop_[i] - start_[i])
              .count());
      var += (dt - avg) * (dt - avg);
    }
    var /= float(start_.size());
    return std::sqrt(var);
  }

  float tot() {
    float dt_tot = 0.0;
    for (size_t i = 0; i < start_.size(); ++i) {
      auto dt = float(
          std::chrono::duration_cast<std::chrono::milliseconds>(stop_[i] - start_[i])
              .count());
      dt_tot += dt;
    }
    return dt_tot;
  }

  int count() { return start_.size(); }

  std::vector<std::chrono::time_point<std::chrono::steady_clock>> start_;
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> stop_;
};

template<class F, class... Args> inline void timeit(F f, Timer &timer, Args... args) {
  timer.start();
  f(args...);
  timer.stop();
}

template<typename T> void run_test(test_options_t &test_opts) {
  std::cout << test_opts;
  const int ntransf   = test_opts.ntransf;
  const int64_t M     = test_opts.M;
  const long N        = test_opts.N[0] * test_opts.N[1] * test_opts.N[2];
  const int type      = test_opts.type;
  constexpr int iflag = 1;

  std::vector<T> x(M * ntransf), y(M * ntransf), z(M * ntransf);
  std::vector<T> s(N * ntransf), t(N * ntransf), u(N * ntransf);
  std::vector<std::complex<T>> c(M * ntransf), fk(N * ntransf);

  std::default_random_engine eng{42};
  std::uniform_real_distribution<T> dist11(-1, 1);
  auto randm11 = [&eng, &dist11]() {
    return dist11(eng);
  };

  // Making data
  for (int64_t i = 0; i < M; i++) {
    x[i] = M_PI * randm11(); // x in [-pi,pi)
    y[i] = M_PI * randm11();
    z[i] = M_PI * randm11();
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
      s[i] = M_PI * randm11() * test_opts.bandwidth;
      t[i] = M_PI * randm11() * test_opts.bandwidth;
      u[i] = M_PI * randm11() * test_opts.bandwidth;
    }

  } else {
    std::cerr << "Invalid type " << type << " supplied\n";
    return;
  }

  finufft_opts opts;
  int dim = 0;
  for (int i = 0; i < 3; ++i) dim = test_opts.N[i] > 1 ? i + 1 : dim;

  finufft_default_opts(&opts);
  opts.debug              = test_opts.debug;
  opts.upsampfac          = test_opts.upsampfact;
  opts.spread_kerevalmeth = test_opts.kerevalmethod;
  opts.spread_sort        = test_opts.sort;
  opts.nthreads           = test_opts.threads;

  Timer makeplan_timer, setpts_timer, execute_timer, amortized_timer;
  amortized_timer.start();

  T *x_p = dim >= 1 ? x.data() : nullptr;
  T *y_p = dim >= 2 ? y.data() : nullptr;
  T *z_p = dim == 3 ? z.data() : nullptr;
  T *s_p = type == 3 && dim >= 1 ? s.data() : nullptr;
  T *t_p = type == 3 && dim >= 2 ? t.data() : nullptr;
  T *u_p = type == 3 && dim == 3 ? u.data() : nullptr;
  if constexpr (std::is_same_v<T, double>) {
    finufft_plan_s *plan{nullptr};
    timeit(finufft_makeplan, makeplan_timer, test_opts.type, dim, test_opts.N, iflag,
           ntransf, test_opts.tol, &plan, &opts);
    for (int i = 0; i < test_opts.n_runs; ++i) {
      timeit(finufft_setpts, setpts_timer, plan, M, x_p, y_p, z_p, N, s_p, t_p, u_p);
      timeit(finufft_execute, execute_timer, plan, c.data(), fk.data());
    }
    finufft_destroy(plan);
  }

  if constexpr (std::is_same_v<T, float>) {
    finufftf_plan_s *plan{nullptr};
    timeit(finufftf_makeplan, makeplan_timer, test_opts.type, dim, test_opts.N, iflag,
           ntransf, test_opts.tol, &plan, &opts);
    for (int i = 0; i < test_opts.n_runs; ++i) {
      timeit(finufftf_setpts, setpts_timer, plan, M, x_p, y_p, z_p, N, s_p, t_p, u_p);
      timeit(finufftf_execute, execute_timer, plan, c.data(), fk.data());
    }
    finufftf_destroy(plan);
  }
  amortized_timer.stop();

  const int64_t nupts_tot = M * test_opts.n_runs * ntransf;

  printf("event,count,tot(ms),mean(ms),min(ms),std(ms),nupts/s,ns/nupt\n");
  printf("makeplan,%d,%f,%f,%f,%f,0.0,0.0\n", makeplan_timer.count(),
         makeplan_timer.tot(), makeplan_timer.mean(), makeplan_timer.min(),
         makeplan_timer.std());
  printf("setpts,%d,%f,%f,%f,%f,%g,%f\n", test_opts.n_runs, setpts_timer.tot(),
         setpts_timer.mean(), setpts_timer.min(), setpts_timer.std(),
         nupts_tot * 1000 / setpts_timer.tot(), setpts_timer.tot() * 1E6 / nupts_tot);
  printf("execute,%d,%f,%f,%f,%f,%g,%f\n", test_opts.n_runs, execute_timer.tot(),
         execute_timer.mean(), execute_timer.min(), execute_timer.std(),
         nupts_tot * 1000 / execute_timer.tot(), execute_timer.tot() * 1E6 / nupts_tot);
  printf("amortized,%d,%f,%f,%f,%f,%g,%f\n", 1, amortized_timer.tot(),
         amortized_timer.mean(), amortized_timer.min(), amortized_timer.std(),
         nupts_tot * 1000 / amortized_timer.tot(),
         amortized_timer.tot() * 1E6 / nupts_tot);
}

int main(int argc, char *argv[]) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    test_options_t default_opts(0, nullptr);
    // clang-format off
        std::cout << "Valid options:\n"
                     "    --prec <char>\n"
                     "           float or double precision. i.e. 'f' or 'd'\n"
                     "           default: " << default_opts.prec << "\n" <<
                     "    --type <int>\n"
                     "           type of transform. 1, 2 or 3\n"
                     "           default: " << default_opts.type << "\n" <<
                     "    --n_runs <int>\n"
                     "           number of runs to average performance over\n"
                     "           default: " << default_opts.n_runs << "\n" <<
                     "    --N1 <int>\n"
                     "           number of modes in first dimension. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.N[0] << "\n" <<
                     "    --N2 <int>\n"
                     "           number of modes in second dimension. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.N[1] << "\n" <<
                     "    --N3 <int>\n"
                     "           number of modes in third dimension. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.N[2] << "\n" <<
                     "    --M <int>\n"
                     "           number of non-uniform points. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.M << "\n" <<
                     "    --ntransf <int>\n"
                     "           number of transforms to do simultaneously\n"
                     "           default: " << default_opts.ntransf << "\n" <<
                     "    --tol <float>\n"
                     "           NUFFT tolerance. Scientific notation accepted (i.e. 1.2E-7)\n"
                     "           default: " << default_opts.tol << "\n" <<
                     "    --kerevalmeth <int>\n"
                     "           kernel evaluation method\n"
                     "               0: Exponential of square root\n"
                     "               1: Horner evaluation\n"
                     "           default: " << default_opts.kerevalmethod << "\n" <<
                     "    --sort: <int>\n"
                     "           sort strategy\n"
                     "               0: do not sort the points\n"
                     "               1: sort the points\n"
                     "           default: " << default_opts.sort << "\n"
                     "    --upsampfact: <float>\n"
                     "           sort strategy\n"
                     "               0: do not sort the points\n"
                     "               1: sort the points\n"
                     "           default: " << default_opts.upsampfact << "\n"
                     "    --debug: <int>\n"
                     "           debug prints\n"
                     "               0: no debug\n"
                     "               1: standard\n"
                     "               2: verbose\n"
                     "           default: " << default_opts.debug << "\n"
                     "    --bandwidth: <float>\n"
                     "           bandwidth for type 3\n"
                     "           default: " << default_opts.bandwidth << "\n";
    // clang-format on
    return 0;
  }
  test_options_t opts(argc, argv);
#ifndef FINUFFT_USE_DUCC0
  fftw_forget_wisdom();
  fftwf_forget_wisdom();
#endif
  if (opts.prec == 'f')
    run_test<float>(opts);
  else if (opts.prec == 'd')
    run_test<double>(opts);

  return 0;
}

#include <cstdint>
#include <getopt.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>

#include <cufinufft.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <type_traits>

#include "../randunif.h"

// Public API and a local PI, as perftest.cpp does: the docs page builds this
// harness against every release it plots, and the internal plan class changes
// shape between them.
static const double PI = 3.141592653589793238462643383279502884;

template<typename T>
using cuplan =
    std::conditional_t<std::is_same_v<T, float>, cufinufftf_plan, cufinufft_plan>;
template<typename T>
using cucomplex =
    std::conditional_t<std::is_same_v<T, float>, cuFloatComplex, cuDoubleComplex>;

void check(int ier, const char *what) {
  if (ier) {
    std::cerr << what << " failed with ier " << ier << "\n";
    std::exit(1);
  }
}

// opts.debug arrived in 2.4, and this builds against 2.2 as well.
template<class O>
auto set_debug(O &opts, int debug, int) -> decltype(opts.debug = debug, void()) {
  opts.debug = debug;
}
template<class O> void set_debug(O &, int, long) {}

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
  int32_t N[3];
  int M;
  int ntransf;
  int kerevalmethod;
  int method;
  int sort;
  double tol;
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
          {"method", required_argument, 0, 0},
          {"kerevalmethod", required_argument, 0, 0},
          {"sort", required_argument, 0, 0},
          {"debug", required_argument, 0, 0},
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
    method        = std::stoi(get_or(options_map, "method", "1"));
    kerevalmethod = std::stoi(get_or(options_map, "kerevalmethod", "1"));
    sort          = std::stoi(get_or(options_map, "sort", "1"));
    tol           = std::stof(get_or(options_map, "tol", "1E-5"));
    debug = std::stof(get_or(options_map, "debug", "0"));
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
                << "# method = " << opts.method << "\n"
                << "# kerevalmethod = " << opts.kerevalmethod << "\n"
                << "# sort = " << opts.sort << "\n"
                << "# tol = " << opts.tol << "\n"
                << "# debug = " << opts.debug << "\n";
  }
};

struct CudaTimer {
  CudaTimer() {}

  ~CudaTimer() {
    for (auto &event : start_) cudaEventDestroy(event);
    for (auto &event : stop_) cudaEventDestroy(event);
  }

  void start() {
    start_.push_back(cudaEvent_t{});
    stop_.push_back(cudaEvent_t{});

    cudaEventCreate(&start_.back());
    cudaEventCreate(&stop_.back());

    cudaEventRecord(start_.back());
  }

  void stop() { cudaEventRecord(stop_.back()); }

  void sync() {
    for (auto &event : stop_) cudaEventSynchronize(event);
  }

  float dt(size_t i) {
    float dt_i;
    cudaEventElapsedTime(&dt_i, start_[i], stop_[i]);
    return dt_i;
  }

  float mean() { return this->tot() / start_.size(); }

  // The reported metric, as in perftest.cpp: interference only ever makes a
  // sample slower, so the fastest is the one least polluted by it.
  float min() {
    float min_dt = std::numeric_limits<float>::max();
    for (size_t i = 0; i < start_.size(); ++i) min_dt = std::min(min_dt, dt(i));
    return min_dt;
  }

  float std() {
    float avg = this->mean();
    double var = 0.0;
    for (size_t i = 0; i < start_.size(); ++i) var += (dt(i) - avg) * (dt(i) - avg);
    var /= float(start_.size());
    return sqrt(var);
  }

  float tot() {
    float dt_tot = 0.;
    for (size_t i = 0; i < start_.size(); ++i) dt_tot += dt(i);
    return dt_tot;
  }

  int count() { return start_.size(); }

  std::vector<cudaEvent_t> start_;
  std::vector<cudaEvent_t> stop_;
};

template<class F, class... Args>
inline void timeit(F f, CudaTimer &timer, Args &&...args) {
  timer.start();
  f(std::forward<Args>(args)...);
  timer.stop();
}
template<typename T>
void makeplan(int type, int dim, const int64_t *nmodes, int iflag, int ntransf, T tol,
              cufinufft_opts *opts, cuplan<T> *plan) {
  if constexpr (std::is_same_v<T, float>)
    check(cufinufftf_makeplan(type, dim, nmodes, iflag, ntransf, tol, plan, opts),
          "makeplan");
  else
    check(cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, plan, opts),
          "makeplan");
}

template<typename T>
void setpts(cuplan<T> plan, int64_t M, T *x, T *y, T *z, int N, T *s, T *t, T *u) {
  if constexpr (std::is_same_v<T, float>)
    check(cufinufftf_setpts(plan, M, x, y, z, N, s, t, u), "setpts");
  else
    check(cufinufft_setpts(plan, M, x, y, z, N, s, t, u), "setpts");
}

template<typename T> void execute(cuplan<T> plan, cucomplex<T> *c, cucomplex<T> *fk) {
  if constexpr (std::is_same_v<T, float>)
    check(cufinufftf_execute(plan, c, fk), "execute");
  else
    check(cufinufft_execute(plan, c, fk), "execute");
}

template<typename T> void destroy(cuplan<T> plan) {
  if constexpr (std::is_same_v<T, float>)
    check(cufinufftf_destroy(plan), "destroy");
  else
    check(cufinufft_destroy(plan), "destroy");
}
template<typename T> void run_test(test_options_t &test_opts) {
  std::cout << test_opts;
  const int ntransf   = test_opts.ntransf;
  const int64_t M     = test_opts.M;
  const int N         = test_opts.N[0] * test_opts.N[1] * test_opts.N[2];
  const int type      = test_opts.type;
  constexpr int iflag = 1;

  // Target frequencies exist for type 3 only, where N counts them.
  const int64_t NT = type == 3 ? int64_t(N) * ntransf : 0;

  // Same as the device vectors: unfilled storage, one parallel pass writes it.
  using perftest_rand::noinit_alloc;
  static_assert(sizeof(thrust::complex<T>) == 2 * sizeof(T),
                "the fill below writes a complex array through a scalar pointer");
  thrust::host_vector<T, noinit_alloc<T>> x(M * ntransf), y(M * ntransf), z(M * ntransf);
  thrust::host_vector<T, noinit_alloc<T>> s(NT), t(NT), u(NT);
  thrust::host_vector<thrust::complex<T>, noinit_alloc<thrust::complex<T>>> c(
      M * ntransf),
      fk(N * ntransf);

  thrust::device_vector<T> d_x(M * ntransf), d_y(M * ntransf), d_z(M * ntransf);
  thrust::device_vector<T> d_s(NT), d_t(NT), d_u(NT);
  thrust::device_vector<thrust::complex<T>> d_c(M * ntransf), d_fk(N * ntransf);

  // Making data: the values depend on the index alone, so both arms of a
  // comparison see the same points on every invocation.
  perftest_rand::fill(x.data(), M, perftest_rand::X, T(PI), T(0));
  perftest_rand::fill(y.data(), M, perftest_rand::Y, T(PI), T(0));
  perftest_rand::fill(z.data(), M, perftest_rand::Z, T(PI), T(0));
  for (int64_t i = M; i < M * ntransf; ++i) {
    int64_t j = i % M;
    x[i]      = x[j];
    y[i]      = y[j];
    z[i]      = z[j];
  }

  // std::complex<T> is [re, im] in memory, so one fill covers both parts.
  if (type == 1) {
    perftest_rand::fill(reinterpret_cast<T *>(c.data()), 2 * M * ntransf,
                        perftest_rand::C, T(1), T(0));

  } else if (type == 2) {
    perftest_rand::fill(reinterpret_cast<T *>(fk.data()), 2 * N * ntransf,
                        perftest_rand::FK, T(1), T(0));

  } else if (type == 3) {
    perftest_rand::fill(reinterpret_cast<T *>(c.data()), 2 * M * ntransf,
                        perftest_rand::C, T(1), T(0));
    // Frequency range of a type 1 of the same size, as finufft_test picks it:
    // S_d = N_d/2, offset so the range is not centred on zero. The
    // space-bandwidth product then tracks N.
    const T S1 = T(0.5) * test_opts.N[0];
    const T S2 = T(0.5) * test_opts.N[1];
    const T S3 = T(0.5) * test_opts.N[2];
    perftest_rand::fill(s.data(), NT, perftest_rand::S, S1, T(1.7));
    perftest_rand::fill(t.data(), NT, perftest_rand::T, S2, T(-0.5));
    perftest_rand::fill(u.data(), NT, perftest_rand::U, S3, T(0.9));
  } else {
    std::cerr << "Invalid type " << type << " supplied\n";
    return;
  }

  cufinufft_opts opts;
  int dim = 0;
  for (int i = 0; i < 3; ++i) dim = test_opts.N[i] > 1 ? i + 1 : dim;

  cufinufft_default_opts(&opts);
  opts.gpu_method      = test_opts.method;
  opts.gpu_sort        = test_opts.sort;
  opts.gpu_kerevalmeth = test_opts.kerevalmethod;
  set_debug(opts, test_opts.debug, 0);

  // The public API takes the mode counts as int64.
  const int64_t nmodes[3]{test_opts.N[0], test_opts.N[1], test_opts.N[2]};

  cuplan<T> dplan;
  CudaTimer h2d_timer, makeplan_timer, setpts_timer, execute_timer, d2h_timer,
      amortized_timer;
  T *d_x_p = dim >= 1 ? d_x.data().get() : nullptr;
  T *d_y_p = dim >= 2 ? d_y.data().get() : nullptr;
  T *d_z_p = dim == 3 ? d_z.data().get() : nullptr;
  T *d_s_p = type == 3 && dim >= 1 ? d_s.data().get() : nullptr;
  T *d_t_p = type == 3 && dim >= 2 ? d_t.data().get() : nullptr;
  T *d_u_p = type == 3 && dim == 3 ? d_u.data().get() : nullptr;
  cucomplex<T> *d_c_p = (cucomplex<T> *)d_c.data().get();
  cucomplex<T> *d_fk_p = (cucomplex<T> *)d_fk.data().get();

  // One untimed transform of this exact shape, outside every timer. The first
  // plan of a size compiles kernels that later plans read back from the CUDA
  // cache, so whichever binary ran first would report that as its own cost.
  {
    d_x = x, d_y = y, d_z = z;
    if (type == 1 || type == 3) d_c = c;
    if (type == 2) d_fk = fk;
    if (type == 3) d_s = s, d_t = t, d_u = u;

    cuplan<T> warm;
    makeplan<T>(test_opts.type, dim, nmodes, iflag, ntransf, T(test_opts.tol), &opts,
                &warm);
    setpts<T>(warm, M, d_x_p, d_y_p, d_z_p, type == 3 ? N : 0, d_s_p, d_t_p, d_u_p);
    execute<T>(warm, d_c_p, d_fk_p);
    cudaDeviceSynchronize();
    destroy<T>(warm);
  }

  {
    amortized_timer.start();
    // Every stage repeats, so one disturbed sample costs a sample rather than
    // the whole case. The plan is rebuilt per run because makeplan allocates
    // the cuFFT workspace, the widest tail of the five stages.
    for (int i = 0; i < test_opts.n_runs; ++i) {
      h2d_timer.start();
      d_x = x, d_y = y, d_z = z;
      if (type == 1 || type == 3) d_c = c;
      if (type == 2) d_fk = fk;
      if (type == 3) d_s = s, d_t = t, d_u = u;
      h2d_timer.stop();

      timeit(makeplan<T>, makeplan_timer, test_opts.type, dim, nmodes, iflag, ntransf,
             T(test_opts.tol), &opts, &dplan);
      timeit(
          [&] {
            setpts<T>(dplan, M, d_x_p, d_y_p, d_z_p, type == 3 ? N : 0, d_s_p, d_t_p,
                      d_u_p);
          },
          setpts_timer);
      timeit([&] { execute<T>(dplan, d_c_p, d_fk_p); }, execute_timer);

      d2h_timer.start();
      if (type == 1 || type == 3) fk = d_fk;
      if (type == 2) c = d_c;
      d2h_timer.stop();

      destroy<T>(dplan);
    }

    amortized_timer.stop();

    h2d_timer.sync();
    makeplan_timer.sync();
    setpts_timer.sync();
    execute_timer.sync();
    d2h_timer.sync();
    amortized_timer.sync();
  }

  const int64_t nupts_tot = M * test_opts.n_runs * ntransf;

  printf("event,count,tot(ms),mean(ms),min(ms),std(ms),nupts/s,ns/nupt\n");
  printf("host_to_device,%d,%f,%f,%f,%f,0.0,0.0\n", h2d_timer.count(), h2d_timer.tot(),
         h2d_timer.mean(), h2d_timer.min(), h2d_timer.std());
  printf("makeplan,%d,%f,%f,%f,%f,0.0,0.0\n", makeplan_timer.count(),
         makeplan_timer.tot(), makeplan_timer.mean(), makeplan_timer.min(),
         makeplan_timer.std());
  printf("setpts,%d,%f,%f,%f,%f,%g,%f\n", test_opts.n_runs, setpts_timer.tot(),
         setpts_timer.mean(), setpts_timer.min(), setpts_timer.std(),
         nupts_tot * 1000 / setpts_timer.tot(), setpts_timer.tot() * 1E6 / nupts_tot);
  printf("execute,%d,%f,%f,%f,%f,%g,%f\n", test_opts.n_runs, execute_timer.tot(),
         execute_timer.mean(), execute_timer.min(), execute_timer.std(),
         nupts_tot * 1000 / execute_timer.tot(), execute_timer.tot() * 1E6 / nupts_tot);
  printf("device_to_host,%d,%f,%f,%f,%f,0.0,0.0\n", d2h_timer.count(), d2h_timer.tot(),
         d2h_timer.mean(), d2h_timer.min(), d2h_timer.std());
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
                 "    --method <int>\n"
                 "           NUFFT method\n"
                 "               1: nupts driven\n"
                 "               2: sub-problem\n"
                 "               3: output-driven subproblem (experimental)\n"
                 "           Note that not all methods are compatible with all dim/type combinations\n"
                 "           default: " << default_opts.method << "\n" <<
                 "    --kerevalmeth <int>\n"
                 "           kernel evaluation method\n"
                 "               0: Exponential of square root\n"
                 "               1: Horner evaluation\n"
                 "           default: " << default_opts.kerevalmethod << "\n" <<
                 "    --sort: <int>\n"
                 "           sort strategy\n"
                 "               0: do not sort the points\n"
                 "               1: sort the points\n"
                 "           default: " << default_opts.sort << "\n" <<
                 "    --debug: <int>\n"
                 "           print cufinufft debug info\n"
                 "               0: do not print debug info\n"
                 "               1: print debug info\n"
                 "           default: " << default_opts.debug << "\n";
    // clang-format on
    return 0;
  }
  test_options_t opts(argc, argv);

  if (opts.prec == 'f')
    run_test<float>(opts);
  else if (opts.prec == 'd')
    run_test<double>(opts);

  return 0;
}

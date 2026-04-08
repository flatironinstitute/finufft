// Empirical sigma_min data generator for the upsampfac warning estimator.
// Outputs CSV: prec,type,dim,tol,sigma_empirical,sigma_analytical,b,delta,N
// See devel/find_sigma_bound.py for model fitting and validation.
// Brodovič; N column added by Barbone 4/3/26.

#include <cmath>
#include <finufft.h>
#include <finufft/test_defs.hpp>
#include <finufft_common/common.h>
#include <getopt.h>
#include <iostream>
#include <mutex>
#include <polynomial_regression.hpp>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utils/dirft1d.hpp>
#include <utils/norms.hpp>

using namespace std;

set<string> get_set(const unordered_map<string, string> &m, const string key) {
  auto it = m.find(key);
  if (it == m.end()) return {};
  set<string> res = {};
  stringstream ss(it->second);
  string tok;
  while (getline(ss, tok, ',')) res.insert(tok);
  return res;
}

struct train_options_t {
  set<int> dim;
  set<int> type;
  set<char> prec;
  int Ntotal;
  int M;
  double sigma_prec;
  int ntol;
  int nthreads;
  unordered_map<string, string> options_map;

  train_options_t(int argc, char *argv[]) {
    options_map["dim"]        = "1";
    options_map["type"]       = "1,2,3";
    options_map["prec"]       = "f,d";
    options_map["N"]          = "300";
    options_map["M"]          = "1000";
    options_map["sigma-prec"] = "1e-5";
    options_map["ntol"]       = "100";
    options_map["nthreads"]   = "0";
    // clang-format off
        static struct option long_options[] {
            {"dim", required_argument, 0, 0},
            {"type", required_argument, 0, 0},
            {"prec", required_argument, 0, 0},
            {"sigma-prec", required_argument, 0, 0},
            {"N", required_argument, 0, 0},
            {"M", required_argument, 0, 0},
            {"ntol", required_argument, 0, 0},
            {"nthreads", required_argument, 0, 0}
        };
    // clang-format on
    int option_index = 0;
    while (true) {
      if (getopt_long(argc, argv, "", long_options, &option_index) == -1) break;
      options_map[long_options[option_index].name] = optarg;
    }
    for (auto &t : get_set(options_map, "type")) type.insert(stoi(t));
    for (auto &t : get_set(options_map, "dim")) dim.insert(stoi(t));
    for (auto &t : get_set(options_map, "prec")) prec.insert(t[0]);

    Ntotal     = stoi(options_map["N"]);
    M          = stoi(options_map["M"]);
    sigma_prec = stod(options_map["sigma-prec"]);
    ntol       = stoi(options_map["ntol"]);
    nthreads   = stoi(options_map["nthreads"]);
  }
};

template<typename T> std::vector<T> log_scale(T low, T hi, int n) {
  vector<T> res;
  T base = log(low);
  T step = (log(hi) - base) / (n - 1);

  for (int i = 0; i < n; i++) {
    res.push_back(exp(base + i * step));
  }
  return res;
}

// Inverse of theoretical_kernel_ns. Computes lowest sigma (upsampfacs) that
// keeps ns below maxns.
double lowest_sigma(double tol, int type, int dim, int maxns, int kerformula) {
  double tolfac;
  double nsoff;
  if (kerformula == 1) {
    tolfac = 1;
    nsoff  = 0;
  } else {
    tolfac = 0.18 * pow(1.4, dim - 1);
    if (type == 3) tolfac *= 1.4;
    nsoff = 1;
  }
  double a = std::log(tolfac / tol);
  double b = pow(a / ((maxns - nsoff) * finufft::common::PI), 2);
  return 1 / (1 - b);
}

template<typename T>
void run_combo(int n_dims, int type, const train_options_t &cmd_opts,
               std::mutex &output_mutex) {
  constexpr double sigma_max = 2.0;
  constexpr int iflag        = 1;

  // Private arrays per thread
  vector<T> x(cmd_opts.M), y(cmd_opts.M), z(cmd_opts.M);
  vector<T> s(cmd_opts.Ntotal), tv(cmd_opts.Ntotal), u(cmd_opts.Ntotal);
  vector<complex<T>> c_est(cmd_opts.M), c_targ(cmd_opts.M), c_input(cmd_opts.M);
  vector<complex<T>> f_est(cmd_opts.Ntotal), f_targ(cmd_opts.Ntotal),
      f_input(cmd_opts.Ntotal);

  // Deterministic per-combo seed (based on dim/type)
  mt19937 gen(12345 + n_dims * 10 + type);
  uniform_real_distribution<T> udis(-1.0, 1.0);
  auto random11 = [&]() {
    return udis(gen);
  };
  auto crandom11 = [&]() {
    return udis(gen) + static_cast<complex<T>>(1.0i) * udis(gen);
  };

  for (int i = 0; i < cmd_opts.M; i++) {
    x[i]       = PI * random11();
    y[i]       = PI * random11();
    z[i]       = PI * random11();
    c_input[i] = crandom11();
  }
  dirft1d1(cmd_opts.M, x, c_input, iflag, cmd_opts.Ntotal, f_input);
  T S = (T)cmd_opts.Ntotal / 4;
  for (int i = 0; i < cmd_opts.Ntotal; i++) {
    s[i]  = S * random11();
    tv[i] = S * random11();
    u[i]  = S * random11();
  }

  switch (type) {
  case 1:
    dirft1d1(cmd_opts.M, x, c_input, iflag, cmd_opts.Ntotal, f_targ);
    break;
  case 2:
    dirft1d2(cmd_opts.M, x, c_targ, iflag, cmd_opts.Ntotal, f_input);
    break;
  case 3:
    dirft1d3(cmd_opts.M, x, c_input, iflag, cmd_opts.Ntotal, s, f_targ);
    break;
  }

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.allow_eps_too_small = 1;
  opts.showwarn            = 0;
  opts.nthreads            = cmd_opts.nthreads;
  int64_t N[3]             = {cmd_opts.Ntotal, 1, 1};

  int maxns = finufft::common::MAX_NSPREAD;
  if constexpr (is_same_v<T, float>) maxns = 8;

  double sigma_upper = sigma_max;
  vector<double> tol_range, sigmas;
  int lowest_tol_idx = 0, upper_tol_idx = -1;

  double exp_tol   = static_cast<double>(numeric_limits<T>::epsilon());
  double exp_limit = log(min(1e-2, exp_tol * 1e6));
  exp_tol          = log(exp_tol);
  double max_step  = (exp_limit - exp_tol) / cmd_opts.ntol;

  for (int idx = 0; exp_tol < exp_limit; idx++) {
    double tol         = exp(exp_tol);
    double comp_sigma  = lowest_sigma(tol, type, n_dims, maxns, 8);
    double sigma_lower = min(sigma_upper, comp_sigma);
    tol_range.push_back(tol);

    while (sigma_upper - sigma_lower > cmd_opts.sigma_prec) {
      opts.upsampfac = (sigma_upper + sigma_lower) / 2;
      float err;
      if constexpr (is_same_v<T, double>) {
        finufft_plan_s *plan{nullptr};
        finufft_makeplan(type, n_dims, N, iflag, 1, tol, &plan, &opts);
        finufft_setpts(plan, cmd_opts.M, x.data(), y.data(), z.data(), cmd_opts.Ntotal,
                       s.data(), tv.data(), u.data());
        if (type == 1 || type == 3) {
          finufft_execute(plan, c_input.data(), f_est.data());
          err = relerrtwonorm(cmd_opts.Ntotal, f_targ.data(), f_est.data());
        } else {
          finufft_execute(plan, c_est.data(), f_input.data());
          err = relerrtwonorm(cmd_opts.M, c_targ.data(), c_est.data());
        }
        finufft_destroy(plan);
      } else {
        finufftf_plan_s *plan{nullptr};
        finufftf_makeplan(type, n_dims, N, iflag, 1, tol, &plan, &opts);
        finufftf_setpts(plan, cmd_opts.M, x.data(), y.data(), z.data(), cmd_opts.Ntotal,
                        s.data(), tv.data(), u.data());
        if (type == 1 || type == 3) {
          finufftf_execute(plan, c_input.data(), f_est.data());
          err = relerrtwonorm(cmd_opts.Ntotal, f_targ.data(), f_est.data());
        } else {
          finufftf_execute(plan, c_est.data(), f_input.data());
          err = relerrtwonorm(cmd_opts.M, c_targ.data(), c_est.data());
        }
        finufftf_destroy(plan);
      }
      if (err < tol)
        sigma_upper = opts.upsampfac;
      else
        sigma_lower = opts.upsampfac;
    }
    sigmas.push_back(sigma_upper);
    if (sigma_upper == sigma_max) lowest_tol_idx = max(0, idx);
    if (sigma_upper < sigma_max && sigma_upper < comp_sigma + cmd_opts.sigma_prec) {
      upper_tol_idx = upper_tol_idx == -1 ? idx : upper_tol_idx;
      break;
    }
    exp_tol += max_step;
  }
  upper_tol_idx = upper_tol_idx == -1 ? (int)tol_range.size() - 1 : upper_tol_idx;

  // Build CSV block for this combo (avoids interleaving with other threads)
  double tolfac = 0.18 * pow(1.4, n_dims - 1);
  if (type == 3) tolfac *= 1.4;
  string csv_block;
  for (size_t i = 0; i < tol_range.size(); i++) {
    double sigma_anal = lowest_sigma(tol_range[i], type, n_dims, maxns, 8);
    double b_val      = log(tolfac / tol_range[i]) / ((maxns - 1) * finufft::common::PI);
    double delta_val  = sigmas[i] - sigma_anal;
    char buf[200];
    snprintf(buf, sizeof(buf), "%c,%d,%d,%.6e,%.8f,%.8f,%.10f,%.10f,%d\n",
             is_same_v<T, float> ? 'f' : 'd', type, n_dims, tol_range[i], sigmas[i],
             sigma_anal, b_val, delta_val, cmd_opts.Ntotal);
    csv_block += buf;
  }

  // Polynomial fit for stderr diagnostics
  string fit_block;
  vector<double> tol_x(tol_range.begin() + lowest_tol_idx,
                       tol_range.begin() + upper_tol_idx + 1);
  if (tol_x.size() >= 2) {
    vector<double> ups_y(sigmas.begin() + lowest_tol_idx,
                         sigmas.begin() + upper_tol_idx + 1);
    transform(tol_x.begin(), tol_x.end(), tol_x.begin(), [](double v) { return log(v); });
    double x_center = 0.5 * (tol_x.front() + tol_x.back());
    for (auto &v : tol_x) v -= x_center;
    vector<double> delta_y(tol_x.size());
    for (size_t i = 0; i < tol_x.size(); i++)
      delta_y[i] =
          ups_y[i] - lowest_sigma(exp(tol_x[i] + x_center), type, n_dims, maxns, 8);

    auto poly1      = andviane::polynomial_regression<1>(tol_x, delta_y);
    auto poly3      = andviane::polynomial_regression<3>(tol_x, delta_y);
    double max_res1 = 0.0, max_res3 = 0.0;
    for (size_t i = 0; i < tol_x.size(); i++) {
      double f1 = poly1[0] + poly1[1] * tol_x[i];
      double f3 = poly3[0] + poly3[1] * tol_x[i] + poly3[2] * tol_x[i] * tol_x[i] +
                  poly3[3] * tol_x[i] * tol_x[i] * tol_x[i];
      max_res1  = std::max(max_res1, std::abs(f1 - delta_y[i]));
      max_res3  = std::max(max_res3, std::abs(f3 - delta_y[i]));
    }
    bool use_deg1 = max_res1 < 0.02;
    vector<double> coeffs(use_deg1 ? vector<double>(poly1.begin(), poly1.end())
                                   : vector<double>(poly3.begin(), poly3.end()));
    ostringstream ss;
    ss << "  delta deg-1 max residual: " << fixed << setprecision(4) << max_res1
       << "  coeffs: [" << setprecision(8) << poly1[0] << ", " << poly1[1] << "]\n"
       << "  delta deg-3 max residual: " << setprecision(4) << max_res3 << "\n"
       << "DeltaEstimator{ type: " << type << ", n_dims: " << n_dims
       << ", maxns: " << maxns << ", x_center: " << setprecision(10) << x_center
       << ", degree: " << (use_deg1 ? 1 : 3) << ", delta_coefficients: {";
    for (size_t i = 0; i < coeffs.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << setprecision(10) << coeffs[i];
    }
    ss << "}, " << scientific << tol_range[lowest_tol_idx] << ", "
       << tol_range[upper_tol_idx] << ", type: " << typeid(T).name() << " }\n";
    fit_block = ss.str();
  }

  {
    std::lock_guard<std::mutex> lock(output_mutex);
    fputs(csv_block.c_str(), stdout);
    fflush(stdout);
    if (!fit_block.empty()) fputs(fit_block.c_str(), stderr);
  }
}

int main(int argc, char *argv[]) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    train_options_t default_opts(0, nullptr);
    std::cout << "Valid options:\n"
                 "    --prec <char>[,<char>...]\n"
                 "           list of precisions: float or double.\n"
                 "           default: "
              << default_opts.options_map["prec"] << "\n"
              << "    --type <int>[,<int>...]\n"
                 "           list of transform types.\n"
                 "           default: "
              << default_opts.options_map["type"] << "\n"
              << "    --dim <int>[,<int>...]\n"
                 "           list of dimension numbers.\n"
                 "           default: "
              << default_opts.options_map["dim"] << "\n"
              << "    --N <int>\n"
                 "           number of modes.\n"
                 "           default: "
              << default_opts.options_map["N"] << "\n"
              << "    --M <int>\n"
                 "           number of non-uniform points.\n"
                 "           default: "
              << default_opts.options_map["M"] << "\n"
              << "    --sigma-prec <int>\n"
                 "           precision of sigma value binary search.\n"
              << "           default: " << default_opts.options_map["sigma-prec"] << "\n"
              << "    --ntol <int>\n"
                 "           Number of tolerance values in a range.\n"
              << "           default: " << default_opts.options_map["ntol"] << "\n"
              << "    --nthreads <int>\n"
                 "           Number of threads for FINUFFT (0=auto).\n"
              << "           default: " << default_opts.options_map["nthreads"] << "\n";
    return 0;
  }
  train_options_t cmd_opts(argc, argv);
  fprintf(stdout, "prec,type,dim,tol,sigma_empirical,sigma_analytical,b,delta,N\n");
  std::mutex output_mutex;
  // Launch all (prec, dim, type) combos simultaneously
  vector<std::thread> threads;
  for (auto &d : cmd_opts.dim)
    for (auto &t : cmd_opts.type) {
      if (cmd_opts.prec.count('f'))
        threads.emplace_back(run_combo<float>, d, t, std::cref(cmd_opts),
                             std::ref(output_mutex));
      if (cmd_opts.prec.count('d'))
        threads.emplace_back(run_combo<double>, d, t, std::cref(cmd_opts),
                             std::ref(output_mutex));
    }
  for (auto &th : threads) th.join();
  return 0;
}

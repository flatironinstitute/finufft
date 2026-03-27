#include <cmath>
#include <finufft.h>
#include <finufft/test_defs.hpp>
#include <finufft_common/common.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <polynomial_regression.hpp>
#include <random>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
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
  unordered_map<string, string> options_map;

  train_options_t(int argc, char *argv[]) {
    options_map["dim"]        = "1";
    options_map["type"]       = "1,2,3";
    options_map["prec"]       = "f,d";
    options_map["N"]          = "5000";
    options_map["M"]          = "5000";
    options_map["sigma-prec"] = "1e-5";
    // clang-format off
        static struct option long_options[] {
            {"dim", required_argument, 0, 0},
            {"type", required_argument, 0, 0},
            {"prec", required_argument, 0, 0},
            {"sigma-prec", required_argument, 0, 0},
            {"N", required_argument, 0, 0},
            {"M", required_argument, 0, 0}
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

template<typename T> void train(train_options_t &cmd_opts) {
  pair<double, double> sigma_bounds = {1.0, 2.0};
  double limit                      = static_cast<double>(numeric_limits<T>::epsilon());
  vector<double> tol_range          = log_scale(limit, min(1e-2, limit * 1e10), 100);
  int64_t N[3];
  const int n_transf  = 1;
  constexpr int iflag = 1;

  std::vector<T> x(cmd_opts.M), y(cmd_opts.M), z(cmd_opts.M);
  std::vector<T> s(cmd_opts.Ntotal), t(cmd_opts.Ntotal), u(cmd_opts.Ntotal);
  std::vector<complex<T>> c_est(cmd_opts.M), c_targ(cmd_opts.M), c_input(cmd_opts.M);
  std::vector<complex<T>> f_est(cmd_opts.Ntotal), f_targ(cmd_opts.Ntotal),
      f_input(cmd_opts.Ntotal);

  random_device rd;
  mt19937 gen(12345);
  uniform_real_distribution<T> udis(-1.0, 1.0);

  auto random11 = [&]() {
    return udis(gen);
  };
  auto crandom11 = [&]() {
    return udis(gen) + static_cast<complex<T>>(1.0i) * udis(gen);
  };

  for (auto &n_dims : cmd_opts.dim) {
    for (int i = 0; i < cmd_opts.M; i++) {
      x[i]       = PI * random11();
      y[i]       = PI * random11();
      z[i]       = PI * random11();
      c_input[i] = crandom11();
    }
    dirft1d1(cmd_opts.M, x, c_input, iflag, cmd_opts.Ntotal, f_input);
    for (int i = 0; i < cmd_opts.Ntotal; i++) {
      s[i] = PI * random11();
      t[i] = PI * random11();
      u[i] = PI * random11();
    }
    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.allow_eps_too_small = 1;
    N[0] = cmd_opts.Ntotal, N[1] = 1, N[2] = 1;
    for (auto &type : cmd_opts.type) {
      int maxns = finufft::common::MAX_NSPREAD;
      if constexpr (is_same_v<T, float>) maxns = 8;
      double sigma_upper = sigma_bounds.second;
      switch (type) {
      case (1):
        dirft1d1(cmd_opts.M, x, c_input, iflag, cmd_opts.Ntotal, f_targ);
        break;
      case (2):
        dirft1d2(cmd_opts.M, x, c_targ, iflag, cmd_opts.Ntotal, f_input);
        break;
      case (3):
        dirft1d3(cmd_opts.M, x, c_input, iflag, cmd_opts.Ntotal, s, f_targ);
        break;
      }
      vector<double> sigmas;
      int lowest_tol_idx = 0;
      for (auto &tol : tol_range) {
        double comp_sigma  = finufft::kernel::lowest_sigma(tol, type, n_dims, maxns, 8);
        double sigma_lower = min(sigma_upper, comp_sigma);
        while (sigma_upper - sigma_lower > cmd_opts.sigma_prec) {
          opts.upsampfac = (sigma_upper + sigma_lower) / 2;
          float err;
          if constexpr (is_same_v<T, double>) {
            finufft_plan_s *plan{nullptr};
            finufft_makeplan(type, n_dims, N, iflag, n_transf, tol, &plan, &opts);
            finufft_setpts(plan, cmd_opts.M, x.data(), y.data(), z.data(),
                           cmd_opts.Ntotal, s.data(), t.data(), u.data());
            if (type == 1 || type == 3) {
              finufft_execute(plan, c_input.data(), f_est.data());
              err = relerrtwonorm(cmd_opts.Ntotal, f_targ.data(), f_est.data());
            } else {
              finufft_execute(plan, c_est.data(), f_input.data());
              err = relerrtwonorm(cmd_opts.M, c_targ.data(), c_est.data());
            }
            finufft_destroy(plan);
          } else if constexpr (is_same_v<T, float>) {
            finufftf_plan_s *plan{nullptr};
            finufftf_makeplan(type, n_dims, N, iflag, n_transf, tol, &plan, &opts);
            finufftf_setpts(plan, cmd_opts.M, x.data(), y.data(), z.data(),
                            cmd_opts.Ntotal, s.data(), t.data(), u.data());
            if (type == 1 || type == 3) {
              finufftf_execute(plan, c_input.data(), f_est.data());
              err = relerrtwonorm(cmd_opts.Ntotal, f_targ.data(), f_est.data());
            } else {
              finufftf_execute(plan, c_est.data(), f_input.data());
              err = relerrtwonorm(cmd_opts.M, c_targ.data(), c_est.data());
            }
            finufftf_destroy(plan);
          } else {
            return;
          }
          if (err < tol) {
            sigma_upper = opts.upsampfac;
          } else
            sigma_lower = opts.upsampfac;
        }
        sigmas.push_back(sigma_upper);
        if (sigma_upper == sigma_bounds.second) lowest_tol_idx++;
        if (sigma_upper < sigma_bounds.second &&
            sigma_upper < comp_sigma + cmd_opts.sigma_prec)
          break;
      }
      if (lowest_tol_idx > 0) lowest_tol_idx--;
      vector<double> tol_x(tol_range.begin() + lowest_tol_idx,
                           tol_range.begin() + sigmas.size());
      if (tol_x.size() < 2) continue;
      double lower_tol = tol_x.front();
      double upper_tol = tol_x.back();
      transform(tol_x.begin(), tol_x.end(), tol_x.begin(), [=](double tol) {
        return finufft::kernel::map_to_domain(tol, lower_tol, upper_tol);
      });
      vector<double> ups_y(sigmas.begin() + lowest_tol_idx, sigmas.end());
      auto polynomial = andviane::polynomial_regression<3>(tol_x, ups_y);
      vector<double> coeffs(polynomial.begin(), polynomial.end());

      cout << "SigmaEstimator{ type: " << type << ", n_dims: " << n_dims
           << ", maxns: " << maxns << ", polynomial_coefficients: [";
      for (size_t i = 0; i < coeffs.size(); ++i) {
        if (i > 0) cout << ", ";
        cout << std::fixed << std::setprecision(4) << coeffs[i];
      }
      cout << "], lower_tol: " << std::scientific << lower_tol
           << ", upper_tol: " << std::scientific << upper_tol
           << ", type: " << typeid(T).name() << " }\n";
    }
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
              << "           default: " << default_opts.options_map["sigma-prec"] << "\n";
    return 0;
  }
  train_options_t cmd_opts(argc, argv);
  if (cmd_opts.prec.find('f') != cmd_opts.prec.end()) train<float>(cmd_opts);
  if (cmd_opts.prec.find('d') != cmd_opts.prec.end()) train<double>(cmd_opts);
  return 0;
}

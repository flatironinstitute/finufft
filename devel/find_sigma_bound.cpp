#include <finufft_common/common.h>
#include "PolynomialRegression.hpp"
#include <iostream>
#include <cmath>
#include <direct/norms.hpp>
#include <direct/dirft1d.hpp>
#include <finufft/test_defs.hpp>
#include <finufft.h>
#include <random>
#include <map>
#include <ranges>
#include <fstream>
#include <getopt.h>
#include <set>
#include <sstream>
#include <string>

using namespace std;

set<string> get_set(const unordered_map<string, string> &m, const string key) {
    auto it = m.find(key);
    if(it == m.end())
        return {};
    set<string> res = {};
    stringstream ss(it->second);
    string tok;
    while(getline(ss, tok, ','))
        res.insert(tok);
    return res;
}

struct train_options_t {
    set<int> dim;
    set<int> type;
    set<char> prec;
    int Ntotal;
    int M;
    double sigma_prec;
    string output_path;
    unordered_map<string, string> options_map;

    train_options_t(int argc, char *argv[]) {
        options_map["dim"] = "1";
        options_map["type"] = "1,2,3";
        options_map["prec"] = "f,d";
        options_map["N"] = "5000";
        options_map["M"] = "5000";
        options_map["sigma-prec"] = "1e-5";
        options_map["output-path"] = SAVE_PATH;
        // clang-format off
        static struct option long_options[] {
            {"dim", required_argument, 0, 0},
            {"type", required_argument, 0, 0},
            {"prec", required_argument, 0, 0},
            {"sigma-prec", required_argument, 0, 0},
            {"output-path", required_argument, 0, 0},
            {"N", required_argument, 0, 0},
            {"M", required_argument, 0, 0}
        };
        // clang-format on
        int option_index = 0;
        while(true) {
            if(getopt_long(argc, argv, "", long_options, &option_index) == -1)
                break;
            options_map[long_options[option_index].name] = optarg;
        }
        for(auto &t: get_set(options_map, "type"))
            type.insert(stoi(t));
        for(auto &t: get_set(options_map, "dim"))
            dim.insert(stoi(t));
        for(auto &t: get_set(options_map, "prec"))
            prec.insert(t[0]);

        Ntotal = stoi(options_map["N"]);
        M = stoi(options_map["M"]);
        sigma_prec = stod(options_map["sigma-prec"]);
        output_path = options_map["output-path"];
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

int main(int argc, char *argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        train_options_t default_opts(0, nullptr);
        std::cout << "Valid options:\n"
                     "    --prec <char>[,<char>...]\n"
                     "           list of precisions: float or double.\n"
                     "           default: " << default_opts.options_map["prec"] << "\n" <<
                     "    --type <int>[,<int>...]\n"
                     "           list of transform types.\n"
                     "           default: " << default_opts.options_map["type"] << "\n" <<
                     "    --dim <int>[,<int>...]\n"
                     "           list of dimension numbers.\n"
                     "           default: " << default_opts.options_map["dim"] << "\n" <<
                     "    --N <int>\n"
                     "           number of modes.\n"
                     "           default: " << default_opts.options_map["N"] << "\n" <<
                     "    --M <int>\n"
                     "           number of non-uniform points.\n"
                     "           default: " << default_opts.options_map["M"] << "\n" <<
                     "    --sigma-prec <int>\n"
                     "           precision of sigma value binary search.\n" <<
                     "           default: " << default_opts.options_map["sigma-prec"] << "\n" <<
                     "    --save-path <str>\n"
                     "           path to save trained predictors at.\n" <<
                     "           default: " << default_opts.options_map["output-path"] << "\n";
        return 0;
    }
    train_options_t cmd_opts(argc, argv);
    pair<double, double> sigma_bounds = {1.0, 2.0};
    vector<double> tol_range          = log_scale(1e-15, 1e-5, 100);
    int64_t N[3];
    const int n_transf = 1;
    constexpr int iflag    = 1;

    std::vector<double> x(cmd_opts.M), y(cmd_opts.M), z(cmd_opts.M);
    std::vector<double> s(cmd_opts.Ntotal), t(cmd_opts.Ntotal), u(cmd_opts.Ntotal);
    std::vector<complex<double>> c_est(cmd_opts.M), c_targ(cmd_opts.M), c_input(cmd_opts.M);
    std::vector<complex<double>> f_est(cmd_opts.Ntotal), f_targ(cmd_opts.Ntotal), f_input(cmd_opts.Ntotal);

    random_device rd;
    mt19937 gen(12345);
    uniform_real_distribution<double> udis(-1.0, 1.0);
    
    auto random11 = [&]() { return udis(gen); };
    auto crandom11 = [&]() { return udis(gen) + 1i * udis(gen); };

    vector<finufft::kernel::SigmaEstimator> predictors;
    for(auto &n_dims: cmd_opts.dim) {
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
        N[0] = cmd_opts.Ntotal, N[1] = 1, N[2] = 1;
        for (auto &type: cmd_opts.type) {
            auto sigma_dynamic = sigma_bounds;
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
            double impossible_tol = tol_range[0];
            int lowest_tol_idx = 0;
            for (auto &tol : tol_range) {
                double comp_sigma = finufft::kernel::get_sigma(tol, type, n_dims);
                sigma_dynamic.first = comp_sigma;
                while (sigma_dynamic.second - sigma_dynamic.first > cmd_opts.sigma_prec) {
                    opts.upsampfac = (sigma_dynamic.second + sigma_dynamic.first) / 2;
                    finufft_plan_s *plan{nullptr};
                    finufft_makeplan(type, n_dims, N, iflag, n_transf, tol, &plan, &opts);
                    finufft_setpts(plan, cmd_opts.M, x.data(), y.data(), z.data(), cmd_opts.Ntotal, s.data(), t.data(), u.data());
                    float err;
                    if (type == 1 || type == 3) {
                        finufft_execute(plan, c_input.data(), f_est.data());
                        err = relerrtwonorm(cmd_opts.Ntotal, f_targ.data(), f_est.data());
                    } else {
                        finufft_execute(plan, c_est.data(), f_input.data());
                        err = relerrtwonorm(cmd_opts.M, c_targ.data(), c_est.data());
                    }
                    finufft_destroy(plan);

                    if (err < tol) {
                        sigma_dynamic.second = opts.upsampfac;
                    } else
                        sigma_dynamic.first = opts.upsampfac;
                }
                if(sigma_dynamic.second == sigma_bounds.second)
                    lowest_tol_idx++;
                sigmas.push_back(sigma_dynamic.second);
                if(sigma_dynamic.second < comp_sigma+cmd_opts.sigma_prec)
                    break;
            }
            if(lowest_tol_idx>0) lowest_tol_idx--;
            vector<double> tol_x(tol_range.begin()+lowest_tol_idx, tol_range.begin()+sigmas.size());
            double lower_tol = tol_x.front();
            double upper_tol = tol_x.back();
            transform(tol_x.begin(), tol_x.end(), tol_x.begin(), [=](double tol){ return finufft::kernel::map_to_domain(tol, lower_tol, upper_tol);});
            vector<double> ups_y(sigmas.begin()+lowest_tol_idx, sigmas.end());
            vector<double> coeffs(finufft::kernel::SigmaEstimator::NCOEFFS);
            PolynomialRegression<double>().fitIt(tol_x, ups_y, finufft::kernel::SigmaEstimator::NCOEFFS-1, coeffs);
            predictors.push_back(finufft::kernel::SigmaEstimator(type, n_dims, coeffs, lower_tol, upper_tol, typeid(double)));
        }
    }
    ofstream out(cmd_opts.output_path);
    out << "#include <finufft_common/common.h>" << endl;
    out << "finufft::kernel::SigmaEstimator trained[] = {";
    for(int i=0;i<predictors.size();i++) {
        if(i)
            out << "," << endl;
        out << predictors[i];
    }
    out << "};";
    out.close();
    return 0;
}

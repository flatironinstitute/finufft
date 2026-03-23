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

using namespace std;

#define NCOEFFS 4

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

    const int64_t M                         = 5000;
    const long Ntotal                       = 5000;
    int n_dims = 1;
    double const sigma_prec = 1e-5;

    pair<double, double> sigma_bounds = {1.0, 2.0};
    vector<double> tol_range          = log_scale(1e-15, 1e-5, 100);
    int64_t N[3];

    const int n_transf = 1;
    constexpr int iflag    = 1;
    std::vector<double> x(M), y(M), z(M);
    std::vector<double> s(Ntotal), t(Ntotal), u(Ntotal);
    std::vector<complex<double>> c_est(M), c_targ(M), c_input(M);
    std::vector<complex<double>> f_est(Ntotal), f_targ(Ntotal), f_input(Ntotal);

    random_device rd;
    mt19937 gen(12345);
    uniform_real_distribution<double> udis(-1.0, 1.0);
    
    auto random11 = [&]() { return udis(gen); };
    auto crandom11 = [&]() { return udis(gen) + 1i * udis(gen); };

    for (int i = 0; i < M; i++) {
      x[i]       = PI * random11();
      y[i]       = PI * random11();
      z[i]       = PI * random11();
      c_input[i] = crandom11();
    }
    dirft1d1(M, x, c_input, iflag, Ntotal, f_input);
    for (int i = 0; i < Ntotal; i++) {
      s[i] = PI * random11();
      t[i] = PI * random11();
      u[i] = PI * random11();
    }
    finufft_opts opts;
    finufft_default_opts(&opts);
    N[0] = Ntotal, N[1] = 1, N[2] = 1;
    vector<finufft::kernel::SigmaEstimator> predictors;
    for (int type = 1; type <= 3; type++) {
        int dim = 1;
        type = 3;
        auto sigma_dynamic = sigma_bounds;
        switch (type) {
            case (1):
              dirft1d1(M, x, c_input, iflag, Ntotal, f_targ);
              break;
            case (2):
              dirft1d2(M, x, c_targ, iflag, Ntotal, f_input);
              break;
            case (3):
              dirft1d3(M, x, c_input, iflag, Ntotal, s, f_targ);
              break;
        }
        vector<double> sigmas;
        double impossible_tol = tol_range[0];
        int lowest_tol_idx = 0;
        for (auto &tol : tol_range) {
            double comp_sigma = finufft::kernel::get_sigma(tol, type, dim);
            sigma_dynamic.first = comp_sigma;
            while (sigma_dynamic.second - sigma_dynamic.first > sigma_prec) {
                opts.upsampfac = (sigma_dynamic.second + sigma_dynamic.first) / 2;
                finufft_plan_s *plan{nullptr};
                finufft_makeplan(type, n_dims, N, iflag, n_transf, tol, &plan, &opts);
                finufft_setpts(plan, M, x.data(), y.data(), z.data(), Ntotal, s.data(), t.data(), u.data());
                float err;
                if (type == 1 || type == 3) {
                  finufft_execute(plan, c_input.data(), f_est.data());
                  err = relerrtwonorm(Ntotal, f_targ.data(), f_est.data());
                } else {
                  finufft_execute(plan, c_est.data(), f_input.data());
                  err = relerrtwonorm(M, c_targ.data(), c_est.data());
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
            if(sigma_dynamic.second < comp_sigma+sigma_prec)
                break;
        }
        if(lowest_tol_idx>0) lowest_tol_idx--;
        vector<double> tol_x(tol_range.begin()+lowest_tol_idx, tol_range.begin()+sigmas.size());
        double lower_tol = tol_x.front();
        double upper_tol = tol_x.back();
        transform(tol_x.begin(), tol_x.end(), tol_x.begin(), [=](double tol){ return finufft::kernel::map_to_domain(tol, lower_tol, upper_tol);});
        vector<double> ups_y(sigmas.begin()+lowest_tol_idx, sigmas.end());
        vector<double> coeffs(NCOEFFS);
        PolynomialRegression<double>().fitIt(tol_x, ups_y, NCOEFFS-1, coeffs);
        predictors.push_back(finufft::kernel::SigmaEstimator(type, dim, coeffs, lower_tol, upper_tol, typeid(double)));
        break;
    }
    cout << SAVE_PATH << endl;
    ofstream out(SAVE_PATH);
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

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
#include <typeindex>

using namespace std;

#define NCOEFFS 4

double get_sigma(double tol, int type, int dim) {
  double maxns  = 16;
  double tolfac = 0.18 * pow(1.4, dim - 1);
  if (type == 3) tolfac *= 1.4;
  double a = log(tolfac / tol);
  double b = pow(a / ((maxns - 1) * PI), 2);
  return 1 / (1 - b);
}
struct Predictor {
public:
    Predictor() {};
    Predictor(istream &byte_stream);
    Predictor(int type, int dim, vector<double> &coeffs, double lower_tol, double upper_tol, type_index &precision);
    bool match(int transform_type, int transform_dim, type_index &transform_precision) {
        return transform_type == type && transform_dim == dim && from_typeid(transform_precision) == precision;
    }
    double best_sigma(double tol) {
        if(tol < lower_tol) 
            return 2.0;
        if(tol < upper_tol) {
            return 1.0;
        }
        return get_sigma(tol, type, dim);
    }
    friend ostream &operator<<(ostream &os, const Predictor &self);
private:
    int type;
    int dim;
    double coeffs[NCOEFFS]; 
    double lower_tol;
    double upper_tol;
    enum {F, D} precision;
    decltype(precision) from_typeid(type_index &prec) {
        if(prec == typeid(double))
            return D;
        return F;
   };
};
Predictor::Predictor(int type, int dim, vector<double> &coeffs, double lower_tol, double upper_tol, type_index &prec): type(type), dim(dim), lower_tol(lower_tol), upper_tol(upper_tol) {
    precision = from_typeid(prec);
}
Predictor::Predictor(istream &byte_stream) {
   byte_stream.read(reinterpret_cast<char*>(this), sizeof(Predictor));
}
ostream &operator<<(ostream &os, const Predictor &self) {
    os.write(reinterpret_cast<const char*>(&self), sizeof(Predictor));
    return os;
}

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
    for (int type = 1; type <= 3; type++) {
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
            double comp_sigma = get_sigma(tol, type, 1);
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
        vector<double> ups_y(sigmas.begin()+lowest_tol_idx, sigmas.end());
        vector<double> coeffs(2);
        PolynomialRegression<double>().fitIt(tol_x, ups_y, coeffs.size(), coeffs);
        for(auto a: coeffs) 
            cout << a << endl;
        ofstream out("predictors.bin", ios::binary);
        out.close();
        break;
    }
    return 0;
}

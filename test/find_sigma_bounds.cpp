#include <finufft/test_defs.hpp>
#include "utils/dirft1d.hpp"
#include "utils/dirft2d.hpp"
#include "utils/dirft3d.hpp"
#include "utils/norms.hpp"
#include <finufft.h>

using namespace std;

double prec = 1e-5;

template<typename T> std::vector<T> log_scale(T low, T hi, int n) {
    vector<T> res;
    T base = log(low);
    T step = (log(hi)-base)/(n-1);

    for(int i=0;i<n;i++) {
        res.push_back(exp(base+i*step));
    }
    return res;
}

tuple<int, int> factor2(int val) {
    int a = static_cast<int>(sqrt(val));
    while(val%a!=0 && a>1) a--;
    return {a, val/a};
}
tuple<int, int, int> factor3(int val) {
    auto f2 = factor2(val);
    auto flarge = factor2(get<1>(f2));
    return {get<0>(f2), get<0>(flarge), get<1>(flarge)};
}

int64_t M = 500; 
pair<double, double> sigma_bounds = {1.15, 3.0};
int n_tol = 15;
vector<double> tol_range = log_scale(1e-12, 1e-5, n_tol);

int main(int argc, char *argv[]) {
    int n_dims = 1;
    int64_t N[3]; 
    long Nmax = 500;

    cout << "Tolerances: ";
    for(auto &tol : tol_range) {
        cout << tol << ",";
    }
    cout << endl;
    int isign = 1;
    std::vector<double> x(M), y(M), z(M);
    std::vector<double> s(Nmax), t(Nmax), u(Nmax);
    std::vector<complex<double>> c_est(M), c_targ(M), c_input(M);
    std::vector<complex<double>> f_est(Nmax), f_targ(Nmax), f_input(Nmax);
    for(int i=0;i<M;i++) {
        x[i] = PI * randm11();
        y[i] = PI * randm11();
        z[i] = PI * randm11();
        c_input[i] = crandm11();
    }
    for(int i=0;i<Nmax;i++) {
        f_input[i] = crandm11();
        s[i] = PI * randm11();
        t[i] = PI * randm11();
        u[i] = PI * randm11();
    }
    finufft_opts opts;
    finufft_default_opts(&opts);
    
    int n_transf = 1;
    int iflag = 1;
    for(int type=1;type<=3;type++) {
    for(int n_dims=1;n_dims<=3;n_dims++) {
        switch(n_dims) {
            case(1): N[0]=Nmax, N[1]=1, N[2]=1; break;
            case(2): {
                auto f2 = factor2(Nmax);
                N[0]=get<0>(f2); 
                N[1]=get<1>(f2); 
                N[2]=1; break;
            }
            case(3): {
                auto f3 = factor3(Nmax);
                N[0]=get<0>(f3); 
                N[1]=get<1>(f3); 
                N[2]=get<2>(f3); 
                break;
            }
        }
        int64_t Ntotal = N[0]*N[1]*N[2];
        cout << "Type: " << type << " Ndims: " << n_dims <<  " -- ";
        auto sigma_dynamic = sigma_bounds;
        switch(type) {
            case(1): switch(n_dims) {
                case(1): dirft1d1(M, x, c_input, iflag, N[0], f_targ); break;
                case(2): dirft2d1(M, x, y, c_input, iflag, N[0], N[1], f_targ); break;
                case(3): dirft3d1(M, x, y, z, c_input, isign, N[0], N[1], N[2], f_targ); break;
            } break;
            case(2): switch(n_dims) {
                case(1): dirft1d2(M, x, c_targ, iflag, N[0], f_input); break;
                case(2): dirft2d2(M, x, y, c_targ, iflag, N[0], N[1], f_input); break;
                case(3): dirft3d2(M, x, y, z, c_targ, isign, N[0], N[1], N[2], f_input); break;
            } break;
            case(3): switch(n_dims) {
                case(1): dirft1d3(M, x, c_input, iflag, Ntotal, s, f_targ); break;
                case(2): dirft2d3(M, x, y, c_input, iflag, Ntotal, s, t, f_targ); break;
                case(3): dirft3d3(M, x, y, z, c_input, isign, Ntotal, s, t, u, f_targ); break;
            } break;
        }
        for(auto &tol: tol_range) {
            sigma_dynamic.first = sigma_bounds.first;
            if(sigma_bounds.first+prec<sigma_dynamic.second)
                while(sigma_dynamic.second - sigma_dynamic.first > prec) {
                    opts.upsampfac = (sigma_dynamic.second + sigma_dynamic.first)/2;
                    finufft_plan_s *plan{nullptr};
                    finufft_makeplan(type, n_dims, N, iflag, n_transf, tol, &plan, &opts);
                    finufft_setpts(plan, M, x.data(), y.data(), z.data(), Ntotal, s.data(), t.data(), u.data());
                    float err;
                    if(type == 1 || type == 3) {
                        finufft_execute(plan, c_input.data(), f_est.data());
                        err = relerrtwonorm(Ntotal, f_targ.data(), f_est.data());
                    } else {
                        finufft_execute(plan, c_est.data(), f_input.data());
                        err = relerrtwonorm(M, c_targ.data(), c_est.data());
                    }
                    finufft_destroy(plan);

                    if(err<tol) 
                        sigma_dynamic.second = opts.upsampfac;
                    else
                        sigma_dynamic.first = opts.upsampfac;
                }
            cout << sigma_dynamic.second << ",";
        }
        cout << endl;
    }}
    return 0;
}

#include <finufft.h>
#include <finufft/test_defs.hpp>
#include <fstream>
#include <typeindex>
#include <cassert>
#include <vector>
#include <algorithm>
#include <array>
#define NCOEFFS 4

using namespace std;

double get_sigma(double tol, int type, int dim) {
  double maxns  = 16;
  double tolfac = 0.18 * pow(1.4, dim - 1);
  if (type == 3) tolfac *= 1.4;
  double a = log(tolfac / tol);
  double b = pow(a / ((maxns - 1) * PI), 2);
  return 1 / (1 - b);
}

double map_to_domain(double x, double lower, double upper) {
    double span = log(upper) - log(lower);
    double sum_endpoints = (log(upper) + log(lower));
    return (log(x)*2-sum_endpoints)/span;
}

struct Predictor {
public:
    Predictor() {};
    Predictor(istream &byte_stream);
    Predictor(int type, int dim, vector<double> &coeffs, double lower_tol, double upper_tol, type_index prec);
    bool match(int transform_type, int transform_dim, type_index &transform_precision);
    double best_sigma(double tol);
    friend ostream &operator<<(ostream &os, const Predictor &self);
private:
    int type;
    int dim;
    array<double, NCOEFFS> coefficients; 
    double lower_tol;
    double upper_tol;
    enum {F, D} precision;
    decltype(precision) from_typeid(type_index &prec) {
        if(prec == typeid(double))
            return D;
        return F;
   };
};
double Predictor::best_sigma(double tol) {
    if(tol < lower_tol) 
        return 2.0;
    if(tol < upper_tol) {
        double mult = 1;
        double res = 0;
        double log_scaled = map_to_domain(tol, lower_tol, upper_tol);
        for(auto &coeff: coefficients) {
            res += coeff*mult; 
            mult *= log_scaled;
        }
        return min(res, 2.0);
    }
    return get_sigma(tol, type, dim);
}
bool Predictor::match(int transform_type, int transform_dim, type_index &transform_precision) {
    return transform_type == type && transform_dim == dim && from_typeid(transform_precision) == precision;
}
Predictor::Predictor(int type, int dim, vector<double> &coeffs, double lower_tol, double upper_tol, type_index prec): type(type), dim(dim), lower_tol(lower_tol), upper_tol(upper_tol) {
    assert(coeffs.size() == coefficients.size());
    //reverse_copy(coeffs.begin(), coeffs.end(), coefficients.begin());
    copy(coeffs.begin(), coeffs.end(), coefficients.begin());
    precision = from_typeid(prec);
}
Predictor::Predictor(istream &byte_stream) {
   byte_stream.read(reinterpret_cast<char*>(this), sizeof(Predictor));
}
ostream &operator<<(ostream &os, const Predictor &self) {
    os.write(reinterpret_cast<const char*>(&self), sizeof(Predictor));
    return os;
}

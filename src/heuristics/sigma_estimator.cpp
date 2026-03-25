#include <finufft/heuristics.hpp>
#include <finufft_common/common.h>
#include <cmath>
#include <cassert>
#include <string>
#include <sstream>

namespace finufft::heuristics {


std::optional<const SigmaEstimator*> get_estimator(int transform_type, int transform_dim, std::type_index transform_precision) {
    for(auto &estimator: trained) {
        if(estimator.match(transform_type, transform_dim, transform_precision))
            return &estimator;
    }
}

double get_sigma(double tol, int type, int dim, int maxns) {
  double tolfac = 0.18 * pow(1.4, dim - 1);
  if (type == 3) tolfac *= 1.4;
  double a = std::log(tolfac / tol);
  double b = pow(a / ((maxns - 1) * finufft::common::PI), 2);
  return 1 / (1 - b);
}
double map_to_domain(double x, double lower, double upper) {
    double span = std::log(upper) - std::log(lower);
    double sum_endpoints = (std::log(upper) + std::log(lower));
    return (std::log(x)*2-sum_endpoints)/span;
}
double SigmaEstimator::best_sigma(double tol) const {
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
        return std::min(res, 2.0);
    }
    int maxns = finufft::common::MAX_NSPREAD;
    if(precision == typeid(float))
        maxns = 8;
    return get_sigma(tol, type, dim, maxns);
}
bool SigmaEstimator::match(int transform_type, int transform_dim, std::type_index transform_precision) const {
    return transform_type == type && transform_dim == dim && transform_precision == precision;
}
SigmaEstimator::SigmaEstimator(int type, int dim, const std::vector<double> &coeffs, double lower_tol, double upper_tol, std::type_index prec): type(type), dim(dim), lower_tol(lower_tol), upper_tol(upper_tol), precision(prec) {
    assert(coeffs.size() == coefficients.size());
    copy(coeffs.begin(), coeffs.end(), coefficients.begin());
}
std::ostream &operator<<(std::ostream &os, const SigmaEstimator &self) {
    std::string type = "typeid(double)";
    if(self.precision == typeid(float))
        type = "typeid(float)";
    std::stringstream inline_coeffs; 
    inline_coeffs << "{";
    for(int i=0;i<self.coefficients.size();i++) {
        if(i)
            inline_coeffs << ",";
        inline_coeffs << self.coefficients[i];
    }
    inline_coeffs << "}";
    os << "finufft::kernel::SigmaEstimator(" << self.type <<  "," << self.dim << "," << inline_coeffs.str() << "," << self.lower_tol << "," << self.upper_tol << "," <<  type << ")";
    return os;
}
}

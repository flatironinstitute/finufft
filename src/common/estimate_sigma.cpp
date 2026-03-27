#include <cmath>
#include <finufft_common/common.h>

namespace finufft::kernel {

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
double map_to_domain(double x, double lower, double upper) {
  double span          = std::log(upper) - std::log(lower);
  double sum_endpoints = (std::log(upper) + std::log(lower));
  return (std::log(x) * 2 - sum_endpoints) / span;
}

} // namespace finufft::kernel

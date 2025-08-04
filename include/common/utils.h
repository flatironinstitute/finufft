#pragma once

#include <tuple>

namespace finufft {
namespace common {

void gaussquad(int n, double *xgl, double *wgl);
std::tuple<double, double> leg_eval(int n, double x);

} // namespace common
} // namespace finufft

#ifndef MATH_PSWF_H
#define MATH_PSWF_H

#include <array>
#include <cmath>
#include <finufft_errors.h>
#include <vector>

namespace finufft::common {

/* Class for evaluation of the prolate spheroidal wavefunction
   of order zero (Psi_0^c) inside [-1,1], for arbitrary frequency parameter c.
   Computation is done using a basis of Legendre polynomials.
   This implementation is based on work by Libin Lu for FINUFFT.
   The orignal implementation was done by Vladimir Rokhlin and
   can be found in src/common/specialfunctions/ of the DMK repo
   https://github.com/flatironinstitute/dmk
   The returned function values are normalized in such a way that
   evaluation at x=0 always returns 1.0.

   CAUTION: the internal routines have been heavily tweaked compared
   to the original versions and cannot be expected to work in a more
   general context! */
class PSWF0 {
private:
  double c;
  std::vector<double> workdata; // Legendre coefficients
  std::vector<std::array<double, 3>> coef;
  double xv0;                   // factor needed for normalization

  template<typename T> T eval_raw(T x) const {
    const T xsq = x * x;
    T pjm1      = 0;
    T pjm2      = 1;
    T val       = workdata[0];

    size_t i = 1;
    for (; i + 1 < coef.size(); i += 2) {
      pjm1 = pjm2 * (xsq * coef[i][0] - coef[i][1]) - pjm1 * coef[i][2];
      val += workdata[i] * pjm1;
      pjm2 = pjm1 * (xsq * coef[i + 1][0] - coef[i + 1][1]) - pjm2 * coef[i + 1][2];
      val += workdata[i + 1] * pjm2;
    }
    for (; i < coef.size(); ++i) {
      T tmp = pjm2 * (xsq * coef[i][0] - coef[i][1]) - pjm1 * coef[i][2];
      val += workdata[i] * tmp;
      pjm1 = pjm2;
      pjm2 = tmp;
    }
    return val;
  }

public:
  PSWF0(double c_);

  double operator()(double x) const {
    if (std::abs(x) > 1) return 0.;
    return eval_raw(x) * xv0;
  }
};

} // namespace finufft::common
#endif // MATH_PSWF_H

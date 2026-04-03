#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cmath>
#include <cuda.h>
#include <cufinufft/cufinufft_plan_t.h>
#include <finufft_common/spread_opts.h>

namespace cufinufft {
namespace spreadinterp {

template<typename T> void cuspreadnd_prop(cufinufft_plan_t<T> &plan);
template<typename T>
void cuspreadnd(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                       cuda_complex<T> *fw, int blksize);
template<typename T>
void cuinterpnd(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                       const cuda_complex<T> *fw, int blksize);

template<typename T>
static inline T evaluate_kernel(T x, const finufft_spread_opts &spopts)
/* ES ("exp sqrt" or "exp semicircle") kernel evaluation, single real argument:
   returns phi(2x/ns) := exp(beta.[sqrt(1 - (2x/ns)^2) - 1]),  for |x| < ns/2.
   This is the reference implementation, used by src/cuda/common.cu for onedim
   FT quadrature approx, so it need not be fast.
   This is the original kernel used 2017-2025 in CPU code, as in [FIN] paper.
   This is related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   To do: *** replace kernel logic/coeffs with CPU codes in common/
*/
{
  T z = 2.0 * x / T(spopts.nspread); // argument on [-1,1]
  if (abs(z) >= 1.0)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else {
    return exp((T)spopts.beta * (sqrt((T)1.0 - z * z) - (T)1.0));
  }
}

} // namespace spreadinterp
} // namespace cufinufft
#endif

#pragma once

#include <finufft/test_defs.h>

// ahb's low-level array helpers
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL relerrtwonorm(BIGINT n, const std::complex<T> *a,
                                             const std::complex<T> *b)
// ||a-b||_2 / ||a||_2
{
  T err = 0.0, nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) {
    // note std::norm here & below is |a|^2 ("field norm") not usual |a| ...
    nrm += std::norm(a[m]);
    err += std::norm(a[m] - b[m]);
  }
  return sqrt(err / nrm);
}
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL errtwonorm(BIGINT n, const std::complex<T> *a,
                                          const std::complex<T> *b)
// ||a-b||_2
{
  T err = 0.0; // compute error 2-norm
  for (BIGINT m = 0; m < n; ++m) err += std::norm(a[m] - b[m]);
  return sqrt(err);
}
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL twonorm(BIGINT n, const std::complex<T> *a)
// ||a||_2
{
  T nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) nrm += std::norm(a[m]);
  return sqrt(nrm);
}
template<typename T>
FINUFFT_EXPORT T FINUFFT_CDECL infnorm(BIGINT n, const std::complex<T> *a)
// ||a||_infty
{
  T nrm = 0.0;
  for (BIGINT m = 0; m < n; ++m) nrm = std::max(nrm, std::norm(a[m]));
  return sqrt(nrm);
}

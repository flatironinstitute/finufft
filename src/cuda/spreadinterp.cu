// Plan-method dispatch entry points for spread / interp / their preparation.
//
// All templated kernels, drivers, and dispatcher structs live in the
// header <cufinufft/spreadinterp.hpp> so they can be instantiated by the
// per-method TUs (spread_*, interp_* in this directory). This file is the
// thin "main" TU: it dispatches on plan->dim at runtime and forwards into
// the per-method entry points.

#include <cufinufft/spreadinterp.hpp>

template<typename T> void cufinufft_plan_t<T>::prep_spreadinterp() {
  using namespace cufinufft::spreadinterp;
  cufinufft::utils::launch_dispatch_ndim<SpreadPropDispatcher, T>(SpreadPropDispatcher(),
                                                                  this->dim, *this);
}

template<typename T>
void cufinufft_plan_t<T>::spread(const cuda_complex<T> *c, cuda_complex<T> *fw,
                                 int blksize) const {
  using namespace cufinufft::spreadinterp;
  cufinufft::utils::launch_dispatch_ndim<SpreadDispatcher, T>(
      SpreadDispatcher(), this->dim, *this, c, fw, blksize);
}

template<typename T>
void cufinufft_plan_t<T>::interp(cuda_complex<T> *c, const cuda_complex<T> *fw,
                                 int blksize) const {
  using namespace cufinufft::spreadinterp;
  cufinufft::utils::launch_dispatch_ndim<InterpDispatcher, T>(
      InterpDispatcher(), this->dim, *this, c, fw, blksize);
}

template void cufinufft_plan_t<float>::prep_spreadinterp();
template void cufinufft_plan_t<double>::prep_spreadinterp();
template void cufinufft_plan_t<float>::spread(const cuda_complex<float> *,
                                              cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread(const cuda_complex<double> *,
                                               cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::interp(cuda_complex<float> *,
                                              const cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::interp(cuda_complex<double> *,
                                               const cuda_complex<double> *, int) const;

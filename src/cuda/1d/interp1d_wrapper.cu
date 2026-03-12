#include <iostream>

#include <cuComplex.h>

#include <cufinufft/common_kernels.hpp>
#include <cufinufft/spreadinterp.h>

namespace cufinufft {
namespace spreadinterp {

// Functor to handle function selection (nuptsdriven vs subprob)
struct Interp1DDispatcher {
  template<int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c, const cuda_complex<T> *fw, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuinterp_nuptsdriven<T, 1, ns>(d_plan, c, fw, blksize);
    case 2:
      return cuinterp_subprob<T, 1, ns>(d_plan, c, fw, blksize);
    default:
      std::cerr << "[cuinterp1d] error: incorrect method, should be 1\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Updated cuinterp1d using generic dispatch
template<typename T> void cuinterp1d(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c, const cuda_complex<T> *fw, int blksize) {
  /*
   A wrapper for different interpolation methods.

   Methods available:
      (1) Non-uniform points driven
      (2) Subproblem

   Melody Shih 11/21/21

   Now the function is updated to dispatch based on ns. This is to avoid alloca which
   it seems slower according to the MRI community.
   Marco Barbone 01/30/25
  */
  launch_dispatch_ns<Interp1DDispatcher, T>(Interp1DDispatcher(), d_plan.spopts.nspread,
                                            d_plan, c, fw, blksize);
}

template void cuinterp1d<float>(const cufinufft_plan_t<float> &d_plan, cuda_complex<float> *c, const cuda_complex<float> *fw, int blksize);
template void cuinterp1d<double>(const cufinufft_plan_t<double> &d_plan, cuda_complex<double> *c, const cuda_complex<double> *fw, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

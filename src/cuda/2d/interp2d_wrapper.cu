#include <iostream>

#include <cuComplex.h>

#include <cufinufft/spreadinterp.h>
#include <cufinufft/common_kernels.hpp>

namespace cufinufft {
namespace spreadinterp {

// Functor to handle function selection (nuptsdriven vs subprob)
struct Interp2DDispatcher {
  template<int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuinterp_nuptsdriven<T, 2, ns>(d_plan, blksize);
    case 2:
      return cuinterp_subprob<T, 2, ns>(d_plan, blksize);
    default:
      std::cerr << "[cuinterp2d] error: incorrect method, should be 1 or 2\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Updated cuinterp2d using generic dispatch
template<typename T> void cuinterp2d(const cufinufft_plan_t<T> &d_plan, int blksize) {
  /*
    A wrapper for different interpolation methods.

    Methods available:
        (1) Non-uniform points driven
        (2) Subproblem

    Melody Shih 07/25/19

    Now the function is updated to dispatch based on ns. This is to avoid alloca which
    it seems slower according to the MRI community.
    Marco Barbone 01/30/25
  */
  launch_dispatch_ns<Interp2DDispatcher, T>(Interp2DDispatcher(), d_plan.spopts.nspread,
                                            d_plan, blksize);
}

template void cuinterp2d<float>(const cufinufft_plan_t<float> &d_plan, int blksize);
template void cuinterp2d<double>(const cufinufft_plan_t<double> &d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

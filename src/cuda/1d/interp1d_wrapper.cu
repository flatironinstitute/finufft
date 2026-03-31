#include <iostream>

#include <cuComplex.h>

#include <cufinufft/common_kernels.hpp>
#include <cufinufft/spreadinterp.h>

namespace cufinufft {
namespace spreadinterp {

// Functor to handle function selection (nuptsdriven vs subprob)
struct Interp1DDispatcher {
  template<int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuinterp_nuptsdriven<T, 1, ns>(d_plan, blksize);
    case 2:
      return cuinterp_subprob<T, 1, ns>(d_plan, blksize);
    default:
      std::cerr << "[cuinterp1d] error: incorrect method, should be 1 or 2\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Thin wrapper that dispatches to the shared interpolation kernels.
template<typename T> void cuinterp1d(const cufinufft_plan_t<T> &d_plan, int blksize) {
  /*
   Dispatch interpolation to the shared CUDA kernels.

   Methods available:
      (1) Non-uniform points driven
      (2) Subproblem

   Melody Shih 11/21/21

   Dispatch is specialized on ns to avoid dynamic stack allocation.
   Marco Barbone 01/30/25
  */
  launch_dispatch_ns<Interp1DDispatcher, T>(Interp1DDispatcher(), d_plan.spopts.nspread,
                                            d_plan, blksize);
}

template void cuinterp1d<float>(const cufinufft_plan_t<float> &d_plan, int blksize);
template void cuinterp1d<double>(const cufinufft_plan_t<double> &d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

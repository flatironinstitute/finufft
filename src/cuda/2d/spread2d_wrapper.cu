#include <iostream>

#include <cuComplex.h>

#include <cufinufft/common_kernels.hpp>
#include <cufinufft/spreadinterp.h>

namespace cufinufft {
namespace spreadinterp {

// Functor to handle function selection (nuptsdriven vs subprob)
struct Spread2DDispatcher {
  template<int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c, cuda_complex<T> *fw, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuspread_nupts_driven<T, 2, ns>(d_plan, c, fw, blksize);
    case 2:
      return cuspread_subprob<T, 2, ns>(d_plan, c, fw, blksize);
    case 3:
      return cuspread_output_driven<T, 2, ns>(d_plan, c, fw, blksize);
    case 42:
      return cuspread_romein<T, 2, ns>(d_plan, c, fw, blksize);
    default:
      std::cerr << "[cuspread2d] error: incorrect method, should be 1, 2 or 3\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Updated cuspread2d using generic dispatch
template<typename T> void cuspread2d(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c, cuda_complex<T> *fw, int blksize) {
  /*
    A wrapper for different spreading methods.

    Methods available:
        (1) Non-uniform points driven
        (2) Subproblem

    Melody Shih 07/25/19

    Now the function is updated to dispatch based on ns. This is to avoid alloca which
    it seems slower according to the MRI community.
    Marco Barbone 01/30/25
  */
  launch_dispatch_ns<Spread2DDispatcher, T>(Spread2DDispatcher(), d_plan.spopts.nspread,
                                            d_plan, c, fw, blksize);
}
template void cuspread2d<float>(const cufinufft_plan_t<float> &d_plan, const cuda_complex<float> *c, cuda_complex<float> *fw, int blksize);
template void cuspread2d<double>(const cufinufft_plan_t<double> &d_plan, const cuda_complex<double> *c, cuda_complex<double> *fw, int blksize);

template<typename T> void cuspread2d_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_method == 1) cuspread_nuptsdriven_prop<T, 2>(d_plan);
  if (d_plan.opts.gpu_method == 2) cuspread_subprob_prop<T, 2>(d_plan);
  if (d_plan.opts.gpu_method == 3) cuspread_subprob_prop<T, 2>(d_plan);
  if (d_plan.opts.gpu_method == 42) cuspread_subprob_prop<T, 2>(d_plan);
}
template void cuspread2d_prop(cufinufft_plan_t<float> &d_plan);
template void cuspread2d_prop(cufinufft_plan_t<double> &d_plan);
} // namespace spreadinterp
} // namespace cufinufft

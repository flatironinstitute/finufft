#include <iostream>

#include <cuComplex.h>

#include <cufinufft/common_kernels.hpp>
#include <cufinufft/spreadinterp.h>

namespace cufinufft {
namespace spreadinterp {

/* ------------------------ 1d Spreading Kernels ----------------------------*/

// Functor to handle function selection (nuptsdriven vs subprob)
struct Spread1DDispatcher {
  template<int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                  cuda_complex<T> *fw, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuspread_nupts_driven<T, 1, ns>(d_plan, c, fw, blksize);
    case 2:
      return cuspread_subprob<T, 1, ns>(d_plan, c, fw, blksize);
    case 3:
      return cuspread_output_driven<T, 1, ns>(d_plan, c, fw, blksize);
    default:
      std::cerr << "[cuspread1d] error: incorrect method, should be 1, 2 or 3\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Thin wrapper that dispatches to the shared spreading kernels.
template<typename T>
void cuspread1d(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                cuda_complex<T> *fw, int blksize) {
  /*
    Dispatch spreading to the shared CUDA kernels.

    Methods available:
        (1) Non-uniform points driven
        (2) Subproblem
        (3) Output-driven

    Melody Shih 11/21/21

    Dispatch is specialized on ns to avoid dynamic stack allocation.
    Marco Barbone 01/30/25
 */
  launch_dispatch_ns<Spread1DDispatcher, T>(Spread1DDispatcher(), d_plan.spopts.nspread,
                                            d_plan, c, fw, blksize);
}
template void cuspread1d<float>(const cufinufft_plan_t<float> &d_plan,
                                const cuda_complex<float> *c, cuda_complex<float> *fw,
                                int blksize);
template void cuspread1d<double>(const cufinufft_plan_t<double> &d_plan,
                                 const cuda_complex<double> *c, cuda_complex<double> *fw,
                                 int blksize);

template<typename T> void cuspread1d_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_method == 1) cuspread_nuptsdriven_prop<T, 1>(d_plan);
  if (d_plan.opts.gpu_method == 2) cuspread_subprob_and_OD_prop<T, 1>(d_plan);
  if (d_plan.opts.gpu_method == 3) cuspread_subprob_and_OD_prop<T, 1>(d_plan);
}
template void cuspread1d_prop(cufinufft_plan_t<float> &d_plan);
template void cuspread1d_prop(cufinufft_plan_t<double> &d_plan);

} // namespace spreadinterp
} // namespace cufinufft

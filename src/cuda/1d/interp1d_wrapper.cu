#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <iostream>

#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>

#include "spreadinterp1d.cuh"

namespace cufinufft {
namespace spreadinterp {

// Functor to handle function selection (nuptsdriven vs subprob)
struct Interp1DDispatcher {
  template<int ns, typename T>
  int operator()(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) const {
    switch (d_plan->opts.gpu_method) {
    case 1:
      return cuinterp1d_nuptsdriven<T, ns>(nf1, M, d_plan, blksize);
    default:
      std::cerr << "[cuinterp1d] error: incorrect method, should be 1\n";
      return FINUFFT_ERR_METHOD_NOTVALID;
    }
  }
};

// Updated cuinterp1d using generic dispatch
template<typename T> int cuinterp1d(cufinufft_plan_t<T> *d_plan, int blksize) {
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
  return launch_dispatch_ns<Interp1DDispatcher, T>(Interp1DDispatcher(),
                                                   d_plan->spopts.nspread, d_plan->nf1,
                                                   d_plan->M, d_plan, blksize);
}

template<typename T, int ns>
int cuinterp1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream = d_plan->stream;
  dim3 threadsPerBlock;
  dim3 blocks;

  T es_c          = d_plan->spopts.ES_c;
  T es_beta       = d_plan->spopts.ES_beta;
  T sigma         = d_plan->opts.upsampfac;
  int *d_idxnupts = d_plan->idxnupts;

  T *d_kx               = d_plan->kx;
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x = 32;
  threadsPerBlock.y = 1;
  blocks.x          = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y          = 1;

  if (d_plan->opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      interp_1d_nuptsdriven<T, 1, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      interp_1d_nuptsdriven<T, 0, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, nf1, es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template int cuinterp1d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp1d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <iostream>

#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>

#include "spreadinterp1d.cuh"

namespace cufinufft {
namespace spreadinterp {

template<typename T>
int cuinterp1d(cufinufft_plan_t<T> *d_plan, int blksize)
/*
    A wrapper for different interpolation methods.

    Methods available:
    (1) Non-uniform points driven
    (2) Subproblem

    Melody Shih 11/21/21
*/
{
  int nf1 = d_plan->nf1;
  int M   = d_plan->M;

  int ier;
  switch (d_plan->opts.gpu_method) {
  case 1: {
    ier = cuinterp1d_nuptsdriven<T>(nf1, M, d_plan, blksize);
  } break;
  default:
    std::cerr << "[cuinterp1d] error: incorrect method, should be 1" << std::endl;
    ier = FINUFFT_ERR_METHOD_NOTVALID;
  }

  return ier;
}

template<typename T>
int cuinterp1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream = d_plan->stream;
  dim3 threadsPerBlock;
  dim3 blocks;

  int ns          = d_plan->spopts.nspread; // psi's support in terms of number of cells
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
      interp_1d_nuptsdriven<T, 1><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1, es_c, es_beta, sigma,
          d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      interp_1d_nuptsdriven<T, 0><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_c + t * M, d_fw + t * nf1, M, ns, nf1, es_c, es_beta, sigma,
          d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template int cuinterp1d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp1d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

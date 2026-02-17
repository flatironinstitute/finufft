#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <iostream>

#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

/* --------------------- 1d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
template<typename T, int KEREVALMETH, int ns>
static __global__ void interp_1d_nuptsdriven(const T *x, cuda_complex<T> *c,
                                      const cuda_complex<T> *fw, int M, int nf1, T es_c,
                                      T es_beta, T sigma, const int *idxnupts) {

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const T x_rescaled        = fold_rescale(x[idxnupts[i]], nf1);
    const auto [xstart, xend] = interval(ns, x_rescaled);

    cuda_complex<T> cnow{0, 0};

    const T x1 = (T)xstart - x_rescaled;
    T ker1[ns];
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
    else
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
    for (int xx = xstart; xx <= xend; xx++) {
      int ix            = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
      const T kervalue1 = ker1[xx - xstart];
      cnow.x += fw[ix].x * kervalue1;
      cnow.y += fw[ix].y * kervalue1;
    }
    c[idxnupts[i]] = cnow;
  }
}

template<typename T, int ns>
static int cuinterp1d_nuptsdriven(int nf1, int M, cufinufft_plan_t<T> *d_plan, int blksize) {
  auto &stream = d_plan->stream;
  dim3 threadsPerBlock;
  dim3 blocks;

  T es_c          = 4.0/T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta       = d_plan->spopts.beta;
  T sigma         = d_plan->opts.upsampfac;
  int *d_idxnupts = d_plan->idxnupts;

  T *d_kx               = d_plan->kxyz[0];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x =
      std::min(optimal_block_threads(d_plan->opts.gpu_device_id), (unsigned)M);
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
                                                   d_plan->spopts.nspread, d_plan->nf123[0],
                                                   d_plan->M, d_plan, blksize);
}

template int cuinterp1d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp1d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

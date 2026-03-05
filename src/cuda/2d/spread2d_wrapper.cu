#include <cassert>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cufinufft/common.h>
#include <cufinufft/common_kernels.hpp>
#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

using namespace cufinufft::common;
using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

using cuda::std::dextents;
using cuda::std::dynamic_extent;
using cuda::std::extents;
using cuda::std::mdspan;
using cuda::std::span;
#if 0
template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_2d_output_driven(
    const T *x, const T *y, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1,
    int nf2, T es_c, T es_beta, T sigma, const int *binstartpts, const int *bin_size,
    int bin_size_x, int bin_size_y, const int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, const int *idxnupts,
    const int np) {
  extern __shared__ char sharedbuf[];

  static constexpr auto ns_2f      = T(ns * .5);
  static constexpr auto ns_2       = (ns + 1) / 2;
  static constexpr auto rounded_ns = ns_2 * 2;

  const auto padded_size_x = bin_size_x + rounded_ns;
  const auto padded_size_y = bin_size_y + rounded_ns;

  const int bidx        = subprob_to_bin[blockIdx.x];
  const int binsubp_idx = blockIdx.x - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const int xoffset = (bidx % nbinx) * bin_size_x;
  const int yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;

  using mdspan_t = mdspan<T, extents<int, dynamic_extent, 2, ns>>;
  auto kerevals  = mdspan_t((T *)sharedbuf, np);
  // sharedbuf + size of kerevals in bytes
  // Offset pointer into sharedbuf after kerevals
  // Create span using pointer + size

  auto nupts_sm = span(
      reinterpret_cast<cuda_complex<T> *>(kerevals.data_handle() + kerevals.size()), np);

  auto shift = span(reinterpret_cast<int2 *>(nupts_sm.data() + nupts_sm.size()), np);

  auto local_subgrid = mdspan<cuda_complex<T>, dextents<int, 2>>(
      reinterpret_cast<cuda_complex<T> *>(shift.data() + shift.size()), padded_size_y,
      padded_size_x);

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid.size(); i += blockDim.x) {
    local_subgrid.data_handle()[i] = {0, 0};
  }
  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx = idxnupts[ptstart + i + batch_begin];
      // index of the current point within the batch
      const auto x_rescaled = fold_rescale(x[nuptsidx], nf1);
      const auto y_rescaled = fold_rescale(y[nuptsidx], nf2);
      nupts_sm[i]           = c[nuptsidx];
      auto [xstart, xend]   = interval(ns, x_rescaled);
      auto [ystart, yend]   = interval(ns, y_rescaled);
      const T x1            = T(xstart) - x_rescaled;
      const T y1            = T(ystart) - y_rescaled;

      shift[i] = {
          xstart - xoffset,
          ystart - yoffset,
      };

      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 0, 0), x1, sigma);
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 1, 0), y1, sigma);
      } else {
        eval_kernel_vec<T, ns>(&kerevals(i, 0, 0), x1, es_c, es_beta);
        eval_kernel_vec<T, ns>(&kerevals(i, 1, 0), y1, es_c, es_beta);
      }
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      // strength from shared memory
      static constexpr int sizex  = ns; // true span in X
      const auto cnow             = nupts_sm[i];
      const auto [xstart, ystart] = shift[i];
      static constexpr auto total = ns * ns;

      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // decompose idx using `plane`
        const int yy = idx / sizex;
        const int xx = idx - yy * sizex;

        // recover global coords
        const int real_yy = ystart + yy;
        const int real_xx = xstart + xx;

        // padded indices
        const int iy = real_yy + ns_2;
        const int ix = real_xx + ns_2;

        if constexpr (std::is_same_v<T, float>) {
          if (ix >= (padded_size_x) || ix < 0) break;
          if (iy >= (padded_size_y) || iy < 0) break;
        }
        // separable window weights
        const auto kervalue = kerevals(i, 0, xx) * kerevals(i, 1, yy);

        // accumulate
        local_subgrid(iy, ix) += {cnow * kervalue};
      }
      __syncthreads();
    }
  }
  for (int n = threadIdx.x; n < local_subgrid.size(); n += blockDim.x) {
    const int i = n % (padded_size_x);
    const int j = n / (padded_size_x);

    int ix = xoffset - ns_2 + i;
    int iy = yoffset - ns_2 + j;

    if (ix < (nf1 + ns_2) && iy < (nf2 + ns_2)) {
      ix               = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy               = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      const int outidx = ix + iy * nf1;
      atomicAddComplexGlobal<T>(fw + outidx, local_subgrid(j, i));
    }
  }
}

template<typename T, int ns>
static void cuspread2d_output_driven(int nf1, int nf2, int M,
                                     const cufinufft_plan_t<T> &d_plan, int blksize) {
  auto &stream = d_plan.stream;

  T es_c             = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta          = d_plan.spopts.beta;
  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan.opts.gpu_binsizex;
  int bin_size_y = d_plan.opts.gpu_binsizey;
  int numbins[2];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);

  const T *d_kx              = d_plan.kxyz[0];
  const T *d_ky              = d_plan.kxyz[1];
  const cuda_complex<T> *d_c = d_plan.c;
  cuda_complex<T> *d_fw      = d_plan.fw;

  const int *d_binsize         = dethrust(d_plan.binsize);
  const int *d_binstartpts     = dethrust(d_plan.binstartpts);
  const int *d_numsubprob      = dethrust(d_plan.numsubprob);
  const int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);
  const int *d_idxnupts        = dethrust(d_plan.idxnupts);

  int totalnumsubprob         = d_plan.totalnumsubprob;
  const int *d_subprob_to_bin = dethrust(d_plan.subprob_to_bin);

  T sigma = d_plan.opts.upsampfac;

  const auto sharedplanorysize = shared_memory_required<T>(
      2, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);

  if (d_plan.opts.gpu_kerevalmeth) {
    cufinufft_set_shared_memory(spread_output_driven<T, 1, 2, ns>, 2, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_output_driven<T, 1, 2, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_plan.kxyz, d_c + t * M, d_fw + t * nf1 * nf2, M, d_plan.nf123, es_c, es_beta,
              sigma, d_binstartpts, d_binsize, {bin_size_x, bin_size_y, 1}, d_subprob_to_bin,
              d_subprobstartpts, d_numsubprob, maxsubprobsize, {numbins[0], numbins[1], 1},
              d_idxnupts, d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(spread_output_driven<T, 0, 2, ns>, 2, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_output_driven<T, 0, 2, ns>
          <<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
              d_plan.kxyz, d_c + t * M, d_fw + t * nf1 * nf2, M, d_plan.nf123, es_c, es_beta,
              sigma, d_binstartpts, d_binsize, {bin_size_x, bin_size_y, 1}, d_subprob_to_bin,
              d_subprobstartpts, d_numsubprob, maxsubprobsize, {numbins[0], numbins[1], 1},
              d_idxnupts, d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  }
}
#endif
// Functor to handle function selection (nuptsdriven vs subprob)
struct Spread2DDispatcher {
  template<int ns, typename T>
  void operator()(int nf1, int nf2, int M, const cufinufft_plan_t<T> &d_plan,
                  int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuspread_nupts_driven<T, 2, ns>(d_plan, blksize);
    case 2:
      return cuspread_subprob<T, 2, ns>(d_plan, blksize);
    case 3:
      return cuspread_output_driven<T, 2, ns>(d_plan, blksize);
    default:
      std::cerr << "[cuspread2d] error: incorrect method, should be 1, 2 or 3\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Updated cuspread2d using generic dispatch
template<typename T> void cuspread2d(const cufinufft_plan_t<T> &d_plan, int blksize) {
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
                                            d_plan.nf123[0], d_plan.nf123[1], d_plan.M,
                                            d_plan, blksize);
}
template void cuspread2d<float>(const cufinufft_plan_t<float> &d_plan, int blksize);
template void cuspread2d<double>(const cufinufft_plan_t<double> &d_plan, int blksize);

template<typename T> void cuspread2d_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_method == 1) cuspread_nuptsdriven_prop<T, 2>(d_plan);
  if (d_plan.opts.gpu_method == 2) cuspread_subprob_prop<T, 2>(d_plan);
  if (d_plan.opts.gpu_method == 3) cuspread_subprob_prop<T, 2>(d_plan);
}
template void cuspread2d_prop(cufinufft_plan_t<float> &d_plan);
template void cuspread2d_prop(cufinufft_plan_t<double> &d_plan);
} // namespace spreadinterp
} // namespace cufinufft

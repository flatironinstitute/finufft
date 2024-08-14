#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <thrust/extrema.h>

#include <cufinufft/defs.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

#include <thrust/sort.h>

using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {
/* ------------------------ 1d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

template<typename T, int KEREVALMETH>
__global__ void spread_1d_nuptsdriven(const T *x, const cuda_complex<T> *c,
                                      cuda_complex<T> *fw, int M, int ns, int nf1, T es_c,
                                      T es_beta, T sigma, const int *idxnupts) {
  // dynamic stack allocation to reduce stack usage
#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns);
  auto *__restrict__ ker1 = ker;
#else
  T ker1[MAX_NSPREAD];
#endif

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto x_rescaled     = fold_rescale(x[idxnupts[i]], nf1);
    const auto cnow           = c[idxnupts[i]];
    const auto [xstart, xend] = interval(ns, x_rescaled);
    const T x1                = (T)xstart - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
    else
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);

    for (auto xx = xstart; xx <= xend; xx++) {
      auto ix          = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
      const T kervalue = ker1[xx - xstart];
      atomicAdd(&fw[ix].x, cnow.x * kervalue);
      atomicAdd(&fw[ix].y, cnow.y * kervalue);
    }
  }
}

/* Kernels for SubProb Method */
// SubProb properties
template<typename T>
__global__ void calc_bin_size_noghost_1d(int M, int nf1, int bin_size_x, int nbinx,
                                         int *bin_size, const T *x, int *sortidx) {
  int binx;
  int oldidx;
  T x_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;
    oldidx     = atomicAdd(&bin_size[binx], 1);
    sortidx[i] = oldidx;
    if (binx >= nbinx) {
      sortidx[i] = -binx;
    }
  }
}

template<typename T>
__global__ void calc_inverse_of_global_sort_idx_1d(
    int M, int bin_size_x, int nbinx, const int *bin_startpts, const int *sortidx,
    const T *x, int *index, int nf1) {
  int binx;
  T x_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;

    index[bin_startpts[binx] + sortidx[i]] = i;
  }
}

template<typename T, int KEREVALMETH>
__global__ void spread_1d_subprob(
    const T *x, const cuda_complex<T> *c, cuda_complex<T> *fw, int M, uint8_t ns, int nf1,
    T es_c, T es_beta, T sigma, const int *binstartpts, const int *bin_size,
    int bin_size_x, const int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, int nbinx, int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto *__restrict__ fwshared = (cuda_complex<T> *)sharedbuf;

  const int subpidx     = blockIdx.x;
  const int bidx        = subprob_to_bin[subpidx];
  const int binsubp_idx = subpidx - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts   = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);
  const int xoffset = (bidx % nbinx) * bin_size_x;
  const auto ns_2   = (ns + 1) / 2;
  const int N       = bin_size_x + 2 * ns_2;

  // dynamic stack allocation
#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns);
  auto *__restrict__ ker1 = ker;
#else
  T ker1[MAX_NSPREAD];
#endif

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

  const T ns_2f = ns * T(.5);

  __syncthreads();

  for (auto i = threadIdx.x; i < nupts; i += blockDim.x) {
    const auto idx            = ptstart + i;
    const auto x_rescaled     = fold_rescale(x[idxnupts[idx]], nf1);
    const auto cnow           = c[idxnupts[idx]];
    const auto [xstart, xend] = interval(ns, x_rescaled);
    const T x1                = T(xstart + xoffset) - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
    else
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
    for (int xx = xstart; xx <= xend; xx++) {
      const auto ix = xx + ns_2;
      if (ix >= (bin_size_x + ns_2) || ix < 0) break;
      const cuda_complex<T> result{cnow.x * ker1[xx - xstart],
                                   cnow.y * ker1[xx - xstart]};
      atomicAddComplexShared<T>(fwshared + ix, result);
    }
  }
  __syncthreads();
  /* write to global memory */
  for (int k = threadIdx.x; k < N; k += blockDim.x) {
    auto ix = xoffset - ns_2 + k;
    if (ix < (nf1 + ns_2)) {
      ix = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      atomicAddComplexGlobal<T>(fw + ix, fwshared[k]);
    }
  }
}

/* --------------------- 1d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
template<typename T, int KEREVALMETH>
__global__ void interp_1d_nuptsdriven(const T *x, cuda_complex<T> *c,
                                      const cuda_complex<T> *fw, int M, int ns, int nf1,
                                      T es_c, T es_beta, T sigma, const int *idxnupts) {
  // dynamic stack allocation
#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns);
  auto *__restrict__ ker1 = ker;
#else
  T ker1[MAX_NSPREAD];
#endif
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const T x_rescaled        = fold_rescale(x[idxnupts[i]], nf1);
    const auto [xstart, xend] = interval(ns, x_rescaled);

    cuda_complex<T> cnow{0, 0};

    const T x1 = (T)xstart - x_rescaled;
    if constexpr (KEREVALMETH == 1)
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
    else
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
    for (int xx = xstart; xx <= xend; xx++) {
      int ix            = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
      const T kervalue1 = ker1[xx - xstart];
      cnow.x += fw[ix].x * kervalue1;
      cnow.y += fw[ix].y * kervalue1;
    }
    c[idxnupts[i]] = cnow;
  }
}

} // namespace spreadinterp
} // namespace cufinufft

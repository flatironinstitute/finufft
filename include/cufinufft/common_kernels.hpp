#pragma once

#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/utils.h>
#include <cufinufft/intrinsics.h>

namespace cufinufft {
namespace spreadinterp {

using namespace cufinufft::utils;

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void interp_nupts_driven(
    const cuda::std::array<const T *, 3> xyz, cuda_complex<T> *c,
    const cuda_complex<T> *fw, int M, const cuda::std::array<int, 3> nf, T es_c,
    T es_beta, T sigma, const int *idxnupts) {

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(idxnupts + i);

    cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
    cuda::std::array<int, ndim> start;
    for (size_t idim = 0; idim < ndim; ++idim) {
      auto rescaled   = fold_rescale(loadReadOnly(xyz[idim] + nuptsidx), nf[idim]);
      auto [s, dummy] = interval(ns, rescaled);
      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&ker[idim][0], T(s) - rescaled, sigma);
      } else {
        eval_kernel_vec<T, ns>(&ker[idim][0], T(s) - rescaled, es_c, es_beta);
      }
      start[idim] = s + ((s < 0) ? nf[idim] : 0);
    }

    cuda_complex<T> cnow{0, 0};
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns; ++x0, ix = (ix + 1 >= nf[0]) ? 0 : ix + 1)
        cnow += fw[ix] * ker[0][x0];
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= nf[1]) ? 0 : iy + 1) {
        const auto inidx0 = iy * nf[0];
        cuda_complex<T> cnowx{0, 0};
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= nf[0]) ? 0 : ix + 1)
          cnowx += fw[inidx0 + ix] * ker[0][x0];
        cnow += cnowx * ker[1][y0];
      }
    } else {
      cuda::std::array<int, ns> xidx;
      for (int x0 = 0, ix = start[0]; x0 < ns; ++x0, ix = (ix + 1 >= nf[0]) ? 0 : ix + 1)
        xidx[x0] = ix;
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= nf[2]) ? 0 : iz + 1) {
        const auto inidx0 = iz * nf[1] * nf[0];
        cuda_complex<T> cnowy{0, 0};
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= nf[1]) ? 0 : iy + 1) {
          const auto inidx1 = inidx0 + iy * nf[0];
          cuda_complex<T> cnowx{0, 0};
          for (int x0 = 0; x0 < ns; ++x0) cnowx += fw[inidx1 + xidx[x0]] * ker[0][x0];
          cnowy += cnowx * ker[1][y0];
        }
        cnow += cnowy * ker[2][z0];
      }
    }
    c[idxnupts[i]] = cnow;
  }
}

template<typename T, int ndim, int ns>
void cuinterp_nuptsdriven(const cufinufft_plan_t<T> &d_plan, int blksize) {
  T es_c    = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta = d_plan.spopts.beta;
  T sigma   = d_plan.spopts.upsampfac;

  const int *d_idxnupts = dethrust(d_plan.idxnupts);

  const dim3 threadsPerBlock{
      std::min(optimal_block_threads(d_plan.opts.gpu_device_id), (unsigned)d_plan.M), 1u,
      1u};
  const dim3 blocks{(d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1};

  if (d_plan.opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      interp_nupts_driven<T, 1, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      interp_nupts_driven<T, 0, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

} // namespace spreadinterp
} // namespace cufinufft


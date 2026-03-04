#pragma once

#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/utils.h>
#include <cufinufft/intrinsics.h>
#include <cufinufft/common.h>

namespace cufinufft {
namespace spreadinterp {

using namespace cufinufft::utils;
using namespace cufinufft::common;

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

/* Kernels for SubProb Method */
template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ void interp_subprob(
    cuda::std::array<const T *, 3> xyz, cuda_complex<T> *c, const cuda_complex<T> *fw,
    int M, cuda::std::array<int, 3> nf, T es_c, T es_beta, T sigma,
    const int *binstartpts, const int *bin_size, cuda::std::array<int, 3> binsizes, const int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, cuda::std::array<int, 3> nbins,
    const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;

  const auto subpidx     = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

//FIXME: loop!
  cuda::std::array<int, 3> offset;
  {
  int tmp = bidx;
  for (int idim=0; idim<ndim; ++idim) {
    offset[idim] = (tmp%nbins[idim]) * binsizes[idim];
    tmp /= nbins[idim];
  }
  }
//  offset[0] = (bidx % nbins[0]) * binsizes[0];
//  offset[1] = ((bidx / nbins[0]) % nbins[1]) * binsizes[1];
//  offset[2] = (bidx / (nbins[0] * nbins[1])) * binsizes[2];

  const T ns_2f         = ns * T(.5);
  const auto ns_2       = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;

  int N = 1;
  for (int idim=0; idim<ndim; ++idim)
    N *= binsizes[idim]+rounded_ns;

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    bool in_region = true;
    int flatidx = n;
    int inidx = 0, sharedidx = 0;
    int instride = 1, sharedstride = 1;
    for (int idim=0; idim<ndim; ++idim) {
      int idx0 = flatidx%(binsizes[idim]+rounded_ns);
      int idx = idx0 + offset[idim] - ns_2;
      if (idx >= nf[idim] + ns_2) in_region = false;
      idx = idx < 0 ? idx + nf[idim] : (idx >= nf[idim] ? idx - nf[idim] : idx);
      inidx += idx*instride;
      instride *= nf[idim];
      sharedidx += idx0*sharedstride;
      sharedstride *= (binsizes[idim]+rounded_ns);
      flatidx /= (binsizes[idim]+rounded_ns);
    }
    if (in_region)
      fwshared[sharedidx] = fw[inidx];
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx = ptstart + i;
    cuda::std::array<int, 3> start;
    for (int idim=0; idim<ndim; ++idim) {
      const auto rescaled = fold_rescale(xyz[idim][idxnupts[idx]], nf[idim]);
      auto [s, dummy] = interval(ns, rescaled);

      const T s1 = T(s) - rescaled;

      s -= offset[idim];
      start[idim] = s;

      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&ker[idim][0], s1, sigma);
      } else {
        eval_kernel_vec<T, ns>(&ker[idim][0], s1, es_c, es_beta);
      }
    }

    cuda_complex<T> cnow{0, 0};
    for (int zz = 0; zz < ns; ++zz) {
      const auto kervalue3 = ker[2][zz];
      const auto iz = zz + start[2] + ns_2;
      for (int yy = 0; yy < ns; ++yy) {
        const auto kervalue2 = ker[1][yy];
        const auto iy = yy + start[1] + ns_2;
        for (int xx = 0; xx < ns; ++xx) {
          const auto ix = xx + start[0] + ns_2;
          const auto outidx = ix + iy * (binsizes[0] + rounded_ns) +
                              iz * (binsizes[0] + rounded_ns) * (binsizes[1] + rounded_ns);
          const auto kervalue1 = ker[0][xx];
          const auto kervalue  = kervalue1 * kervalue2 * kervalue3;
          cnow += {fwshared[outidx] * kervalue};
        }
      }
    }
    c[idxnupts[idx]] = cnow;
  }
}

template<typename T, int ndim, int ns>
static void cuinterp_subprob(const cufinufft_plan_t<T> &d_plan, int blksize) {
  auto &stream = d_plan.stream;

  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  cuda::std::array<int, 3> binsizes {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};

  cuda::std::array<int, 3> numbins;
  for (int idim=0; idim<ndim; ++idim)
    numbins[idim] = ceil((T)d_plan.nf123[idim] / binsizes[idim]);

  const int *d_binsize         = dethrust(d_plan.binsize);
  const int *d_binstartpts     = dethrust(d_plan.binstartpts);
  const int *d_numsubprob      = dethrust(d_plan.numsubprob);
  const int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);
  const int *d_idxnupts        = dethrust(d_plan.idxnupts);
  const int *d_subprob_to_bin  = dethrust(d_plan.subprob_to_bin);
  int totalnumsubprob          = d_plan.totalnumsubprob;

  T sigma                      = d_plan.spopts.upsampfac;
  T es_c                       = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta                    = d_plan.spopts.beta;
  const auto sharedplanorysize = shared_memory_required<T>(
      3, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);

  if (d_plan.opts.gpu_kerevalmeth == 1) {
    cufinufft_set_shared_memory(interp_subprob<T, 1, 3, ns>, 3, d_plan);
    for (int t = 0; t < blksize; t++) {
      interp_subprob<T, 1, 3, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_binstartpts, d_binsize, binsizes,
          d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(interp_subprob<T, 0, 3, ns>, 3, d_plan);
    for (int t = 0; t < blksize; t++) {
      interp_subprob<T, 0, 3, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_binstartpts, d_binsize, binsizes,
          d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

} // namespace spreadinterp
} // namespace cufinufft


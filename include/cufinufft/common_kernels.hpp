#pragma once

#include <thrust/sequence.h>

#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/utils.h>
#include <cufinufft/intrinsics.h>
#include <cufinufft/common.h>

namespace cufinufft {
namespace spreadinterp {

using namespace cufinufft::utils;
using namespace cufinufft::common;

template<int ndim, typename T> inline auto get_nbin_info(const cufinufft_plan_t<T> &plan, cuda::std::array<int, 3> binsizes) {
  cuda::std::array<int, 3> nbins{1,1,1};
  int nbins_tot=1;
  for (int idim=0; idim<ndim; ++idim) {
    if (binsizes[idim] < 0) {
      std::cerr << "[cuspread_nuptsdriven_prop] error: invalid binsize (dim "<<idim<<") = ("
                << binsizes[idim] << ")\n";
      throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
    }
    nbins[idim] = ceil(T(plan.nf123[idim]) / binsizes[idim]);
    nbins_tot *= nbins[idim];
  }
  return std::make_tuple(nbins, nbins_tot);
}

template<typename T, int KEREVALMETH, int ndim, int ns> __device__ inline auto
  get_kerval_and_startpos_subprob(int idx, cuda::std::array<const T *,3> xyz, cuda::std::array<int,3> nf, cuda::std::array<int,ndim> offset, T sigma, T es_c, T es_beta) {
  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
  cuda::std::array<int, ndim> start;
  for (int idim=0; idim<ndim; ++idim) {
    const auto rescaled = fold_rescale(xyz[idim][idx], nf[idim]);
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
  return make_tuple(ker,start);
}

template<typename T, int KEREVALMETH, int ndim, int ns> __device__ inline auto
  get_kerval_and_startpos_nuptsdriven(int idx, cuda::std::array<const T *,3> xyz, cuda::std::array<int,3> nf, T sigma, T es_c, T es_beta) {
  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
  cuda::std::array<int, ndim> start;
  for (size_t idim = 0; idim < ndim; ++idim) {
    auto rescaled   = fold_rescale(loadReadOnly(xyz[idim] + idx), nf[idim]);
    auto [s, dummy] = interval(ns, rescaled);
    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(&ker[idim][0], T(s) - rescaled, sigma);
    } else {
      eval_kernel_vec<T, ns>(&ker[idim][0], T(s) - rescaled, es_c, es_beta);
    }
    start[idim] = s + ((s < 0) ? nf[idim] : 0);
  }
  return make_tuple(ker,start);
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void interp_nupts_driven(
    const cuda::std::array<const T *, 3> xyz, cuda_complex<T> *c,
    const cuda_complex<T> *fw, int M, const cuda::std::array<int, 3> nf, T es_c,
    T es_beta, T sigma, const int *idxnupts) {

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(idxnupts + i);

    auto [ker, start] = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(nuptsidx, xyz, nf, sigma, es_c, es_beta);

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
__global__ void interp_subprob(
    cuda::std::array<const T *, 3> xyz, cuda_complex<T> *c, const cuda_complex<T> *fw,
    int M, cuda::std::array<int, 3> nf, T es_c, T es_beta, T sigma,
    const int *binstartpts, const int *bin_size, cuda::std::array<int, 3> binsizes, const int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, cuda::std::array<int, 3> nbins,
    const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  const auto subpidx     = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  cuda::std::array<int, ndim> offset;
  {
  int tmp = bidx;
  for (int idim=0; idim<ndim; ++idim) {
    offset[idim] = (tmp%nbins[idim]) * binsizes[idim];
    tmp /= nbins[idim];
  }
  }

  const T ns_2f         = ns * T(.5);
  const auto ns_2       = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;

  int N = 1;
  for (int idim=0; idim<ndim; ++idim)
    N *= binsizes[idim]+rounded_ns;

  for (int n = threadIdx.x; n < N; n += blockDim.x) {

//FIXME: spreadidx == n by construction?
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
    auto [ker, start] = get_kerval_and_startpos_subprob<T, KEREVALMETH, ndim, ns>(idxnupts[idx], xyz, nf, offset, sigma, es_c, es_beta);

    cuda_complex<T> cnow{0, 0};
    if constexpr (ndim==1) {
      for (int xx = 0; xx < ns; ++xx) {
        const auto ix = xx + start[0] + ns_2;
        cnow += {fwshared[ix] * ker[0][xx]};
      }
    } else if constexpr (ndim==2) {
      for (int yy = 0; yy < ns; ++yy) {
        const auto kervalue2 = ker[1][yy];
        const auto iy = yy + start[1] + ns_2;
        for (int xx = 0; xx < ns; ++xx) {
          const auto ix = xx + start[0] + ns_2;
          const auto inidx = ix + iy * (binsizes[0] + rounded_ns);
          const auto kervalue1 = ker[0][xx];
          const auto kervalue  = kervalue1 * kervalue2;
          cnow += {fwshared[inidx] * kervalue};
        }
      }
    } else {
      for (int zz = 0; zz < ns; ++zz) {
        const auto kervalue3 = ker[2][zz];
        const auto iz = zz + start[2] + ns_2;
        for (int yy = 0; yy < ns; ++yy) {
          const auto kervalue2 = ker[1][yy];
          const auto iy = yy + start[1] + ns_2;
          for (int xx = 0; xx < ns; ++xx) {
            const auto ix = xx + start[0] + ns_2;
            const auto inidx = ix + iy * (binsizes[0] + rounded_ns) +
                               iz * (binsizes[0] + rounded_ns) * (binsizes[1] + rounded_ns);
            const auto kervalue1 = ker[0][xx];
            const auto kervalue  = kervalue1 * kervalue2 * kervalue3;
            cnow += {fwshared[inidx] * kervalue};
          }
        }
      }
    }
    c[idxnupts[idx]] = cnow;
  }
}

template<typename T, int ndim, int ns>
void cuinterp_subprob(const cufinufft_plan_t<T> &d_plan, int blksize) {
  auto &stream = d_plan.stream;

  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;

  // assume that bin_size > ns/2;
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
      ndim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);

  if (d_plan.opts.gpu_kerevalmeth == 1) {
    cufinufft_set_shared_memory(interp_subprob<T, 1, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      interp_subprob<T, 1, ndim, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_binstartpts, d_binsize, binsizes,
          d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(interp_subprob<T, 0, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      interp_subprob<T, 0, ndim, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_binstartpts, d_binsize, binsizes,
          d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
          maxsubprobsize, numbins, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void spread_nupts_driven(
    cuda::std::array<const T *, 3> xyz, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, cuda::std::array<int, 3> nf, T es_c, T es_beta, T sigma, const int *idxnupts) {

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(idxnupts + i);
    auto [ker, start] = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(nuptsidx, xyz, nf, sigma, es_c, es_beta);

    cuda_complex<T> val = c[idxnupts[i]];
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns; ++x0, ix = (ix + 1 >= nf[0]) ? 0 : ix + 1)
        atomicAddComplexGlobal<T>(fw+ix, ker[0][x0]*val);
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= nf[1]) ? 0 : iy + 1) {
        const auto outidx0 = iy * nf[0];
        cuda_complex<T> valy = val*ker[1][y0];
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= nf[0]) ? 0 : ix + 1)
          atomicAddComplexGlobal<T>(fw+outidx0+ix, ker[0][x0]*valy);
      }
    } else {
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= nf[2]) ? 0 : iz + 1) {
        const auto outidx0 = iz * nf[1] * nf[0];
        cuda_complex<T> valz = val*ker[2][z0];
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= nf[1]) ? 0 : iy + 1) {
          const auto outidx1 = outidx0 + iy * nf[0];
          cuda_complex<T> valy = valz*ker[1][y0];
          for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= nf[0]) ? 0 : ix + 1) {
            atomicAddComplexGlobal<T>(fw+outidx1+ix, ker[0][x0]*valy);
          }
        }
      }
    }
  }
}

template<typename T, int ndim, int ns>
void cuspread_nupts_driven(const cufinufft_plan_t<T> &d_plan, int blksize) {
  auto &stream = d_plan.stream;

  T sigma   = d_plan.spopts.upsampfac;
  T es_c    = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta = d_plan.spopts.beta;

  const int *d_idxnupts      = dethrust(d_plan.idxnupts);

  dim3 threadsPerBlock;
  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  dim3 blocks;
  blocks.x = (d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y = 1;

  if (d_plan.opts.gpu_kerevalmeth == 1) {
    for (int t = 0; t < blksize; t++) {
      spread_nupts_driven<T, 1, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M, d_plan.nf123,
          es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      spread_nupts_driven<T, 0, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M, d_plan.nf123,
          es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

template<typename T, int ndim>
__global__ void calc_bin_size_noghost(
    int M, cuda::std::array<int, 3> nf, cuda::std::array<int, 3> binsizes,
    cuda::std::array<int, 3> nbins, int *bin_size, cuda::std::array<const T *, 3> xyz,
    int *sortidx) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    int binidx = 0;
    int stride = 1;
    for (int idim=0; idim<ndim; ++idim) {
      T rescaled = fold_rescale(xyz[idim][i], nf[idim]);
      int bin = floor(rescaled / binsizes[idim]);
      bin = bin >= nbins[idim] ? bin - 1 : bin;
      bin = bin < 0 ? 0 : bin;
      binidx += bin*stride;
      stride *= nbins[idim];
    }
    int oldidx = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename T, int ndim>
__global__ void calc_inverse_of_global_sort_idx(
    int M, cuda::std::array<int, 3> binsizes, cuda::std::array<int, 3> nbins, const int *bin_startpts, const int *sortidx,
    cuda::std::array<const T *, 3> xyz, int *index, cuda::std::array<int, 3> nf) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    int binidx = 0;
    int stride = 1;
    for (int idim=0; idim<ndim; ++idim) {
      T rescaled = fold_rescale(xyz[idim][i], nf[idim]);
      int bin = floor(rescaled / binsizes[idim]);
      bin = bin >= nbins[idim] ? bin - 1 : bin;
      bin = bin < 0 ? 0 : bin;
      binidx += bin*stride;
      stride *= nbins[idim];
    }
    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

template<typename T, int ndim>
void cuspread_nuptsdriven_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_sort) {
    cuda::std::array<int,3> binsizes = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};

    auto [nbins, nbins_tot] = get_nbin_info<ndim>(d_plan, binsizes);

    checkCudaErrors(cudaMemsetAsync(dethrust(d_plan.binsize), 0, nbins_tot * sizeof(int), d_plan.stream));
    calc_bin_size_noghost<T,ndim><<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
        d_plan.M, d_plan.nf123, binsizes, nbins, dethrust(d_plan.binsize), d_plan.kxyz, dethrust(d_plan.sortidx));
    THROW_IF_CUDA_ERROR

    thrust::exclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.binsize.begin(), d_plan.binsize.end(), d_plan.binstartpts.begin());
    THROW_IF_CUDA_ERROR

    calc_inverse_of_global_sort_idx<T,ndim> <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
        d_plan.M, binsizes, nbins, dethrust(d_plan.binstartpts), dethrust(d_plan.sortidx), d_plan.kxyz, dethrust(d_plan.idxnupts), d_plan.nf123);
    THROW_IF_CUDA_ERROR
  } else {
    int *d_idxnupts = dethrust(d_plan.idxnupts);
    thrust::sequence(thrust::cuda::par.on(d_plan.stream), d_plan.idxnupts.begin(), d_plan.idxnupts.begin() + d_plan.M);
    THROW_IF_CUDA_ERROR
  }
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void spread_subprob(
    cuda::std::array<const T *, 3> xyz, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, cuda::std::array<int, 3> nf, T sigma, T es_c, T es_beta, const int *binstartpts,
    const int *bin_size, cuda::std::array<int, 3> binsizes,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, cuda::std::array<int, 3> nbins, const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  const auto subpidx     = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  cuda::std::array<int, ndim> offset;
  {
  int tmp = bidx;
  for (int idim=0; idim<ndim; ++idim) {
    offset[idim] = (tmp%nbins[idim]) * binsizes[idim];
    tmp /= nbins[idim];
  }
  }

  const T ns_2f         = ns * T(.5);
  const auto ns_2       = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;

  int N = 1;
  for (int idim=0; idim<ndim; ++idim)
    N *= binsizes[idim]+rounded_ns;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx = ptstart + i;
    auto [ker, start] = get_kerval_and_startpos_subprob<T, KEREVALMETH, ndim, ns>(idxnupts[idx], xyz, nf, offset, sigma, es_c, es_beta);

    const auto cnow = c[idxnupts[idx]];
    if constexpr (ndim==1) {
      for (int xx = 0; xx < ns; ++xx) {
        const auto ix = xx + start[0] + ns_2;
        atomicAddComplexShared<T>(fwshared + ix, cnow*ker[0][xx]);
      }
    } else if constexpr (ndim==2) {
      for (int yy = 0; yy < ns; ++yy) {
        const auto iy = yy + start[1] + ns_2;
        for (int xx = 0; xx < ns; ++xx) {
          const auto ix = xx + start[0] + ns_2;
          const auto outidx = ix + iy * (binsizes[0] + rounded_ns);
          atomicAddComplexShared<T>(fwshared + outidx, cnow*(ker[0][xx]*ker[1][yy]));
        }
      }
    } else {
      for (int zz = 0; zz < ns; ++zz) {
        const auto kervalue3 = ker[2][zz];
        const auto iz = zz + start[2] + ns_2;
        for (int yy = 0; yy < ns; ++yy) {
          const auto iy = yy + start[1] + ns_2;
          for (int xx = 0; xx < ns; ++xx) {
            const auto ix = xx + start[0] + ns_2;
            const auto outidx = ix + iy * (binsizes[0] + rounded_ns) +
                                iz * (binsizes[0] + rounded_ns) * (binsizes[1] + rounded_ns);
            atomicAddComplexShared<T>(fwshared + outidx, cnow*(ker[0][xx]*ker[1][yy]*ker[2][zz]));
          }
        }
      }
    }
  }

  __syncthreads();

  /* write to global memory */
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    bool in_region = true;
    int flatidx = n;
    int outidx = 0, sharedidx = 0;
    int outstride = 1, sharedstride = 1;
    for (int idim=0; idim<ndim; ++idim) {
      int idx0 = flatidx%(binsizes[idim]+rounded_ns);
      int idx = idx0 + offset[idim] - ns_2;
      if (idx >= nf[idim] + ns_2) in_region = false;
      idx = idx < 0 ? idx + nf[idim] : (idx >= nf[idim] ? idx - nf[idim] : idx);
      outidx += idx*outstride;
      outstride *= nf[idim];
      sharedidx += idx0*sharedstride;
      sharedstride *= (binsizes[idim]+rounded_ns);
      flatidx /= (binsizes[idim]+rounded_ns);
    }
    if (in_region)
      atomicAddComplexGlobal<T>(fw + outidx, fwshared[sharedidx]);
  }
}

template<typename T, int ndim, int ns>
static void cuspread_subprob(const cufinufft_plan_t<T> &d_plan, int blksize) {

  // assume that bin_size > ns/2;
  cuda::std::array<int, 3> binsizes = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};
  auto [nbins, nbins_tot] = get_nbin_info<ndim>(d_plan, binsizes);

  T sigma                      = d_plan.spopts.upsampfac;
  T es_c                       = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta                    = d_plan.spopts.beta;
  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);
  if (d_plan.opts.gpu_kerevalmeth) {
    cufinufft_set_shared_memory(spread_subprob<T, 1, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_subprob<T, 1, ndim, ns><<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M, d_plan.nf123,
          sigma, es_c, es_beta, dethrust(d_plan.binstartpts), dethrust(d_plan.binsize), binsizes,
          dethrust(d_plan.subprob_to_bin), dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob), d_plan.opts.gpu_maxsubprobsize,
          nbins, dethrust(d_plan.idxnupts));
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(spread_subprob<T, 0, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_subprob<T, 1, ndim, ns><<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M, d_plan.nf123,
          sigma, es_c, es_beta, dethrust(d_plan.binstartpts), dethrust(d_plan.binsize), binsizes,
          dethrust(d_plan.subprob_to_bin), dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob), d_plan.opts.gpu_maxsubprobsize,
          nbins, dethrust(d_plan.idxnupts));
      THROW_IF_CUDA_ERROR
    }
  }
}


static __global__ void calc_subprob(const int *bin_size, int *num_subprob,
                                       int maxsubprobsize, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    num_subprob[i] = ceil(bin_size[i] / (float)maxsubprobsize);
  }
}
static __global__ void map_b_into_subprob(int *d_subprob_to_bin,
                                          const int *d_subprobstartpts,
                                          const int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d_numsubprob[i]; j++) {
      d_subprob_to_bin[d_subprobstartpts[i] + j] = i;
    }
  }
}

template<typename T, int ndim> static void cuspread_subprob_prop(cufinufft_plan_t<T> &d_plan) {
  cuda::std::array<int,3> binsizes = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};

  auto [nbins, nbins_tot] = get_nbin_info<ndim>(d_plan, binsizes);

  checkCudaErrors(cudaMemsetAsync(dethrust(d_plan.binsize), 0, nbins_tot * sizeof(int), d_plan.stream));
  calc_bin_size_noghost<T,ndim><<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
      d_plan.M, d_plan.nf123, binsizes, nbins, dethrust(d_plan.binsize), d_plan.kxyz, dethrust(d_plan.sortidx));
  THROW_IF_CUDA_ERROR
  thrust::exclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.binsize.begin(), d_plan.binsize.end(), d_plan.binstartpts.begin());
  THROW_IF_CUDA_ERROR
  calc_inverse_of_global_sort_idx<T,ndim> <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
      d_plan.M, binsizes, nbins, dethrust(d_plan.binstartpts), dethrust(d_plan.sortidx), d_plan.kxyz, dethrust(d_plan.idxnupts), d_plan.nf123);
  THROW_IF_CUDA_ERROR

  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  calc_subprob<<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
      dethrust(d_plan.binsize), dethrust(d_plan.numsubprob), d_plan.opts.gpu_maxsubprobsize, nbins_tot);
  THROW_IF_CUDA_ERROR

  thrust::inclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.numsubprob.begin(), d_plan.numsubprob.begin() + nbins_tot, d_plan.subprobstartpts.begin()+1);
  THROW_IF_CUDA_ERROR

  int totalnumsubprob;
  checkCudaErrors(cudaMemsetAsync(dethrust(d_plan.subprobstartpts), 0, sizeof(int), d_plan.stream));
  checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &(dethrust(d_plan.subprobstartpts)[nbins_tot]), sizeof(int),
                                  cudaMemcpyDeviceToHost, d_plan.stream));
  cudaStreamSynchronize(d_plan.stream);
  d_plan.subprob_to_bin.resize(totalnumsubprob);

  map_b_into_subprob<<<(nbins[0] * nbins[1] + 1024 - 1) / 1024, 1024, 0,
                        d_plan.stream>>>(dethrust(d_plan.subprob_to_bin), dethrust(d_plan.subprobstartpts),
                                 dethrust(d_plan.numsubprob),
                                 nbins_tot);

  d_plan.totalnumsubprob = totalnumsubprob;
}

} // namespace spreadinterp
} // namespace cufinufft


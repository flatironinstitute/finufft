#pragma once

#include <thrust/sequence.h>

#include <cufinufft/common.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>
#include <cufinufft/intrinsics.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

namespace cufinufft {
namespace spreadinterp {

using namespace cufinufft::utils;
using namespace cufinufft::common;

template<int ndim, typename T>
inline auto get_nbin_info(const cufinufft_plan_t<T> &plan,
                          cuda::std::array<int, 3> binsizes) {
  cuda::std::array<int, 3> nbins{1, 1, 1};
  int nbins_tot = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    if (binsizes[idim] < 0) {
      std::cerr << "[cuspread_nuptsdriven_prop] error: invalid binsize (dim " << idim
                << ") = (" << binsizes[idim] << ")\n";
      throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
    }
    nbins[idim] = ceil(T(plan.nf123[idim]) / binsizes[idim]);
    nbins_tot *= nbins[idim];
  }
  return std::make_tuple(nbins, nbins_tot);
}

template<typename T, int KEREVALMETH, int ndim, int ns>
inline __device__ auto get_kerval_and_startpos_subprob(
    int idx, cuda::std::array<const T *, 3> xyz, cuda::std::array<int, 3> nf,
    cuda::std::array<int, ndim> offset, T sigma, T es_c, T es_beta) {
  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
  cuda::std::array<int, ndim> start;
  for (int idim = 0; idim < ndim; ++idim) {
    const auto rescaled = fold_rescale(xyz[idim][idx], nf[idim]);
    auto [s, dummy]     = interval(ns, rescaled);

    const T s1 = T(s) - rescaled;

    s -= offset[idim];
    start[idim] = s;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(&ker[idim][0], s1, sigma);
    } else {
      eval_kernel_vec<T, ns>(&ker[idim][0], s1, es_c, es_beta);
    }
  }
  return make_tuple(ker, start);
}

template<typename T, int KEREVALMETH, int ndim, int ns>
inline __device__ auto get_kerval_and_startpos_nuptsdriven(
    int idx, cuda::std::array<const T *, 3> xyz, cuda::std::array<int, 3> nf, T sigma,
    T es_c, T es_beta) {
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
  return make_tuple(ker, start);
}

template<int ndim>
__device__ auto compute_offset(int bidx, const cuda::std::array<int, 3> &nbins,
                               const cuda::std::array<int, 3> &binsizes) {
  cuda::std::array<int, ndim> offset;
  int tmp = bidx;
  for (int idim = 0; idim + 1 < ndim; ++idim) {
    offset[idim] = (tmp % nbins[idim]) * binsizes[idim];
    tmp /= nbins[idim];
  }
  // last dimension can be done more cheaply
  offset[ndim - 1] = tmp * binsizes[ndim - 1];
  return offset;
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void interp_nupts_driven(cufinufft_gpu_data<T> p) {
  T es_c    = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta = p.spopts.beta;
  T sigma   = p.spopts.upsampfac;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < p.M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(p.idxnupts + i);

    auto [ker, start] = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, sigma, es_c, es_beta);

    cuda_complex<T> cnow{0, 0};
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns; ++x0, ix = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        cnow += p.fw[ix] * ker[0][x0];
    }
    if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
        const auto inidx0 = iy * p.nf123[0];
        cuda_complex<T> cnowx{0, 0};
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
          cnowx += p.fw[inidx0 + ix] * ker[0][x0];
        cnow += cnowx * ker[1][y0];
      }
    }
    if constexpr (ndim == 3) {
      cuda::std::array<int, ns> xidx;
      for (int x0 = 0, ix = start[0]; x0 < ns; ++x0, ix = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        xidx[x0] = ix;
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= p.nf123[2]) ? 0 : iz + 1) {
        const auto inidx0 = iz * p.nf123[1] * p.nf123[0];
        cuda_complex<T> cnowy{0, 0};
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
          const auto inidx1 = inidx0 + iy * p.nf123[0];
          cuda_complex<T> cnowx{0, 0};
          for (int x0 = 0; x0 < ns; ++x0) cnowx += p.fw[inidx1 + xidx[x0]] * ker[0][x0];
          cnowy += cnowx * ker[1][y0];
        }
        cnow += cnowy * ker[2][z0];
      }
    }
    p.c[p.idxnupts[i]] = cnow;
  }
}

template<typename T, int ndim, int ns>
void cuinterp_nuptsdriven(const cufinufft_plan_t<T> &d_plan, int blksize) {
  const dim3 threadsPerBlock{
      std::min(optimal_block_threads(d_plan.opts.gpu_device_id), (unsigned)d_plan.M), 1u,
      1u};
  const dim3 blocks{(d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1};

  for (int t = 0; t < blksize; t++) {
    if (d_plan.opts.gpu_kerevalmeth)
      interp_nupts_driven<T, 1, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan);
    else
      interp_nupts_driven<T, 0, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan);
  THROW_IF_CUDA_ERROR
  }
}

template<typename T, int ndim, int ns, typename Func>
inline __device__ void shared_mem_copy_helper(cuda::std::array<int, 3> binsizes,
                                              cuda::std::array<int, ndim> offset,
                                              cuda::std::array<int, 3> nf, Func func) {
  constexpr T ns_2f         = ns * T(.5);
  constexpr auto ns_2       = (ns + 1) / 2;
  constexpr auto rounded_ns = ns_2 * 2;

  int N = 1;
  for (int idim = 0; idim < ndim; ++idim) N *= binsizes[idim] + rounded_ns;

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    bool in_region = true;
    int flatidx    = n;
    int globidx    = 0;
    int globstride = 1;
    for (int idim = 0; idim < ndim; ++idim) {
      int idx0 = flatidx % (binsizes[idim] + rounded_ns);
      int idx  = idx0 + offset[idim] - ns_2;
      if (idx >= nf[idim] + ns_2) in_region = false;
      idx = idx < 0 ? idx + nf[idim] : (idx >= nf[idim] ? idx - nf[idim] : idx);
      globidx += idx * globstride;
      globstride *= nf[idim];
      flatidx /= (binsizes[idim] + rounded_ns);
    }
    if (in_region)
      func(n, globidx); // atomicAddComplexGlobal<T>(fw + outidx, fwshared[n]);
  }
}

/* Kernels for SubProb Method */
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void interp_subprob(
    cuda::std::array<int, 3> binsizes,
    cuda::std::array<int, 3> nbins, cufinufft_gpu_data<T> p) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  T sigma   = p.spopts.upsampfac;
  T es_c    = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta = p.spopts.beta;

  const auto subpidx     = blockIdx.x;
  const auto bidx        = p.subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - p.subprobstartpts[bidx];
  const auto ptstart     = p.binstartpts[bidx] + binsubp_idx * p.opts.gpu_maxsubprobsize;
  const auto nupts = min(p.opts.gpu_maxsubprobsize, p.binsize[bidx] - binsubp_idx * p.opts.gpu_maxsubprobsize);

  auto offset = compute_offset<ndim>(bidx, nbins, binsizes);

  constexpr auto ns_2       = (ns + 1) / 2;
  constexpr auto rounded_ns = ns_2 * 2;

  shared_mem_copy_helper<T, ndim, ns>(binsizes, offset, p.nf123,
                                      [&p, &fwshared](int idx_shared, int idx_global) {
                                        fwshared[idx_shared] = p.fw[idx_global];
                                      });
  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx     = ptstart + i;
    auto [ker, start] = get_kerval_and_startpos_subprob<T, KEREVALMETH, ndim, ns>(
        p.idxnupts[idx], p.xyz, p.nf123, offset, sigma, es_c, es_beta);

    cuda_complex<T> cnow{0, 0};
    if constexpr (ndim == 1) {
      const auto ofs0 = start[0] + ns_2;
      for (int xx = 0; xx < ns; ++xx) cnow += {fwshared[ofs0 + xx] * ker[0][xx]};
    }
    if constexpr (ndim == 2) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto ofs0    = (start[1] + ns_2) * delta_y + (start[0] + ns_2);
      for (int yy = 0; yy < ns; ++yy) {
        cuda_complex<T> cnowy{0, 0};
        const auto ofs = ofs0 + yy * delta_y;
        for (int xx = 0; xx < ns; ++xx) {
          cnowy += fwshared[ofs + xx] * ker[0][xx];
        }
        cnow += cnowy * ker[1][yy];
      }
    }
    if constexpr (ndim == 3) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto delta_z = delta_y * (binsizes[1] + rounded_ns);
      const auto ofs0 =
          (start[2] + ns_2) * delta_z + (start[1] + ns_2) * delta_y + (start[0] + ns_2);
      for (int zz = 0; zz < ns; ++zz) {
        cuda_complex<T> cnowz{0, 0};
        const auto ofs1 = ofs0 + zz * delta_z;
        for (int yy = 0; yy < ns; ++yy) {
          cuda_complex<T> cnowy{0, 0};
          const auto ofs = ofs1 + yy * delta_y;
          for (int xx = 0; xx < ns; ++xx) {
            cnowy += {fwshared[ofs + xx] * ker[0][xx]};
          }
          cnowz += cnowy * ker[1][yy];
        }
        cnow += cnowz * ker[2][zz];
      }
    }
    p.c[p.idxnupts[idx]] = cnow;
  }
}

template<typename T, int ndim, int ns>
void cuinterp_subprob(const cufinufft_plan_t<T> &d_plan, int blksize) {
  // assume that bin_size > ns/2;
  cuda::std::array<int, 3> binsizes{d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
                                    d_plan.opts.gpu_binsizez};

  cuda::std::array<int, 3> numbins;
  for (int idim = 0; idim < ndim; ++idim)
    numbins[idim] = ceil((T)d_plan.nf123[idim] / binsizes[idim]);

  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);

  if (d_plan.opts.gpu_kerevalmeth == 1) {
    cufinufft_set_shared_memory(interp_subprob<T, 1, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      interp_subprob<T, 1, ndim, ns><<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          binsizes, numbins, d_plan);
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(interp_subprob<T, 0, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      interp_subprob<T, 0, ndim, ns><<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          binsizes, numbins, d_plan);
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
    auto [ker, start]   = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(
        nuptsidx, xyz, nf, sigma, es_c, es_beta);

    cuda_complex<T> val = c[idxnupts[i]];
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns; ++x0, ix = (ix + 1 >= nf[0]) ? 0 : ix + 1)
        atomicAddComplexGlobal<T>(fw + ix, ker[0][x0] * val);
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= nf[1]) ? 0 : iy + 1) {
        const auto outidx0   = iy * nf[0];
        cuda_complex<T> valy = val * ker[1][y0];
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= nf[0]) ? 0 : ix + 1)
          atomicAddComplexGlobal<T>(fw + outidx0 + ix, ker[0][x0] * valy);
      }
    } else {
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= nf[2]) ? 0 : iz + 1) {
        const auto outidx0   = iz * nf[1] * nf[0];
        cuda_complex<T> valz = val * ker[2][z0];
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= nf[1]) ? 0 : iy + 1) {
          const auto outidx1   = outidx0 + iy * nf[0];
          cuda_complex<T> valy = valz * ker[1][y0];
          for (int x0 = 0, ix = start[0]; x0 < ns;
               ++x0, ix       = (ix + 1 >= nf[0]) ? 0 : ix + 1) {
            atomicAddComplexGlobal<T>(fw + outidx1 + ix, ker[0][x0] * valy);
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

  const int *d_idxnupts = dethrust(d_plan.idxnupts);

  dim3 threadsPerBlock;
  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  dim3 blocks;
  blocks.x = (d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y = 1;

  if (d_plan.opts.gpu_kerevalmeth == 1) {
    for (int t = 0; t < blksize; t++) {
      spread_nupts_driven<T, 1, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      spread_nupts_driven<T, 0, ndim, ns><<<blocks, threadsPerBlock, 0, d_plan.stream>>>(
          d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
          d_plan.nf123, es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

// FIXME unify the next two functions and templatize on a lambda?
template<typename T, int ndim>
__global__ void calc_bin_size_noghost(int M, cuda::std::array<int, 3> nf,
                                      cuda::std::array<int, 3> binsizes,
                                      cuda::std::array<int, 3> nbins, int *bin_size,
                                      cuda::std::array<const T *, 3> xyz, int *sortidx) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    int binidx = 0;
    int stride = 1;
    for (int idim = 0; idim < ndim; ++idim) {
      T rescaled = fold_rescale(xyz[idim][i], nf[idim]);
      int bin    = floor(rescaled / binsizes[idim]);
      bin        = bin >= nbins[idim] ? bin - 1 : bin;
      bin        = bin < 0 ? 0 : bin;
      binidx += bin * stride;
      stride *= nbins[idim];
    }
    int oldidx = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename T, int ndim>
__global__ void calc_inverse_of_global_sort_idx(
    int M, cuda::std::array<int, 3> binsizes, cuda::std::array<int, 3> nbins,
    const int *bin_startpts, const int *sortidx, cuda::std::array<const T *, 3> xyz,
    int *index, cuda::std::array<int, 3> nf) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    int binidx = 0;
    int stride = 1;
    for (int idim = 0; idim < ndim; ++idim) {
      T rescaled = fold_rescale(xyz[idim][i], nf[idim]);
      int bin    = floor(rescaled / binsizes[idim]);
      bin        = bin >= nbins[idim] ? bin - 1 : bin;
      bin        = bin < 0 ? 0 : bin;
      binidx += bin * stride;
      stride *= nbins[idim];
    }
    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

template<typename T, int ndim>
void cuspread_nuptsdriven_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_sort) {
    cuda::std::array<int, 3> binsizes = {
        d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};

    auto [nbins, nbins_tot] = get_nbin_info<ndim>(d_plan, binsizes);

    checkCudaErrors(cudaMemsetAsync(dethrust(d_plan.binsize), 0, nbins_tot * sizeof(int),
                                    d_plan.stream));
    calc_bin_size_noghost<T, ndim>
        <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
            d_plan.M, d_plan.nf123, binsizes, nbins, dethrust(d_plan.binsize),
            d_plan.kxyz, dethrust(d_plan.sortidx));
    THROW_IF_CUDA_ERROR

    thrust::exclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.binsize.begin(),
                           d_plan.binsize.end(), d_plan.binstartpts.begin());
    THROW_IF_CUDA_ERROR

    calc_inverse_of_global_sort_idx<T, ndim>
        <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
            d_plan.M, binsizes, nbins, dethrust(d_plan.binstartpts),
            dethrust(d_plan.sortidx), d_plan.kxyz, dethrust(d_plan.idxnupts),
            d_plan.nf123);
    THROW_IF_CUDA_ERROR
  } else {
    int *d_idxnupts = dethrust(d_plan.idxnupts);
    thrust::sequence(thrust::cuda::par.on(d_plan.stream), d_plan.idxnupts.begin(),
                     d_plan.idxnupts.begin() + d_plan.M);
    THROW_IF_CUDA_ERROR
  }
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void spread_subprob(
    cuda::std::array<const T *, 3> xyz, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, cuda::std::array<int, 3> nf, T sigma, T es_c, T es_beta,
    const int *binstartpts, const int *bin_size, cuda::std::array<int, 3> binsizes,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, cuda::std::array<int, 3> nbins, const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  const auto subpidx     = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  auto offset = compute_offset<ndim>(bidx, nbins, binsizes);

  constexpr auto ns_2       = (ns + 1) / 2;
  constexpr auto rounded_ns = ns_2 * 2;

  int N = 1;
  for (int idim = 0; idim < ndim; ++idim) N *= binsizes[idim] + rounded_ns;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx     = ptstart + i;
    auto [ker, start] = get_kerval_and_startpos_subprob<T, KEREVALMETH, ndim, ns>(
        idxnupts[idx], xyz, nf, offset, sigma, es_c, es_beta);

    const auto cnow = c[idxnupts[idx]];
    if constexpr (ndim == 1) {
      const auto ofs = start[0] + ns_2;
      for (int xx = 0; xx < ns; ++xx) {
        atomicAddComplexShared<T>(fwshared + xx + ofs, cnow * ker[0][xx]);
      }
    }
    if constexpr (ndim == 2) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto ofs0    = (start[1] + ns_2) * delta_y + start[0] + ns_2;
      for (int yy = 0; yy < ns; ++yy) {
        const auto ofs   = ofs0 + yy * delta_y;
        const auto cnowy = cnow * ker[1][yy];
        for (int xx = 0; xx < ns; ++xx) {
          atomicAddComplexShared<T>(fwshared + xx + ofs, cnowy * ker[0][xx]);
        }
      }
    }
    if constexpr (ndim == 3) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto delta_z = delta_y * (binsizes[1] + rounded_ns);
      const auto ofs0 =
          (start[2] + ns_2) * delta_z + (start[1] + ns_2) * delta_y + (start[0] + ns_2);
      for (int zz = 0; zz < ns; ++zz) {
        const auto cnowz = cnow * ker[2][zz];
        const auto ofs1  = ofs0 + zz * delta_z;
        for (int yy = 0; yy < ns; ++yy) {
          const auto cnowy = cnowz * ker[1][yy];
          const auto ofs   = ofs1 + yy * delta_y;
          for (int xx = 0; xx < ns; ++xx) {
            atomicAddComplexShared<T>(fwshared + xx + ofs, cnowy * ker[0][xx]);
          }
        }
      }
    }
  }

  __syncthreads();

  /* write to global memory */
  shared_mem_copy_helper<T, ndim, ns>(
      binsizes, offset, nf, [fw, fwshared](int idx_shared, int idx_global) {
        atomicAddComplexGlobal<T>(fw + idx_global, fwshared[idx_shared]);
      });
}
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void spread_romein(
    cuda::std::array<const T *, 3> xyz, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, cuda::std::array<int, 3> nf, T sigma, T es_c, T es_beta,
    const int *binstartpts, const int *bin_size, cuda::std::array<int, 3> binsizes,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, cuda::std::array<int, 3> nbins, const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  const auto subpidx     = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  auto offset = compute_offset<ndim>(bidx, nbins, binsizes);

  constexpr auto ns_2       = (ns + 1) / 2;
  constexpr auto rounded_ns = ns_2 * 2;

  int N = 1;
  for (int idim = 0; idim < ndim; ++idim) N *= binsizes[idim] + rounded_ns;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

  __syncthreads();

  // starting indices modulus for this thread with respect to start of local memory
  cuda::std::array<int,ndim> mod0;
  {
  int tmp = threadIdx.x;
  for (int idim=0; idim<ndim; ++idim) {
  mod0[idim] = tmp%ns;
  tmp /= ns;
  }
  }
int maxi = ((nupts+blockDim.x-1)/blockDim.x)*blockDim.x;
  for (int i = threadIdx.x; i < maxi; i += blockDim.x) {
    bool beyond_end = i>=nupts;
    const int idx     = beyond_end ? ptstart : ptstart + i;
    auto [ker, start] = get_kerval_and_startpos_subprob<T, KEREVALMETH, ndim, ns>(
        idxnupts[idx], xyz, nf, offset, sigma, es_c, es_beta);

    cuda::std::array<int,ndim> ofsxx;
    for (int idim=0; idim<ndim; ++idim) {
      int tmp = start[idim]%ns;
      if (tmp<=mod0[idim])
        tmp = mod0[idim]-tmp;
      else
        tmp = mod0[idim]+ns-tmp;
      ofsxx[idim] = tmp;
    }

    const auto cnow = c[idxnupts[idx]];
    if constexpr (ndim == 1) {
      const auto ofs = start[0] + ns_2;
      for (int ix = 0, xx=ofsxx[0]; ix < ns; ++ix, xx = (xx + 1 >= ns) ? 0 : xx + 1) {
        if (!beyond_end)
          fwshared[xx + ofs] += cnow * ker[0][xx];
        __syncthreads();
      }
    }
    if constexpr (ndim == 2) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto ofs0    = (start[1] + ns_2) * delta_y + start[0] + ns_2;
      for (int iy = 0, yy=ofsxx[1]; iy < ns; ++iy, yy = (yy + 1 >= ns) ? 0 : yy + 1) {
        const auto ofs   = ofs0 + yy * delta_y;
        const auto cnowy = cnow * ker[1][yy];
        for (int ix = 0, xx=ofsxx[0]; ix < ns; ++ix, xx = (xx + 1 >= ns) ? 0 : xx + 1) {
          if (!beyond_end)
            fwshared[xx + ofs] += cnowy * ker[0][xx];
          __syncthreads();
        }
      }
    }
    if constexpr (ndim == 3) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto delta_z = delta_y * (binsizes[1] + rounded_ns);
      const auto ofs0 =
          (start[2] + ns_2) * delta_z + (start[1] + ns_2) * delta_y + (start[0] + ns_2);
      for (int iz = 0, zz=ofsxx[2]; iz < ns; ++iz, zz = (zz + 1 >= ns) ? 0 : zz + 1) {
        const auto cnowz = cnow * ker[2][zz];
        const auto ofs1  = ofs0 + zz * delta_z;
        for (int iy = 0, yy=ofsxx[1]; iy < ns; ++iy, yy = (yy + 1 >= ns) ? 0 : yy + 1) {
          const auto cnowy = cnowz * ker[1][yy];
          const auto ofs   = ofs1 + yy * delta_y;
          for (int ix = 0, xx=ofsxx[0]; ix < ns; ++ix, xx = (xx + 1 >= ns) ? 0 : xx + 1) {
            if (!beyond_end)
              fwshared[xx + ofs] += cnowy * ker[0][xx];
            __syncthreads();
          }
        }
      }
    }
  }

  __syncthreads();

  /* write to global memory */
  shared_mem_copy_helper<T, ndim, ns>(
      binsizes, offset, nf, [fw, fwshared](int idx_shared, int idx_global) {
        atomicAddComplexGlobal<T>(fw + idx_global, fwshared[idx_shared]);
      });
}

template<typename T, int ndim, int ns>
static void cuspread_subprob(const cufinufft_plan_t<T> &d_plan, int blksize) {

  // assume that bin_size > ns/2;
  cuda::std::array<int, 3> binsizes = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
                                       d_plan.opts.gpu_binsizez};
  auto [nbins, nbins_tot]           = get_nbin_info<ndim>(d_plan, binsizes);

  T sigma                      = d_plan.spopts.upsampfac;
  T es_c                       = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta                    = d_plan.spopts.beta;
  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);
  if (d_plan.opts.gpu_kerevalmeth) {
    cufinufft_set_shared_memory(spread_subprob<T, 1, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_subprob<T, 1, ndim, ns>
          <<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
              d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
              d_plan.nf123, sigma, es_c, es_beta, dethrust(d_plan.binstartpts),
              dethrust(d_plan.binsize), binsizes, dethrust(d_plan.subprob_to_bin),
              dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob),
              d_plan.opts.gpu_maxsubprobsize, nbins, dethrust(d_plan.idxnupts));
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(spread_subprob<T, 0, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_subprob<T, 1, ndim, ns>
          <<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
              d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
              d_plan.nf123, sigma, es_c, es_beta, dethrust(d_plan.binstartpts),
              dethrust(d_plan.binsize), binsizes, dethrust(d_plan.subprob_to_bin),
              dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob),
              d_plan.opts.gpu_maxsubprobsize, nbins, dethrust(d_plan.idxnupts));
      THROW_IF_CUDA_ERROR
    }
  }
}

template<typename T, int ndim, int ns>
static void cuspread_romein(const cufinufft_plan_t<T> &d_plan, int blksize) {

  // assume that bin_size > ns/2;
  cuda::std::array<int, 3> binsizes = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
                                       d_plan.opts.gpu_binsizez};
  auto [nbins, nbins_tot]           = get_nbin_info<ndim>(d_plan, binsizes);

  T sigma                      = d_plan.spopts.upsampfac;
  T es_c                       = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta                    = d_plan.spopts.beta;
  int bufsz=1;
  for (int idim=0; idim<ndim; ++idim)
    bufsz*=ns;
  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);
  if (d_plan.opts.gpu_kerevalmeth) {
    cufinufft_set_shared_memory(spread_romein<T, 1, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_romein<T, 1, ndim, ns>
          <<<d_plan.totalnumsubprob, min(256,bufsz), sharedplanorysize, d_plan.stream>>>(
              d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
              d_plan.nf123, sigma, es_c, es_beta, dethrust(d_plan.binstartpts),
              dethrust(d_plan.binsize), binsizes, dethrust(d_plan.subprob_to_bin),
              dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob),
              d_plan.opts.gpu_maxsubprobsize, nbins, dethrust(d_plan.idxnupts));
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(spread_romein<T, 0, ndim, ns>, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      spread_romein<T, 1, ndim, ns>
          <<<d_plan.totalnumsubprob, min(256,bufsz), sharedplanorysize, d_plan.stream>>>(
              d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
              d_plan.nf123, sigma, es_c, es_beta, dethrust(d_plan.binstartpts),
              dethrust(d_plan.binsize), binsizes, dethrust(d_plan.subprob_to_bin),
              dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob),
              d_plan.opts.gpu_maxsubprobsize, nbins, dethrust(d_plan.idxnupts));
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

template<typename T, int ndim>
static void cuspread_subprob_prop(cufinufft_plan_t<T> &d_plan) {
  cuda::std::array<int, 3> binsizes = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
                                       d_plan.opts.gpu_binsizez};

  auto [nbins, nbins_tot] = get_nbin_info<ndim>(d_plan, binsizes);

  checkCudaErrors(cudaMemsetAsync(dethrust(d_plan.binsize), 0, nbins_tot * sizeof(int),
                                  d_plan.stream));
  calc_bin_size_noghost<T, ndim>
      <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
          d_plan.M, d_plan.nf123, binsizes, nbins, dethrust(d_plan.binsize), d_plan.kxyz,
          dethrust(d_plan.sortidx));
  THROW_IF_CUDA_ERROR
  thrust::exclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.binsize.begin(),
                         d_plan.binsize.end(), d_plan.binstartpts.begin());
  THROW_IF_CUDA_ERROR
  calc_inverse_of_global_sort_idx<T, ndim>
      <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
          d_plan.M, binsizes, nbins, dethrust(d_plan.binstartpts),
          dethrust(d_plan.sortidx), d_plan.kxyz, dethrust(d_plan.idxnupts), d_plan.nf123);
  THROW_IF_CUDA_ERROR

  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  calc_subprob<<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
      dethrust(d_plan.binsize), dethrust(d_plan.numsubprob),
      d_plan.opts.gpu_maxsubprobsize, nbins_tot);
  THROW_IF_CUDA_ERROR

  thrust::inclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.numsubprob.begin(),
                         d_plan.numsubprob.begin() + nbins_tot,
                         d_plan.subprobstartpts.begin() + 1);
  THROW_IF_CUDA_ERROR

  int totalnumsubprob;
  checkCudaErrors(
      cudaMemsetAsync(dethrust(d_plan.subprobstartpts), 0, sizeof(int), d_plan.stream));
  checkCudaErrors(
      cudaMemcpyAsync(&totalnumsubprob, &(dethrust(d_plan.subprobstartpts)[nbins_tot]),
                      sizeof(int), cudaMemcpyDeviceToHost, d_plan.stream));
  cudaStreamSynchronize(d_plan.stream);
  d_plan.subprob_to_bin.resize(totalnumsubprob);

  map_b_into_subprob<<<(nbins[0] * nbins[1] + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
      dethrust(d_plan.subprob_to_bin), dethrust(d_plan.subprobstartpts),
      dethrust(d_plan.numsubprob), nbins_tot);

  d_plan.totalnumsubprob = totalnumsubprob;
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ void spread_output_driven(
    cuda::std::array<const T *, 3> xyz, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, cuda::std::array<int, 3> nf, T sigma, T es_c, T es_beta,
    const int *binstartpts, const int *bin_size, cuda::std::array<int, 3> binsizes,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, cuda::std::array<int, 3> nbins, const int *idxnupts, int np) {
  extern __shared__ char sharedbuf[];

  static constexpr auto ns_2f      = T(ns * .5);
  static constexpr auto ns_2       = (ns + 1) / 2;
  static constexpr auto rounded_ns = ns_2 * 2;
  int total                        = 1;
  for (int idim = 0; idim < ndim; ++idim) total *= ns;

  cuda::std::array<int, ndim> padded_size;
  for (int idim = 0; idim < ndim; ++idim) padded_size[idim] = binsizes[idim] + rounded_ns;

  const int bidx        = subprob_to_bin[blockIdx.x];
  const int binsubp_idx = blockIdx.x - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  auto offset = compute_offset<ndim>(bidx, nbins, binsizes);

  using kernel_data = cuda::std::array<cuda::std::array<T, ns>, ndim>;
  auto *kerevals    = reinterpret_cast<kernel_data *>(sharedbuf);
  // Offset pointer into sharedbuf after kerevals
  auto *nupts_sm =
      reinterpret_cast<cuda_complex<T> *>(sharedbuf + np * sizeof(kernel_data));
  auto *shift = reinterpret_cast<cuda::std::array<int, ndim> *>(
      sharedbuf + np * sizeof(kernel_data) + np * sizeof(cuda_complex<T>));

  auto *local_subgrid = reinterpret_cast<cuda_complex<T> *>(
      sharedbuf + np * sizeof(kernel_data) + np * sizeof(cuda_complex<T>) +
      np * sizeof(cuda::std::array<int, ndim>));

  int local_subgrid_size = 1;
  for (int idim = 0; idim < ndim; ++idim) local_subgrid_size *= padded_size[idim];

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid_size; i += blockDim.x) {
    local_subgrid[i] = {0, 0};
  }

  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx = loadReadOnly(idxnupts + ptstart + i + batch_begin);
      nupts_sm[i]        = c[nuptsidx];
      for (size_t idim = 0; idim < ndim; ++idim) {
        auto rescaled    = fold_rescale(xyz[idim][nuptsidx], nf[idim]);
        const auto start = int(std::ceil(rescaled - ns_2f));
        // FIXME: used get_kerval_and_startpos?
        if constexpr (KEREVALMETH == 1) {
          eval_kernel_vec_horner<T, ns>(&kerevals[i][idim][0], T(start) - rescaled,
                                        sigma);
        } else {
          eval_kernel_vec<T, ns>(&kerevals[i][idim][0], T(start) - rescaled, es_c,
                                 es_beta);
        }
        shift[i][idim] = start - offset[idim]; // + ((s < 0) ? nf[idim] : 0);
      }
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      const auto cnow  = nupts_sm[i];
      const auto start = shift[i];

      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // strength from shared memory
        int tmp       = idx;
        int idxout    = 0;
        T kervalue    = 1;
        int strideout = 1;
        for (int idim = 0; idim + 1 < ndim; ++idim) {
          int s = tmp % ns;
          kervalue *= kerevals[i][idim][s];
          idxout += strideout * (s + start[idim] + ns_2);
          strideout *= padded_size[idim];
          tmp /= ns;
        }
        // last dimension can be done more cheaply
        kervalue *= kerevals[i][ndim - 1][tmp];
        idxout += strideout * (tmp + start[ndim - 1] + ns_2);

        local_subgrid[idxout] += cnow * kervalue;
      }
      __syncthreads();
    }
  }
  /* write to global memory */
  for (int n = threadIdx.x; n < local_subgrid_size; n += blockDim.x) {
    int flatidx   = n;
    int outidx    = 0;
    int outstride = 1;
    for (int idim = 0; idim < ndim; ++idim) {
      int idx0 = flatidx % padded_size[idim];
      int idx  = idx0 + offset[idim] - ns_2;
      idx      = idx < 0 ? idx + nf[idim] : (idx >= nf[idim] ? idx - nf[idim] : idx);
      outidx += idx * outstride;
      outstride *= nf[idim];
      flatidx /= padded_size[idim];
    }
    atomicAddComplexGlobal<T>(fw + outidx, local_subgrid[n]);
  }
}

template<typename T, int ndim, int ns>
static void cuspread_output_driven(const cufinufft_plan_t<T> &d_plan, int blksize) {
  // assume that bin_size_x > ns/2;
  cuda::std::array<int, 3> binsizes{d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
                                    d_plan.opts.gpu_binsizez};
  cuda::std::array<int, 3> nbins{1, 1, 1};
  for (int idim = 0; idim < ndim; ++idim)
    nbins[idim] = ceil(T(d_plan.nf123[idim]) / binsizes[idim]);

  T sigma   = d_plan.spopts.upsampfac;
  T es_c    = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta = d_plan.spopts.beta;

  int bufsz = 1;
  for (int idim = 0; idim < ndim; ++idim) bufsz *= ns;

  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, ns, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);
  if (d_plan.opts.gpu_kerevalmeth) {
    cufinufft_set_shared_memory(spread_output_driven<T, 1, ndim, ns>, ndim, d_plan);
    cudaFuncSetSharedMemConfig(spread_output_driven<T, 1, ndim, ns>,
                               cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      spread_output_driven<T, 1, ndim, ns>
          <<<d_plan.totalnumsubprob, std::min(256, std::max(bufsz, d_plan.opts.gpu_np)),
             sharedplanorysize, d_plan.stream>>>(
              d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
              d_plan.nf123, sigma, es_c, es_beta, dethrust(d_plan.binstartpts),
              dethrust(d_plan.binsize), binsizes, dethrust(d_plan.subprob_to_bin),
              dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob),
              d_plan.opts.gpu_maxsubprobsize, nbins, dethrust(d_plan.idxnupts),
              d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(spread_output_driven<T, 0, ndim, ns>, ndim, d_plan);
    cudaFuncSetSharedMemConfig(spread_output_driven<T, 0, ndim, ns>,
                               cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      spread_output_driven<T, 0, ndim, ns>
          <<<d_plan.totalnumsubprob, std::min(256, std::max(bufsz, d_plan.opts.gpu_np)),
             sharedplanorysize, d_plan.stream>>>(
              d_plan.kxyz, d_plan.c + t * d_plan.M, d_plan.fw + t * d_plan.nf, d_plan.M,
              d_plan.nf123, sigma, es_c, es_beta, dethrust(d_plan.binstartpts),
              dethrust(d_plan.binsize), binsizes, dethrust(d_plan.subprob_to_bin),
              dethrust(d_plan.subprobstartpts), dethrust(d_plan.numsubprob),
              d_plan.opts.gpu_maxsubprobsize, nbins, dethrust(d_plan.idxnupts),
              d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  }
}

} // namespace spreadinterp
} // namespace cufinufft

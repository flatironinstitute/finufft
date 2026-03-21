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

/* --------------------------- Shared Helpers ---------------------------- */

template<int ndim>
__host__ __device__ auto get_nbins(cuda::std::array<int, 3> nf123,
                                   cuda::std::array<int, 3> binsizes) {
  cuda::std::array<int, 3> nbins{1, 1, 1};
  for (int idim = 0; idim < ndim; ++idim)
    nbins[idim] = (nf123[idim] + binsizes[idim] - 1) / binsizes[idim];
  return nbins;
}

inline __host__ __device__ int nbins_total(const cuda::std::array<int, 3> &nbins) {
  return nbins[0] * nbins[1] * nbins[2];
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__device__ auto get_kerval_and_local_start(
    int idx, cuda::std::array<const T *, 3> xyz, cuda::std::array<int, 3> nf,
    cuda::std::array<int, ndim> offset, T sigma, T es_c, T es_beta) {
  constexpr auto ns_2f = T(ns * .5);
  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
  cuda::std::array<int, ndim> start;
  for (int idim = 0; idim < ndim; ++idim) {
    const auto rescaled = fold_rescale(loadReadOnly(xyz[idim] + idx), nf[idim]);
    const auto s        = int(std::ceil(rescaled - ns_2f));
    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(&ker[idim][0], T(s) - rescaled, sigma);
    } else {
      eval_kernel_vec<T, ns>(&ker[idim][0], T(s) - rescaled, es_c, es_beta);
    }
    start[idim] = s - offset[idim];
  }
  return cuda::std::make_tuple(ker, start);
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__device__ auto get_kerval_and_startpos_nuptsdriven(
    int idx, cuda::std::array<const T *, 3> xyz, cuda::std::array<int, 3> nf, T sigma,
    T es_c, T es_beta) {
  constexpr auto ns_2f = T(ns * .5);
  cuda::std::array<cuda::std::array<T, ns>, ndim> ker;
  cuda::std::array<int, ndim> start;
  for (size_t idim = 0; idim < ndim; ++idim) {
    const auto rescaled = fold_rescale(loadReadOnly(xyz[idim] + idx), nf[idim]);
    const auto s        = int(std::ceil(rescaled - ns_2f));
    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(&ker[idim][0], T(s) - rescaled, sigma);
    } else {
      eval_kernel_vec<T, ns>(&ker[idim][0], T(s) - rescaled, es_c, es_beta);
    }
    start[idim] = s + ((s < 0) ? nf[idim] : 0);
  }
  return cuda::std::make_tuple(ker, start);
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

template<int ndim, typename T>
__device__ int compute_bin_index(
    int idx, cuda::std::array<int, 3> nf, cuda::std::array<T, 3> inv_binsizes,
    cuda::std::array<int, 3> nbins, cuda::std::array<const T *, 3> xyz) {
  int binidx = 0;
  int stride = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    const T rescaled = fold_rescale(loadReadOnly(xyz[idim] + idx), nf[idim]);
    int bin          = floor(rescaled * inv_binsizes[idim]);
    bin              = bin >= nbins[idim] ? bin - 1 : bin;
    bin              = bin < 0 ? 0 : bin;
    binidx += bin * stride;
    stride *= nbins[idim];
  }
  return binidx;
}

template<int ndim, int ns>
__device__ auto get_padded_subgrid_info(const cuda::std::array<int, 3> &binsizes) {
  constexpr auto rounded_ns = ((ns + 1) / 2) * 2;
  cuda::std::array<int, ndim> padded_size;
  int total = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    padded_size[idim] = binsizes[idim] + rounded_ns;
    total *= padded_size[idim];
  }
  return cuda::std::make_tuple(padded_size, total);
}

template<int ndim, int ns>
__device__ int output_index_from_flat_local_index(
    int flatidx, const cuda::std::array<int, ndim> &padded_size,
    const cuda::std::array<int, ndim> &offset, const cuda::std::array<int, 3> &nf) {
  constexpr auto ns_2 = (ns + 1) / 2;

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
  return outidx;
}

/* ------------------------- Interp Kernels ------------------------------ */

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void interp_nupts_driven(
    cufinufft_gpu_data<T> p, cuda_complex<T> *c, const cuda_complex<T> *fw) {
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
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        cnow += loadReadOnly(fw + ix) * ker[0][x0];
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
        const auto inidx0 = iy * p.nf123[0];
        cuda_complex<T> cnowx{0, 0};
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
          cnowx += loadReadOnly(fw + inidx0 + ix) * ker[0][x0];
        cnow += cnowx * ker[1][y0];
      }
    }
    if constexpr (ndim == 3) {
      cuda::std::array<int, ns> xidx;
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        xidx[x0] = ix;
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= p.nf123[2]) ? 0 : iz + 1) {
        const auto inidx0 = iz * p.nf123[1] * p.nf123[0];
        cuda_complex<T> cnowy{0, 0};
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
          const auto inidx1 = inidx0 + iy * p.nf123[0];
          cuda_complex<T> cnowx{0, 0};
          for (int x0 = 0; x0 < ns; ++x0)
            cnowx += loadReadOnly(fw + inidx1 + xidx[x0]) * ker[0][x0];
          cnowy += cnowx * ker[1][y0];
        }
        cnow += cnowy * ker[2][z0];
      }
    }
    storeCacheStreaming(c + nuptsidx, cnow);
  }
}

template<typename T, int ndim, int ns>
void cuinterp_nuptsdriven(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                          const cuda_complex<T> *fw, int blksize) {
  const dim3 threadsPerBlock{
      std::min(optimal_block_threads(d_plan.opts.gpu_device_id), (unsigned)d_plan.M), 1u,
      1u};
  const dim3 blocks{(d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1};

  const auto launch = [&](auto kernel) {
    for (int t = 0; t < blksize; t++) {
      kernel<<<blocks, threadsPerBlock, 0, d_plan.stream>>>(d_plan, c + t * d_plan.M,
                                                            fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(interp_nupts_driven<T, 1, ndim, ns>)
                                     : launch(interp_nupts_driven<T, 0, ndim, ns>);
}

template<typename T, int ndim, int ns, typename Func>
__device__ void shared_mem_copy_helper(cuda::std::array<int, 3> binsizes,
                                       cuda::std::array<int, ndim> offset,
                                       cuda::std::array<int, 3> nf, Func func) {
  constexpr auto ns_2 = (ns + 1) / 2;

  auto [padded_size, N] = get_padded_subgrid_info<ndim, ns>(binsizes);

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    bool in_region = true;
    int flatidx    = n;
    int globidx    = 0;
    int globstride = 1;
    for (int idim = 0; idim < ndim; ++idim) {
      int idx0 = flatidx % padded_size[idim];
      int idx  = idx0 + offset[idim] - ns_2;
      if (idx >= nf[idim] + ns_2) in_region = false;
      idx = idx < 0 ? idx + nf[idim] : (idx >= nf[idim] ? idx - nf[idim] : idx);
      globidx += idx * globstride;
      globstride *= nf[idim];
      flatidx /= padded_size[idim];
    }
    if (in_region)
      func(n, globidx); // atomicAddComplexGlobal<T>(fw + outidx, fwshared[n]);
  }
}

/* Kernels for SubProb Method */
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void interp_subprob(
    cufinufft_gpu_data<T> p, cuda_complex<T> *c, const cuda_complex<T> *fw) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  T sigma   = p.spopts.upsampfac;
  T es_c    = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta = p.spopts.beta;

  // assume that bin_size > ns/2;
  cuda::std::array<int, 3> binsizes{p.opts.gpu_binsizex, p.opts.gpu_binsizey,
                                    p.opts.gpu_binsizez};
  auto nbins = get_nbins<ndim>(p.nf123, binsizes);

  const auto subpidx     = blockIdx.x;
  const auto bidx        = loadReadOnly(p.subprob_to_bin + subpidx);
  const auto binsubp_idx = subpidx - loadReadOnly(p.subprobstartpts + bidx);
  const auto ptstart =
      loadReadOnly(p.binstartpts + bidx) + binsubp_idx * p.opts.gpu_maxsubprobsize;
  const auto nupts =
      min(p.opts.gpu_maxsubprobsize,
          loadReadOnly(p.binsize + bidx) - binsubp_idx * p.opts.gpu_maxsubprobsize);

  auto offset = compute_offset<ndim>(bidx, nbins, binsizes);

  constexpr auto ns_2       = (ns + 1) / 2;
  constexpr auto rounded_ns = ns_2 * 2;

  shared_mem_copy_helper<T, ndim, ns>(
      binsizes, offset, p.nf123, [fw, fwshared](int idx_shared, int idx_global) {
        fwshared[idx_shared] = loadReadOnly(fw + idx_global);
      });
  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx       = ptstart + i;
    const auto nuptsidx = loadReadOnly(p.idxnupts + idx);
    auto [ker, start]   = get_kerval_and_local_start<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, offset, sigma, es_c, es_beta);

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
    storeCacheStreaming(c + nuptsidx, cnow);
  }
}

template<typename T, int ndim, int ns>
void cuinterp_subprob(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                      const cuda_complex<T> *fw, int blksize) {
  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);

  const auto launch = [&](auto kernel) {
    cufinufft_set_shared_memory(kernel, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      kernel<<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          d_plan, c + t * d_plan.M, fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(interp_subprob<T, 1, ndim, ns>)
                                     : launch(interp_subprob<T, 0, ndim, ns>);
}

/* ------------------------- Spread Kernels ------------------------------ */

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void spread_nupts_driven(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw) {

  T sigma   = p.spopts.upsampfac;
  T es_c    = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta = p.spopts.beta;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < p.M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(p.idxnupts + i);
    auto [ker, start]   = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, sigma, es_c, es_beta);

    const auto val = loadReadOnly(c + nuptsidx);
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        atomicAddComplexGlobal<T>(fw + ix, ker[0][x0] * val);
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
        const auto outidx0   = iy * p.nf123[0];
        cuda_complex<T> valy = val * ker[1][y0];
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
          atomicAddComplexGlobal<T>(fw + outidx0 + ix, ker[0][x0] * valy);
      }
    } else {
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= p.nf123[2]) ? 0 : iz + 1) {
        const auto outidx0   = iz * p.nf123[1] * p.nf123[0];
        cuda_complex<T> valz = val * ker[2][z0];
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
          const auto outidx1   = outidx0 + iy * p.nf123[0];
          cuda_complex<T> valy = valz * ker[1][y0];
          for (int x0 = 0, ix = start[0]; x0 < ns;
               ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1) {
            atomicAddComplexGlobal<T>(fw + outidx1 + ix, ker[0][x0] * valy);
          }
        }
      }
    }
  }
}

template<typename T, int ndim, int ns>
void cuspread_nupts_driven(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                           cuda_complex<T> *fw, int blksize) {
  auto &stream = d_plan.stream;

  dim3 threadsPerBlock;
  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  dim3 blocks;
  blocks.x = (d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y = 1;

  const auto launch = [&](auto kernel) {
    for (int t = 0; t < blksize; t++) {
      kernel<<<blocks, threadsPerBlock, 0, d_plan.stream>>>(d_plan, c + t * d_plan.M,
                                                            fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(spread_nupts_driven<T, 1, ndim, ns>)
                                     : launch(spread_nupts_driven<T, 0, ndim, ns>);
}

// FIXME unify the next two functions and templatize on a lambda?
template<typename T, int ndim>
__global__ FINUFFT_FLATTEN void calc_bin_size_noghost(
    const int M, const cuda::std::array<int, 3> nf,
    const cuda::std::array<T, 3> inv_binsizes, const cuda::std::array<int, 3> nbins,
    int *FINUFFT_RESTRICT bin_size, const cuda::std::array<const T *, 3> xyz,
    int *FINUFFT_RESTRICT sortidx) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    const int binidx = compute_bin_index<ndim>(i, nf, inv_binsizes, nbins, xyz);
    const int oldidx = atomicAdd(&bin_size[binidx], 1);
    storeCacheStreaming(sortidx + i, oldidx);
  }
}

template<typename T, int ndim>
__global__ FINUFFT_FLATTEN void calc_inverse_of_global_sort_idx(
    const int M, const cuda::std::array<T, 3> inv_binsizes,
    const cuda::std::array<int, 3> nbins, const int *FINUFFT_RESTRICT bin_startpts,
    const int *FINUFFT_RESTRICT sortidx, const cuda::std::array<const T *, 3> xyz,
    int *FINUFFT_RESTRICT index, const cuda::std::array<int, 3> nf) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    const int binidx = compute_bin_index<ndim>(i, nf, inv_binsizes, nbins, xyz);
    storeCacheStreaming(
        index + loadReadOnly(bin_startpts + binidx) + loadReadOnly(sortidx + i), i);
  }
}

template<typename T, int ndim>
void cuspread_nuptsdriven_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_sort) {
    cuda::std::array<int, 3> binsizes = {
        d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};

    auto nbins          = get_nbins<ndim>(d_plan.nf123, binsizes);
    const int nbins_tot = nbins_total(nbins);
    const cuda::std::array<T, 3> inv_binsizes{T(1) / binsizes[0], T(1) / binsizes[1],
                                              T(1) / binsizes[2]};

    checkCudaErrors(cudaMemsetAsync(dethrust(d_plan.binsize), 0, nbins_tot * sizeof(int),
                                    d_plan.stream));
    calc_bin_size_noghost<T, ndim>
        <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
            d_plan.M, d_plan.nf123, inv_binsizes, nbins, dethrust(d_plan.binsize),
            d_plan.kxyz, dethrust(d_plan.sortidx));
    THROW_IF_CUDA_ERROR

    thrust::exclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.binsize.begin(),
                           d_plan.binsize.end(), d_plan.binstartpts.begin());
    THROW_IF_CUDA_ERROR

    calc_inverse_of_global_sort_idx<T, ndim>
        <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
            d_plan.M, inv_binsizes, nbins, dethrust(d_plan.binstartpts),
            dethrust(d_plan.sortidx), d_plan.kxyz, dethrust(d_plan.idxnupts),
            d_plan.nf123);
    THROW_IF_CUDA_ERROR
  } else {
    thrust::sequence(thrust::cuda::par.on(d_plan.stream), d_plan.idxnupts.begin(),
                     d_plan.idxnupts.begin() + d_plan.M);
    THROW_IF_CUDA_ERROR
  }
}

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void spread_subprob(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  T sigma   = p.spopts.upsampfac;
  T es_c    = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta = p.spopts.beta;

  // assume that bin_size > ns/2;
  cuda::std::array<int, 3> binsizes{p.opts.gpu_binsizex, p.opts.gpu_binsizey,
                                    p.opts.gpu_binsizez};
  auto nbins = get_nbins<ndim>(p.nf123, binsizes);

  const auto subpidx     = blockIdx.x;
  const auto bidx        = loadReadOnly(p.subprob_to_bin + subpidx);
  const auto binsubp_idx = subpidx - loadReadOnly(p.subprobstartpts + bidx);
  const auto ptstart =
      loadReadOnly(p.binstartpts + bidx) + binsubp_idx * p.opts.gpu_maxsubprobsize;
  const auto nupts =
      min(p.opts.gpu_maxsubprobsize,
          loadReadOnly(p.binsize + bidx) - binsubp_idx * p.opts.gpu_maxsubprobsize);

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
    const int idx       = ptstart + i;
    const auto nuptsidx = loadReadOnly(p.idxnupts + idx);
    auto [ker, start]   = get_kerval_and_local_start<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, offset, sigma, es_c, es_beta);

    const auto cnow = loadReadOnly(c + nuptsidx);
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
      binsizes, offset, p.nf123, [fw, fwshared](int idx_shared, int idx_global) {
        atomicAddComplexGlobal<T>(fw + idx_global, fwshared[idx_shared]);
      });
}

template<typename T, int ndim, int ns>
static void cuspread_subprob(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                             cuda_complex<T> *fw, int blksize) {
  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);

  const auto launch = [&](auto kernel) {
    cufinufft_set_shared_memory(kernel, ndim, d_plan);
    for (int t = 0; t < blksize; t++) {
      kernel<<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          d_plan, c + t * d_plan.M, fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(spread_subprob<T, 1, ndim, ns>)
                                     : launch(spread_subprob<T, 0, ndim, ns>);
}

static __global__ void calc_subprob(const int *FINUFFT_RESTRICT bin_size,
                                    int *FINUFFT_RESTRICT num_subprob,
                                    const int maxsubprobsize, const int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    num_subprob[i] = (loadReadOnly(bin_size + i) + maxsubprobsize - 1) / maxsubprobsize;
  }
}
static __global__ void map_b_into_subprob(
    int *FINUFFT_RESTRICT d_subprob_to_bin, const int *FINUFFT_RESTRICT d_subprobstartpts,
    const int *FINUFFT_RESTRICT d_numsubprob, const int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < loadReadOnly(d_numsubprob + i); j++) {
      d_subprob_to_bin[loadReadOnly(d_subprobstartpts + i) + j] = i;
    }
  }
}

template<typename T, int ndim>
static void cuspread_subprob_prop(cufinufft_plan_t<T> &d_plan) {
  cuda::std::array<int, 3> binsizes = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
                                       d_plan.opts.gpu_binsizez};

  auto nbins          = get_nbins<ndim>(d_plan.nf123, binsizes);
  const int nbins_tot = nbins_total(nbins);
  const cuda::std::array<T, 3> inv_binsizes{T(1) / binsizes[0], T(1) / binsizes[1],
                                            T(1) / binsizes[2]};

  checkCudaErrors(cudaMemsetAsync(dethrust(d_plan.binsize), 0, nbins_tot * sizeof(int),
                                  d_plan.stream));
  calc_bin_size_noghost<T, ndim>
      <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
          d_plan.M, d_plan.nf123, inv_binsizes, nbins, dethrust(d_plan.binsize),
          d_plan.kxyz, dethrust(d_plan.sortidx));
  THROW_IF_CUDA_ERROR
  thrust::exclusive_scan(thrust::cuda::par.on(d_plan.stream), d_plan.binsize.begin(),
                         d_plan.binsize.end(), d_plan.binstartpts.begin());
  THROW_IF_CUDA_ERROR
  calc_inverse_of_global_sort_idx<T, ndim>
      <<<(d_plan.M + 1024 - 1) / 1024, 1024, 0, d_plan.stream>>>(
          d_plan.M, inv_binsizes, nbins, dethrust(d_plan.binstartpts),
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

/* ---------------------- Output-Driven Kernels -------------------------- */

template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void spread_output_driven(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw, int np) {
  extern __shared__ char sharedbuf[];

  T sigma   = p.spopts.upsampfac;
  T es_c    = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta = p.spopts.beta;

  // assume that bin_size > ns/2;
  cuda::std::array<int, 3> binsizes{p.opts.gpu_binsizex, p.opts.gpu_binsizey,
                                    p.opts.gpu_binsizez};
  auto nbins = get_nbins<ndim>(p.nf123, binsizes);

  static constexpr auto ns_2f = T(ns * .5);
  static constexpr auto ns_2  = (ns + 1) / 2;
  int total                   = 1;

  for (int idim = 0; idim < ndim; ++idim) total *= ns;

  auto [padded_size, local_subgrid_size] = get_padded_subgrid_info<ndim, ns>(binsizes);

  const int bidx        = loadReadOnly(p.subprob_to_bin + blockIdx.x);
  const int binsubp_idx = blockIdx.x - loadReadOnly(p.subprobstartpts + bidx);
  const int ptstart =
      loadReadOnly(p.binstartpts + bidx) + binsubp_idx * p.opts.gpu_maxsubprobsize;
  const int nupts =
      min(p.opts.gpu_maxsubprobsize,
          loadReadOnly(p.binsize + bidx) - binsubp_idx * p.opts.gpu_maxsubprobsize);

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

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid_size; i += blockDim.x) {
    local_subgrid[i] = {0, 0};
  }

  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx      = loadReadOnly(p.idxnupts + ptstart + i + batch_begin);
      nupts_sm[i]             = loadReadOnly(c + nuptsidx);
      auto [ker, local_shift] = get_kerval_and_local_start<T, KEREVALMETH, ndim, ns>(
          nuptsidx, p.xyz, p.nf123, offset, sigma, es_c, es_beta);
      kerevals[i] = ker;
      shift[i]    = local_shift;
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
    const int outidx =
        output_index_from_flat_local_index<ndim, ns>(n, padded_size, offset, p.nf123);
    atomicAddComplexGlobal<T>(fw + outidx, local_subgrid[n]);
  }
}

template<typename T, int ndim, int ns>
static void cuspread_output_driven(const cufinufft_plan_t<T> &d_plan,
                                   const cuda_complex<T> *c, cuda_complex<T> *fw,
                                   int blksize) {
  int bufsz = 1;
  for (int idim = 0; idim < ndim; ++idim) bufsz *= ns;

  const auto sharedplanorysize = shared_memory_required<T>(
      ndim, ns, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);
  const int nthreads = std::min(256, std::max(bufsz, d_plan.opts.gpu_np));

  const auto launch = [&](auto kernel) {
    cufinufft_set_shared_memory(kernel, ndim, d_plan);
    cudaFuncSetSharedMemConfig(kernel, cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      kernel<<<d_plan.totalnumsubprob, std::min(256, std::max(bufsz, d_plan.opts.gpu_np)),
               sharedplanorysize, d_plan.stream>>>(
          d_plan, c + t * d_plan.M, fw + t * d_plan.nf, d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(spread_output_driven<T, 1, ndim, ns>)
                                     : launch(spread_output_driven<T, 0, ndim, ns>);
}

} // namespace spreadinterp
} // namespace cufinufft

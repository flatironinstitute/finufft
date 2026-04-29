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

template<typename T>
static constexpr __forceinline__ __device__ T cudaFMA(const T a, const T b, const T c) {
  if constexpr (std::is_same_v<T, float>) {
    // fused multiply-add, round to nearest even
    return __fmaf_rn(a, b, c);
  } else if constexpr (std::is_same_v<T, double>) {
    // fused multiply-add, round to nearest even
    return __fma_rn(a, b, c);
  }
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Only float and double are supported.");
  return std::fma(a, b, c);
}

/**
 * local NU coord fold+rescale macro: does the following affine transform to x:
 *   (x+PI) mod PI    each to [0,N)
 */
template<typename T>
static constexpr __forceinline__ __host__ __device__ T fold_rescale(T x, int N) {
  constexpr auto x2pi = T(0.159154943091895345554011992339482617);
  constexpr auto half = T(0.5);
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) {
    // fused multiply-add, round to nearest even
    auto result = cudaFMA(x, x2pi, half);
    // subtract, round down
    result = __fsub_rd(result, floorf(result));
    // multiply, round down
    return __fmul_rd(result, static_cast<T>(N));
  } else if constexpr (std::is_same_v<T, double>) {
    // fused multiply-add, round to nearest even
    auto result = cudaFMA(x, x2pi, half);
    // subtract, round down
    result = __dsub_rd(result, floor(result));
    // multiply, round down
    return __dmul_rd(result, static_cast<T>(N));
  } else {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Only float and double are supported.");
  }
#else
  const auto result = std::fma(x, x2pi, half);
  return (result - std::floor(result)) * static_cast<T>(N);
#endif
}

template<typename T, int ns>
static __forceinline__ __device__ T evaluate_kernel(T x, T es_c, T es_beta)
// ns (spread-width) templated fast evaluator for the above kernel function.
// Direct eval, hardwired for this kernel, with shape param es_beta.
// es_c must have been set up as (2/ns)^2; the point of precomputing this is to
// reduce flops (as in the original CPU kernel). es_halfwidth has been cut.
// Used only by the below eval_kernel_vec(), hence when gpu_kerevalmeth=0.
// To do: *** unify kernel logic/coeffs with CPU codes in common/
{
  const T zsq = es_c * x * x; // z^2, where z is arg for std interval [-1,1]
  return (zsq < 1.0) ? exp((T)es_beta * (sqrt((T)1.0 - zsq) - (T)1.0)) : 0.0;
}

template<typename T, int w>
static __inline__ __device__ void eval_kernel_vec(T *ker, const T x, const T es_c,
                                                  const T es_beta) {
  // Eval the above direct ES kernel evaluator for arguments x+j, for j=0,..,w-1.
  // This is used when gpu_kerevalmeth=0.
  // Serves the same purpose as the below function eval_kernel_vec_horner.
  for (int i = 0; i < w; i++) ker[i] = evaluate_kernel<T, w>(abs(x + i), es_c, es_beta);
}

template<typename T, int w>
static __device__ void eval_kernel_vec_horner(T *ker, const T x, const double upsampfac)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   Two upsampfacs implemented (same as CPU coeffs created in 2018, used to 2025).
   The parameter (w and beta) choice in setup_spreader must match these coeffs.
   This is used when gpu_kerevalmeth=1.
   To do: *** update horner evaluator and coeffs to match CPU in common/
   */
{
  // const T z = T(2) * x + T(w - 1);
  const auto z = cudaFMA(T(2), x, T(w - 1)); // scale so local grid offset z in [-1,1]
  // insert the auto-generated code which expects z, w args, writes to ker...
  if (upsampfac == 2.0) { // floating point equality is fine here
    using FLT = T;
#include "cufinufft/contrib/ker_horner_allw_loop.inc"
  }
  if (upsampfac == 1.25) { // floating point equality is fine here
    using FLT = T;
#include "cufinufft/contrib/ker_lowupsampfac_horner_allw_loop.inc"
  }
}

// Given grid sizes (via nf123) and bin sizes, compute the number of bins
// along every axis.
template<int ndim>
static inline __host__ __device__ auto get_nbins(cuda::std::array<int, 3> nf123,
                                                 cuda::std::array<int, 3> binsizes) {
  cuda::std::array<int, 3> nbins{1, 1, 1};
  for (int idim = 0; idim < ndim; ++idim)
    nbins[idim] = (nf123[idim] + binsizes[idim] - 1) / binsizes[idim];
  return nbins;
}

static inline __host__ __device__ int nbins_total(const cuda::std::array<int, 3> &nbins) {
  return nbins[0] * nbins[1] * nbins[2];
}

// For the current nonuniform point (given via idx), compute the set of
// start indices of the spreading/interpolation area in the locally stored subgrid.
// Also compute the kernel values to use in spreading/interpolation.
template<typename T, int KEREVALMETH, int ndim, int ns>
static __device__ auto get_kerval_and_local_start(
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

// For the current nonuniform point (given via idx), compute the set of
// start indices of the spreading/interpolation area in the locally stored subgrid.
// Also compute the kernel values to use in spreading/interpolation.
// (Version for nonunifom-points-driven algorithm.)
template<typename T, int KEREVALMETH, int ndim, int ns>
static __device__ auto get_kerval_and_startpos_nuptsdriven(
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
static __device__ auto compute_offset(const int bidx,
                                      const cuda::std::array<int, 3> &nbins,
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

// For the current nonuniform point (given via idx), compute the flat index
// of the bin it falls into.
template<int ndim, typename T>
static __device__ int compute_bin_index(
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

// Given bin sizes and kernel support, compute the size of a padded subgrid.
template<int ndim, int ns>
static __device__ auto get_padded_subgrid_info(const cuda::std::array<int, 3> &binsizes) {
  constexpr auto rounded_ns = ((ns + 1) / 2) * 2;
  cuda::std::array<int, ndim> padded_size;
  int total = 1;
  for (int idim = 0; idim < ndim; ++idim) {
    padded_size[idim] = binsizes[idim] + rounded_ns;
    total *= padded_size[idim];
  }
  return cuda::std::make_tuple(padded_size, total);
}

// Given a flat index in a local padded subgrid, compute its location in the global grid.
template<int ndim, int ns>
static __device__ int output_index_from_flat_local_index(
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

/* ------------------------- Interpolation section ------------------------------ */

// Nupts-driven interpolation kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ FINUFFT_FLATTEN void interp_nupts_driven(
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

// Nupts-driven interpolation CPU driver
template<typename T, int ndim, int ns>
static void cuinterp_nuptsdriven(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                                 const cuda_complex<T> *fw, int blksize) {
  const dim3 threadsPerBlock{
      optimal_interp_block_threads(d_plan.opts.gpu_device_id, d_plan.M), 1u, 1u};
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

// Iterates over all locations in a local subgrid, and for all pairs of
// corresponding local and global pixels, calls the provided function.
// Useful for copying between global and local grids.
template<typename T, int ndim, int ns, typename Func>
static __device__ void shared_mem_copy_helper(cuda::std::array<int, 3> binsizes,
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

// Subprob interpolation kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ FINUFFT_FLATTEN void interp_subprob(
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
      for (int xx = 0; xx < ns; ++xx) cnow += fwshared[ofs0 + xx] * ker[0][xx];
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
            cnowy += fwshared[ofs + xx] * ker[0][xx];
          }
          cnowz += cnowy * ker[1][yy];
        }
        cnow += cnowz * ker[2][zz];
      }
    }
    storeCacheStreaming(c + nuptsidx, cnow);
  }
}

// Subprob interpolation CPU driver
template<typename T, int ndim, int ns>
static void cuinterp_subprob(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
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

/* ------------------------- Spreading section ------------------------------ */

// Nupts-driven spreading kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ FINUFFT_FLATTEN void spread_nupts_driven(
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

// Nupts-driven spreading CPU driver
template<typename T, int ndim, int ns>
static void cuspread_nupts_driven(const cufinufft_plan_t<T> &d_plan,
                                  const cuda_complex<T> *c, cuda_complex<T> *fw,
                                  int blksize) {
  auto &stream = d_plan.stream;

  const dim3 threadsPerBlock{16, 1, 1};
  const dim3 blocks{(unsigned(d_plan.M) + 15) / 16, 1, 1};

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
static __global__ FINUFFT_FLATTEN void calc_bin_size_noghost(
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
static __global__ FINUFFT_FLATTEN void calc_inverse_of_global_sort_idx(
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

// Preparation(?) function for nupts-driven spreading
template<typename T, int ndim>
static void cuspread_nuptsdriven_prop(cufinufft_plan_t<T> &d_plan) {
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

// Subprob spreading kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ FINUFFT_FLATTEN void spread_subprob(
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

// Subprob spreading CPU driver
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

// Preparation(?) function for subprob and output-driven spreading
template<typename T, int ndim>
static void cuspread_subprob_and_OD_prop(cufinufft_plan_t<T> &d_plan) {
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

// Output-driven spreading kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ FINUFFT_FLATTEN void spread_output_driven(
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

// Output-driven spreading CPU driver
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

// Functions for block-gather spreading
// Only implemented for 3D at the moment and potentially obsolete.

static __host__ __device__ int calc_global_index(cuda::std::array<int, 3> idx,
                                                 cuda::std::array<int, 3> on,
                                                 cuda::std::array<int, 3> bn) {
  cuda::std::array<int, 3> oi{idx[0] / bn[0], idx[1] / bn[1], idx[2] / bn[2]};
  return (oi[0] + oi[1] * on[0] + oi[2] * on[1] * on[0]) * (bn[0] * bn[1] * bn[2]) +
         (idx[0] % bn[0] + idx[1] % bn[1] * bn[0] + idx[2] % bn[2] * bn[1] * bn[0]);
}

static __global__ void calc_subprob_3d_v1(cuda::std::array<int, 3> binsperobin,
                                          const int *bin_size, int *num_subprob,
                                          int maxsubprobsize, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    int numnupts        = 0;
    int binsperobin_tot = binsperobin[0] * binsperobin[1] * binsperobin[2];
    for (int b = 0; b < binsperobin_tot; b++) {
      numnupts += bin_size[binsperobin_tot * i + b];
    }
    // FIXME: why is there a hardcoded "float" here?
    num_subprob[i] = ceil(numnupts / (float)maxsubprobsize);
  }
}

static __global__ void map_b_into_subprob_3d_v1(int *d_subprob_to_obin,
                                                const int *d_subprobstartpts,
                                                const int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x)
    for (int j = 0; j < d_numsubprob[i]; j++)
      d_subprob_to_obin[d_subprobstartpts[i] + j] = i;
}

static __global__ void fill_ghost_bins(cuda::std::array<int, 3> binsperobin,
                                       cuda::std::array<int, 3> nobin, int *binsize) {
  constexpr int ndim = 3;

  cuda::std::array<int, 3> bin = {int(threadIdx.x + blockIdx.x * blockDim.x),
                                  int(threadIdx.y + blockIdx.y * blockDim.y),
                                  int(threadIdx.z + blockIdx.z * blockDim.z)};

  cuda::std::array<int, 3> nbin;
  for (int idim = 0; idim < ndim; ++idim) nbin[idim] = nobin[idim] * binsperobin[idim];

  if (bin[0] < nbin[0] && bin[1] < nbin[1] && bin[2] < nbin[2]) {
    int binidx = calc_global_index(bin, nobin, binsperobin);
    cuda::std::array<int, 3> ijk;
    for (int idim = 0; idim < ndim; ++idim) {
      ijk[idim] = bin[idim];
      if (bin[idim] % binsperobin[idim] == 0) {
        ijk[idim] = bin[idim] - 2;
        ijk[idim] = ijk[idim] < 0 ? ijk[idim] + nbin[idim] : ijk[idim];
      }
      if (bin[idim] % binsperobin[idim] == binsperobin[idim] - 1) {
        ijk[idim] = bin[idim] + 2;
        ijk[idim] = (ijk[idim] >= nbin[idim]) ? ijk[idim] - nbin[idim] : ijk[idim];
      }
    }
    int idxtoupdate = calc_global_index(ijk, nobin, binsperobin);
    if (idxtoupdate != binidx) {
      binsize[binidx] = binsize[idxtoupdate];
    }
  }
}

static __global__ void ghost_bin_pts_index(
    cuda::std::array<int, 3> binsperobin, cuda::std::array<int, 3> nobin,
    const int *binsize, int *index, const int *binstartpts, int M) {
  constexpr int ndim = 3;

  cuda::std::array<int, 3> bin = {int(threadIdx.x + blockIdx.x * blockDim.x),
                                  int(threadIdx.y + blockIdx.y * blockDim.y),
                                  int(threadIdx.z + blockIdx.z * blockDim.z)};

  cuda::std::array<int, 3> nbin;
  for (int idim = 0; idim < ndim; ++idim) nbin[idim] = nobin[idim] * binsperobin[idim];

  bool w = false;
  cuda::std::array<int, 3> box;
  if (bin[0] < nbin[0] && bin[1] < nbin[1] && bin[2] < nbin[2]) {
    int binidx = calc_global_index(bin, nobin, binsperobin);
    cuda::std::array<int, 3> ijk;
    for (int idim = 0; idim < ndim; ++idim) {
      box[idim] = 0;
      ijk[idim] = bin[idim];
      if (bin[idim] % binsperobin[idim] == 0) {
        ijk[idim] = bin[idim] - 2;
        box[idim] = (ijk[idim] < 0);
        ijk[idim] = ijk[idim] < 0 ? ijk[idim] + nbin[idim] : ijk[idim];
        w         = true;
      }
      if (bin[idim] % binsperobin[idim] == binsperobin[idim] - 1) {
        ijk[idim] = bin[idim] + 2;
        box[idim] = (ijk[idim] > nbin[idim]) * 2;
        ijk[idim] = (ijk[idim] > nbin[idim]) ? ijk[idim] - nbin[idim] : ijk[idim];
        w         = true;
      }
    }
    int corbinidx = calc_global_index(ijk, nobin, binsperobin);
    if (w) {
      for (int n = 0; n < binsize[binidx]; n++) {
        index[binstartpts[binidx] + n] =
            M * (box[0] + box[1] * 3 + box[2] * 9) + index[binstartpts[corbinidx] + n];
      }
    }
  }
}

template<typename T>
static __global__ void locate_nupts_to_bins_ghost(
    int M, cuda::std::array<int, 3> binsize, cuda::std::array<int, 3> nobin,
    cuda::std::array<int, 3> binsperobin, int *bin_size,
    cuda::std::array<const T *, 3> xyz, int *sortidx, cuda::std::array<int, 3> nf123) {
  int binidx;
  constexpr int ndim = 3;
  cuda::std::array<int, 3> bin;
  int oldidx;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {

    for (int idim = 0; idim < ndim; ++idim) {
      T rescaled = fold_rescale(xyz[idim][i], nf123[idim]);
      bin[idim]  = floor(rescaled / binsize[idim]);
      bin[idim]  = bin[idim] / (binsperobin[idim] - 2) * binsperobin[idim] +
                  (bin[idim] % (binsperobin[idim] - 2) + 1);
    }

    binidx     = calc_global_index(bin, nobin, binsperobin);
    oldidx     = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename T>
static __global__ void calc_inverse_of_global_sort_index_ghost(
    int M, cuda::std::array<int, 3> binsize, cuda::std::array<int, 3> nobin,
    cuda::std::array<int, 3> binsperobin, const int *bin_startpts, const int *sortidx,
    cuda::std::array<const T *, 3> xyz, int *index, cuda::std::array<int, 3> nf123) {
  constexpr int ndim = 3;
  cuda::std::array<int, 3> bin;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {

    for (int idim = 0; idim < ndim; ++idim) {
      T rescaled = fold_rescale(xyz[idim][i], nf123[idim]);
      bin[idim]  = floor(rescaled / binsize[idim]);
      bin[idim]  = bin[idim] / (binsperobin[idim] - 2) * binsperobin[idim] +
                  (bin[idim] % (binsperobin[idim] - 2) + 1);
    }

    int binidx = calc_global_index(bin, nobin, binsperobin);

    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ void spread_3d_block_gather(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw) {
  static_assert(ndim == 3, "unsupported dimensionality");

  T es_c             = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta          = p.spopts.beta;
  T sigma            = p.spopts.upsampfac;
  int maxsubprobsize = p.opts.gpu_maxsubprobsize;

  cuda::std::array<int, 3> obin_size{p.opts.gpu_obinsizex, p.opts.gpu_obinsizey,
                                     p.opts.gpu_obinsizez};
  cuda::std::array<int, 3> bin_size{p.opts.gpu_binsizex, p.opts.gpu_binsizey,
                                    p.opts.gpu_binsizez};
  cuda::std::array<int, ndim> nobin;
  for (size_t idim = 0; idim < ndim; ++idim)
    nobin[idim] = ceil((T)p.nf123[idim] / obin_size[idim]);

  cuda::std::array<int, ndim> binsperobin;
  int binsperobin_tot = 1;
  for (size_t idim = 0; idim < ndim; ++idim) {
    binsperobin[idim] = obin_size[idim] / bin_size[idim] + 2;
    binsperobin_tot *= binsperobin[idim];
  }

  extern __shared__ char sharedbuf[];
  cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;
  const int subpidx         = blockIdx.x;
  const int obidx           = p.subprob_to_bin[subpidx];
  const int bidx            = obidx * binsperobin_tot;

  const int obinsubp_idx = subpidx - p.subprobstartpts[obidx];
  const int ptstart      = p.binstartpts[bidx] + obinsubp_idx * p.opts.gpu_maxsubprobsize;
  const int nupts =
      min(maxsubprobsize, p.binstartpts[bidx + binsperobin_tot] - p.binstartpts[bidx] -
                              obinsubp_idx * p.opts.gpu_maxsubprobsize);

  auto offset = compute_offset<ndim>(obidx, nobin, obin_size);

  const int N = obin_size[0] * obin_size[1] * obin_size[2];

  cuda::std::array<cuda::std::array<T, ns>, 3> ker;

  for (int i = threadIdx.x; i < N; i += blockDim.x) fwshared[i] = {0, 0};

  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    int nidx = p.idxnupts[ptstart + i];
    int b    = nidx / p.M;
    int box[3];
    for (int &d : box) {
      d = b % 3;
      if (d == 1) d = -1;
      if (d == 2) d = 1;
      b = b / 3;
    }
    const int ii = nidx % p.M;
    cuda::std::array<int, 3> start, startnew, endnew;
    for (int idim = 0; idim < ndim; ++idim) {
      const auto rescaled =
          fold_rescale(p.xyz[idim][ii], p.nf123[idim]) + box[idim] * p.nf123[idim];
      auto [start_, end] = interval(ns, rescaled);

      const T pos = T(start_) - rescaled;
      start_ -= offset[idim];
      end -= offset[idim];

      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&ker[idim][0], pos, sigma);
      } else {
        eval_kernel_vec<T, ns>(&ker[idim][0], pos, es_c, es_beta);
      }

      start[idim]    = start_;
      startnew[idim] = start_ < 0 ? 0 : start_;
      endnew[idim]   = end >= obin_size[idim] ? obin_size[idim] - 1 : end;
    }

    const auto cnow = c[ii];
    for (int zz = startnew[2]; zz <= endnew[2]; zz++) {
      const T kervalue3 = ker[2][zz - start[2]];
      for (int yy = startnew[1]; yy <= endnew[1]; yy++) {
        const T kervalue2 = ker[1][yy - start[1]];
        for (int xx = startnew[0]; xx <= endnew[0]; xx++) {
          const auto outidx = xx + yy * obin_size[0] + zz * obin_size[1] * obin_size[0];
          const T kervalue1 = ker[0][xx - start[0]];
          atomicAddComplexShared<T>(fwshared + outidx,
                                    cnow * kervalue1 * kervalue2 * kervalue3);
        }
      }
    }
  }
  __syncthreads();

  /* write to global memory */
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int outidx = 0, stride = 1;
    int tmp = n;
    for (size_t idim = 0; idim < ndim; ++idim) {
      int idx = tmp % obin_size[idim] + offset[idim];
      outidx += idx * stride;
      tmp /= obin_size[idim];
      stride *= p.nf123[idim];
    }
    atomicAddComplexGlobal<T>(fw + outidx, fwshared[n]);
  }
}

template<typename T, int ndim>
static void cuspread3d_blockgather_prop(cufinufft_plan_t<T> &d_plan) {
  if constexpr (ndim == 3) {

    auto &stream = d_plan.stream;
    int M        = d_plan.M;

    int maxsubprobsize                  = d_plan.opts.gpu_maxsubprobsize;
    cuda::std::array<int, 3> o_bin_size = {
        d_plan.opts.gpu_obinsizex, d_plan.opts.gpu_obinsizey, d_plan.opts.gpu_obinsizez};

    if (d_plan.nf123[0] % o_bin_size[0] != 0 || d_plan.nf123[1] % o_bin_size[1] != 0 ||
        d_plan.nf123[2] % o_bin_size[2] != 0) {
      std::cerr << "[cuspread3d_blockgather_prop] error:\n";
      std::cerr << "       mod(nf(1|2|3), opts.gpu_obinsize(x|y|z)) != 0" << std::endl;
      std::cerr << "       (nf1, nf2, nf3) = (" << d_plan.nf123[0] << ", "
                << d_plan.nf123[1] << ", " << d_plan.nf123[2] << ")" << std::endl;
      std::cerr << "       (obinsizex, obinsizey, obinsizez) = (" << o_bin_size[0] << ", "
                << o_bin_size[1] << ", " << o_bin_size[2] << ")" << std::endl;
      throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
    }

    cuda::std::array<int, 3> numobins;
    for (int idim = 0; idim < ndim; ++idim)
      numobins[idim] = ceil((T)d_plan.nf123[idim] / o_bin_size[idim]);

    cuda::std::array<int, 3> bin_size = {
        d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};
    if (o_bin_size[0] % bin_size[0] != 0 || o_bin_size[1] % bin_size[1] != 0 ||
        o_bin_size[2] % bin_size[2] != 0) {
      std::cerr << "[cuspread3d_blockgather_prop] error:\n";
      std::cerr << "      mod(ops.gpu_obinsize(x|y|z), opts.gpu_binsize(x|y|z)) != 0"
                << std::endl;
      std::cerr << "      (binsizex, binsizey, binsizez) = (" << bin_size[0] << ", "
                << bin_size[1] << ", " << bin_size[2] << ")" << std::endl;
      std::cerr << "      (obinsizex, obinsizey, obinsizez) = (" << o_bin_size[0] << ", "
                << o_bin_size[1] << ", " << o_bin_size[2] << ")" << std::endl;
      throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
    }

    cuda::std::array<int, 3> binsperobin;
    cuda::std::array<int, 3> numbins;
    for (int idim = 0; idim < ndim; ++idim) {
      binsperobin[idim] = o_bin_size[idim] / bin_size[idim] + 2;
      numbins[idim]     = numobins[idim] * binsperobin[idim];
    }

    int *d_binsize         = dethrust(d_plan.binsize);
    int *d_sortidx         = dethrust(d_plan.sortidx);
    int *d_binstartpts     = dethrust(d_plan.binstartpts);
    int *d_numsubprob      = dethrust(d_plan.numsubprob);
    int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);

    checkCudaErrors(cudaMemsetAsync(
        d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));

    locate_nupts_to_bins_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size, numobins, binsperobin, d_binsize, d_plan.kxyz, d_sortidx,
        d_plan.nf123);
    THROW_IF_CUDA_ERROR

    dim3 threadsPerBlock = {8, 8, 8};

    dim3 blocks;
    blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
    blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
    blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

    fill_ghost_bins<<<blocks, threadsPerBlock, 0, stream>>>(binsperobin, numobins,
                                                            d_binsize);
    THROW_IF_CUDA_ERROR

    int n = numbins[0] * numbins[1] * numbins[2];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts + 1);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

    checkCudaErrors(cudaMemsetAsync(d_binstartpts, 0, sizeof(int), stream));

    int totalNUpts;
    checkCudaErrors(cudaMemcpyAsync(&totalNUpts, &d_binstartpts[n], sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    d_plan.idxnupts.resize(totalNUpts);

    calc_inverse_of_global_sort_index_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size, numobins, binsperobin, d_binstartpts, d_sortidx, d_plan.kxyz,
        dethrust(d_plan.idxnupts), d_plan.nf123);

    threadsPerBlock = {2, 2, 2};

    blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
    blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
    blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

    ghost_bin_pts_index<<<blocks, threadsPerBlock, 0, stream>>>(
        binsperobin, numobins, d_binsize, dethrust(d_plan.idxnupts), d_binstartpts, M);

    /* --------------------------------------------- */
    //        Determining Subproblem properties      //
    /* --------------------------------------------- */
    n = numobins[0] * numobins[1] * numobins[2];
    calc_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
        binsperobin, d_binsize, d_numsubprob, maxsubprobsize,
        numobins[0] * numobins[1] * numobins[2]);
    THROW_IF_CUDA_ERROR

    n        = numobins[0] * numobins[1] * numobins[2];
    d_ptr    = thrust::device_pointer_cast(d_numsubprob);
    d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

    checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream));

    int totalnumsubprob;
    checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    d_plan.subprob_to_bin.resize(totalnumsubprob);
    map_b_into_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
        dethrust(d_plan.subprob_to_bin), d_subprobstartpts, d_numsubprob, n);

    d_plan.totalnumsubprob = totalnumsubprob;
  } else
    throw int(FINUFFT_ERR_DIM_NOTVALID);
}

template<typename T, int ndim, int ns>
static void cuspread3d_blockgather(const cufinufft_plan_t<T> &d_plan,
                                   const cuda_complex<T> *c, cuda_complex<T> *fw,
                                   int blksize) {
  if constexpr (ndim == 3) {
    size_t sharedplanorysize = d_plan.opts.gpu_obinsizex * d_plan.opts.gpu_obinsizey *
                               d_plan.opts.gpu_obinsizez * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
      std::cerr << "[cuspread3d_blockgather] error: not enough shared memory"
                << std::endl;
      throw int(FINUFFT_ERR_INSUFFICIENT_SHMEM);
    }

    const auto launch = [&](auto kernel) {
      //   cufinufft_set_shared_memory(kernel, ndim, d_plan);
      for (int t = 0; t < blksize; t++) {
        kernel<<<d_plan.totalnumsubprob, 64, sharedplanorysize, d_plan.stream>>>(
            d_plan, c + t * d_plan.M, fw + t * d_plan.nf);
        THROW_IF_CUDA_ERROR
      }
    };
    (d_plan.opts.gpu_kerevalmeth == 1) ? launch(spread_3d_block_gather<T, 1, ndim, ns>)
                                       : launch(spread_3d_block_gather<T, 0, ndim, ns>);
  } else
    throw int(FINUFFT_ERR_DIM_NOTVALID);
}

// End of block-gather spreading

// Dispatchers for spreading preparation, spreading and interpolation

struct SpreadPropDispatcher {
  template<int ndim, typename T> void operator()(cufinufft_plan_t<T> &d_plan) {
    if (d_plan.opts.gpu_method == 1) cuspread_nuptsdriven_prop<T, ndim>(d_plan);
    if (d_plan.opts.gpu_method == 2) cuspread_subprob_and_OD_prop<T, ndim>(d_plan);
    if (d_plan.opts.gpu_method == 3) cuspread_subprob_and_OD_prop<T, ndim>(d_plan);
    if (d_plan.opts.gpu_method == 4) cuspread3d_blockgather_prop<T, ndim>(d_plan);
  }
};

template<typename T> void cuspreadnd_prop(cufinufft_plan_t<T> &plan) {
  using namespace cufinufft::spreadinterp;
  cufinufft::utils::launch_dispatch_ndim<SpreadPropDispatcher, T>(SpreadPropDispatcher(),
                                                                  plan.dim, plan);
}
template void cuspreadnd_prop(cufinufft_plan_t<float> &plan);
template void cuspreadnd_prop(cufinufft_plan_t<double> &plan);

struct SpreadDispatcher {
  template<int ndim, int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                  cuda_complex<T> *fw, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuspread_nupts_driven<T, ndim, ns>(d_plan, c, fw, blksize);
    case 2:
      return cuspread_subprob<T, ndim, ns>(d_plan, c, fw, blksize);
    case 3:
      return cuspread_output_driven<T, ndim, ns>(d_plan, c, fw, blksize);
    case 4:
      return cuspread3d_blockgather<T, ndim, ns>(d_plan, c, fw, blksize);
    default:
      std::cerr << "[cuspreadnd] error: incorrect method, should be 1, 2 or 3\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

template<typename T>
void cuspreadnd(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                cuda_complex<T> *fw, int blksize) {
  cufinufft::utils::launch_dispatch_ndim_ns<cufinufft::spreadinterp::SpreadDispatcher, T>(
      cufinufft::spreadinterp::SpreadDispatcher(), d_plan.dim, d_plan.spopts.nspread,
      d_plan, c, fw, blksize);
}
template void cuspreadnd(const cufinufft_plan_t<float> &d_plan,
                         const cuda_complex<float> *c, cuda_complex<float> *fw,
                         int blksize);
template void cuspreadnd(const cufinufft_plan_t<double> &d_plan,
                         const cuda_complex<double> *c, cuda_complex<double> *fw,
                         int blksize);

struct InterpDispatcher {
  template<int ndim, int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                  const cuda_complex<T> *fw, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuinterp_nuptsdriven<T, ndim, ns>(d_plan, c, fw, blksize);
    case 2:
      return cuinterp_subprob<T, ndim, ns>(d_plan, c, fw, blksize);
    default:
      std::cerr << "[cuinterpnd] error: incorrect method, should be 1 or 2\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

template<typename T>
void cuinterpnd(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                const cuda_complex<T> *fw, int blksize) {
  cufinufft::utils::launch_dispatch_ndim_ns<cufinufft::spreadinterp::InterpDispatcher, T>(
      cufinufft::spreadinterp::InterpDispatcher(), d_plan.dim, d_plan.spopts.nspread,
      d_plan, c, fw, blksize);
}
template void cuinterpnd(const cufinufft_plan_t<float> &d_plan, cuda_complex<float> *c,
                         const cuda_complex<float> *fw, int blksize);
template void cuinterpnd(const cufinufft_plan_t<double> &d_plan, cuda_complex<double> *c,
                         const cuda_complex<double> *fw, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

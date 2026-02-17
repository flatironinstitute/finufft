#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>

#include <cufinufft/common.h>
#include <cufinufft/memtransfer.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>
#include <cufinufft/intrinsics.h>

using namespace cufinufft::memtransfer;
using namespace cufinufft::common;
using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

template<typename T, int KEREVALMETH, int ns>
static __global__ void interp_3d_nupts_driven(
    const T *x, const T *y, const T *z, cuda_complex<T> *c, const cuda_complex<T> *fw,
    int M, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma, int *idxnupts) {

  T ker1[ns];
  T ker2[ns];
  T ker3[ns];

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx   = loadReadOnly(idxnupts + i);
    const auto x_rescaled = fold_rescale(loadReadOnly(x + nuptsidx), nf1);
    const auto y_rescaled = fold_rescale(loadReadOnly(y + nuptsidx), nf2);
    const auto z_rescaled = fold_rescale(loadReadOnly(z + nuptsidx), nf3);

    const auto [xstart, xend] = interval(ns, x_rescaled);
    const auto [ystart, yend] = interval(ns, y_rescaled);
    const auto [zstart, zend] = interval(ns, z_rescaled);

    const T x1 = T(xstart) - x_rescaled;
    const T y1 = T(ystart) - y_rescaled;
    const T z1 = T(zstart) - z_rescaled;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
      eval_kernel_vec_horner<T, ns>(ker3, z1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker3, z1, es_c, es_beta);
    }

    cuda_complex<T> cnow{0, 0};
    for (int zz = zstart; zz <= zend; zz++) {
      const auto kervalue3 = ker3[zz - zstart];
      const auto iz        = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
      for (int yy = ystart; yy <= yend; yy++) {
        const auto kervalue2 = ker2[yy - ystart];
        const int iy         = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
        for (int xx = xstart; xx <= xend; xx++) {
          const auto ix        = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
          const auto inidx     = ix + iy * nf1 + iz * nf2 * nf1;
          const auto kervalue1 = ker1[xx - xstart];
          const auto kervalue  = kervalue1 * kervalue2 * kervalue3;
          cnow += {fw[inidx] * kervalue};
        }
      }
    }
    c[idxnupts[i]] = cnow;
  }
}

/* Kernels for SubProb Method */
template<typename T, int KEREVALMETH, int ns>
static __global__ void interp_3d_subprob(
    const T *x, const T *y, const T *z, cuda_complex<T> *c, const cuda_complex<T> *fw,
    int M, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma, const int *binstartpts,
    const int *bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
    const int *subprob_to_bin, const int *subprobstartpts, const int *numsubprob,
    int maxsubprobsize, int nbinx, int nbiny, int nbinz, const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  T ker1[ns];
  T ker2[ns];
  T ker3[ns];

  const auto subpidx     = blockIdx.x;
  const auto bidx        = subprob_to_bin[subpidx];
  const auto binsubp_idx = subpidx - subprobstartpts[bidx];
  const auto ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const auto nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const auto xoffset = (bidx % nbinx) * bin_size_x;
  const auto yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;
  const auto zoffset = (bidx / (nbinx * nbiny)) * bin_size_z;

  const T ns_2f         = ns * T(.5);
  const auto ns_2       = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;

  const int N =
      (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns) * (bin_size_z + rounded_ns);

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int i   = n % (bin_size_x + rounded_ns);
    int j   = (n / (bin_size_x + rounded_ns)) % (bin_size_y + rounded_ns);
    int k   = n / ((bin_size_x + rounded_ns) * (bin_size_y + rounded_ns));
    auto ix = xoffset - ns_2 + i;
    auto iy = yoffset - ns_2 + j;
    auto iz = zoffset - ns_2 + k;
    if (ix < (nf1 + ns_2) && iy < (nf2 + ns_2) && iz < (nf3 + ns_2)) {
      ix                = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy                = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      iz                = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
      const auto outidx = ix + iy * nf1 + iz * nf1 * nf2;
      int sharedidx     = i + j * (bin_size_x + rounded_ns) +
                      k * (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns);
      fwshared[sharedidx] = fw[outidx];
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx         = ptstart + i;
    const auto x_rescaled = fold_rescale(x[idxnupts[idx]], nf1);
    const auto y_rescaled = fold_rescale(y[idxnupts[idx]], nf2);
    const auto z_rescaled = fold_rescale(z[idxnupts[idx]], nf3);
    cuda_complex<T> cnow{0, 0};

    auto [xstart, xend] = interval(ns, x_rescaled);
    auto [ystart, yend] = interval(ns, y_rescaled);
    auto [zstart, zend] = interval(ns, z_rescaled);

    const T x1 = T(xstart) - x_rescaled;
    const T y1 = T(ystart) - y_rescaled;
    const T z1 = T(zstart) - z_rescaled;

    xstart -= xoffset;
    ystart -= yoffset;
    zstart -= zoffset;

    xend -= xoffset;
    yend -= yoffset;
    zend -= zoffset;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
      eval_kernel_vec_horner<T, ns>(ker3, z1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker3, z1, es_c, es_beta);
    }

    for (int zz = zstart; zz <= zend; zz++) {
      const auto kervalue3 = ker3[zz - zstart];
      const auto iz        = zz + ns_2;
      for (int yy = ystart; yy <= yend; yy++) {
        const auto kervalue2 = ker2[yy - ystart];
        const auto iy        = yy + ns_2;
        for (int xx = xstart; xx <= xend; xx++) {
          const auto ix     = xx + ns_2;
          const auto outidx = ix + iy * (bin_size_x + rounded_ns) +
                              iz * (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns);
          const auto kervalue1 = ker1[xx - xstart];
          const auto kervalue  = kervalue1 * kervalue2 * kervalue3;
          cnow += {fwshared[outidx] * kervalue};
        }
      }
    }
    c[idxnupts[idx]] = cnow;
  }
}

template<typename T, int ns>
static int cuinterp3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize) {
  const auto stream = d_plan->stream;

  T es_c    = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta = d_plan->spopts.beta;
  T sigma   = d_plan->spopts.upsampfac;

  int *d_idxnupts = d_plan->idxnupts;

  T *d_kx               = d_plan->kxyz[0];
  T *d_ky               = d_plan->kxyz[1];
  T *d_kz               = d_plan->kxyz[2];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  const dim3 threadsPerBlock{
      std::min(optimal_block_threads(d_plan->opts.gpu_device_id), (unsigned)M), 1u, 1u};
  const dim3 blocks{(M + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1};

  if (d_plan->opts.gpu_kerevalmeth) {
    for (int t = 0; t < blksize; t++) {
      interp_3d_nupts_driven<T, 1, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      interp_3d_nupts_driven<T, 0, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

template<typename T, int ns>
static int cuinterp3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                       int blksize) {
  auto &stream = d_plan->stream;

  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan->opts.gpu_binsizex;
  int bin_size_y = d_plan->opts.gpu_binsizey;
  int bin_size_z = d_plan->opts.gpu_binsizez;
  int numbins[3];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);
  numbins[2] = ceil((T)nf3 / bin_size_z);

  T *d_kx               = d_plan->kxyz[0];
  T *d_ky               = d_plan->kxyz[1];
  T *d_kz               = d_plan->kxyz[2];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;
  int *d_subprob_to_bin  = d_plan->subprob_to_bin;
  int totalnumsubprob    = d_plan->totalnumsubprob;

  T sigma                      = d_plan->spopts.upsampfac;
  T es_c                       = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta                    = d_plan->spopts.beta;
  const auto sharedplanorysize = shared_memory_required<T>(
      3, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);

  if (d_plan->opts.gpu_kerevalmeth == 1) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(interp_3d_subprob<T, 1, ns>, 3, *d_plan)) {
      return finufft_err;
    }
    for (int t = 0; t < blksize; t++) {
      interp_3d_subprob<T, 1, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
          bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
          numbins[0], numbins[1], numbins[2], d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(interp_3d_subprob<T, 0, ns>, 3, *d_plan)) {
      return finufft_err;
    }
    for (int t = 0; t < blksize; t++) {
      interp_3d_subprob<T, 0, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
          bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
          numbins[0], numbins[1], numbins[2], d_idxnupts);
      RETURN_IF_CUDA_ERROR
    }
  }

  return 0;
}

// Functor to handle function selection (nuptsdriven vs subprob)
struct Interp3DDispatcher {
  template<int ns, typename T>
  int operator()(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                 int blksize) const {
    switch (d_plan->opts.gpu_method) {
    case 1:
      return cuinterp3d_nuptsdriven<T, ns>(nf1, nf2, nf3, M, d_plan, blksize);
    case 2:
      return cuinterp3d_subprob<T, ns>(nf1, nf2, nf3, M, d_plan, blksize);
    default:
      std::cerr << "[cuinterp3d] error: incorrect method, should be 1 or 2\n";
      return FINUFFT_ERR_METHOD_NOTVALID;
    }
  }
};

// Updated cuinterp3d using generic dispatch
template<typename T> int cuinterp3d(cufinufft_plan_t<T> *d_plan, int blksize) {
  /*
    A wrapper for different interpolation methods.

    Methods available:
        (1) Non-uniform points driven
        (2) Subproblem

    Melody Shih 07/25/19

    Now the function is updated to dispatch based on ns. This is to avoid alloca which
    it seems slower according to the MRI community.
    Marco Barbone 01/30/25
  */
  return launch_dispatch_ns<Interp3DDispatcher, T>(
      Interp3DDispatcher(), d_plan->spopts.nspread, d_plan->nf123[0], d_plan->nf123[1], d_plan->nf123[2],
      d_plan->M, d_plan, blksize);
}
template int cuinterp3d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template int cuinterp3d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

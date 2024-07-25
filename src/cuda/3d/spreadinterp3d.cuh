#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cufinufft/contrib/helper_cuda.h>

#include <cuda_runtime.h>
#include <cufinufft/defs.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/types.h>
#include <cufinufft/utils.h>

using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {
/* ---------------------- 3d Spreading Kernels -------------------------------*/
/* Kernels for bin sort NUpts */

template<typename T>
__global__ void calc_bin_size_noghost_3d(int M, int nf1, int nf2, int nf3, int bin_size_x,
                                         int bin_size_y, int bin_size_z, int nbinx,
                                         int nbiny, int nbinz, int *bin_size, const T *x,
                                         const T *y, const T *z, int *sortidx) {
  int binidx, binx, biny, binz;
  int oldidx;
  T x_rescaled, y_rescaled, z_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    y_rescaled = fold_rescale(y[i], nf2);
    z_rescaled = fold_rescale(z[i], nf3);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;

    biny = floor(y_rescaled / bin_size_y);
    biny = biny >= nbiny ? biny - 1 : biny;
    biny = biny < 0 ? 0 : biny;

    binz       = floor(z_rescaled / bin_size_z);
    binz       = binz >= nbinz ? binz - 1 : binz;
    binz       = binz < 0 ? 0 : binz;
    binidx     = binx + biny * nbinx + binz * nbinx * nbiny;
    oldidx     = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename T>
__global__ void calc_inverse_of_global_sort_index_3d(
    int M, int bin_size_x, int bin_size_y, int bin_size_z, int nbinx, int nbiny,
    int nbinz, const int *bin_startpts, const int *sortidx, const T *x, const T *y,
    const T *z, int *index, int nf1, int nf2, int nf3) {
  int binx, biny, binz;
  int binidx;
  T x_rescaled, y_rescaled, z_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    y_rescaled = fold_rescale(y[i], nf2);
    z_rescaled = fold_rescale(z[i], nf3);
    binx       = floor(x_rescaled / bin_size_x);
    binx       = binx >= nbinx ? binx - 1 : binx;
    binx       = binx < 0 ? 0 : binx;
    biny       = floor(y_rescaled / bin_size_y);
    biny       = biny >= nbiny ? biny - 1 : biny;
    biny       = biny < 0 ? 0 : biny;
    binz       = floor(z_rescaled / bin_size_z);
    binz       = binz >= nbinz ? binz - 1 : binz;
    binz       = binz < 0 ? 0 : binz;
    binidx     = common::calc_global_index_v2(binx, biny, binz, nbinx, nbiny, nbinz);

    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

/* Kernels for NUptsdriven method */
template<typename T, int KEREVALMETH>
__global__ void spread_3d_nupts_driven(const T *x, const T *y, const T *z,
                                       const cuda_complex<T> *c, cuda_complex<T> *fw,
                                       int M, int ns, int nf1, int nf2, int nf3, T es_c,
                                       T es_beta, T sigma, const int *idxnupts) {
#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns * 3);
  auto *__restrict__ ker1 = ker;
  auto *__restrict__ ker2 = ker + ns;
  auto *__restrict__ ker3 = ker + ns + ns;
#else
  T ker1[MAX_NSPREAD];
  T ker2[MAX_NSPREAD];
  T ker3[MAX_NSPREAD];
#endif
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto x_rescaled = fold_rescale(x[idxnupts[i]], nf1);
    const auto y_rescaled = fold_rescale(y[idxnupts[i]], nf2);
    const auto z_rescaled = fold_rescale(z[idxnupts[i]], nf3);

    const auto [xstart, xend] = interval(ns, x_rescaled);
    const auto [ystart, yend] = interval(ns, y_rescaled);
    const auto [zstart, zend] = interval(ns, z_rescaled);

    const auto x1 = T(xstart) - x_rescaled;
    const auto y1 = T(ystart) - y_rescaled;
    const auto z1 = T(zstart) - z_rescaled;

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
      eval_kernel_vec_horner(ker2, y1, ns, sigma);
      eval_kernel_vec_horner(ker3, z1, ns, sigma);
    } else {
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
      eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
      eval_kernel_vec(ker3, z1, ns, es_c, es_beta);
    }

    for (int zz = zstart; zz <= zend; zz++) {
      const auto ker3val = ker3[zz - zstart];
      for (int yy = ystart; yy <= yend; yy++) {
        const auto ker2val = ker2[yy - ystart];
        for (int xx = xstart; xx <= xend; xx++) {
          const int ix        = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
          const int iy        = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
          const int iz        = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
          const int outidx    = ix + iy * nf1 + iz * nf1 * nf2;
          const auto ker1val  = ker1[xx - xstart];
          const auto kervalue = ker1val * ker2val * ker3val;
          const cuda_complex<T> res{c[idxnupts[i]].x * kervalue,
                                    c[idxnupts[i]].y * kervalue};
          atomicAddComplexGlobal<T>(fw + outidx, res);
        }
      }
    }
  }
}

/* Kernels for Subprob method */
template<typename T, int KEREVALMETH>
__global__ void spread_3d_subprob(
    T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M, int ns, int nf1,
    int nf2, int nf3, T sigma, T es_c, T es_beta, int *binstartpts, int *bin_size,
    int bin_size_x, int bin_size_y, int bin_size_z, int *subprob_to_bin,
    int *subprobstartpts, int *numsubprob, int maxsubprobsize, int nbinx, int nbiny,
    int nbinz, int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  const int bidx        = subprob_to_bin[blockIdx.x];
  const int binsubp_idx = blockIdx.x - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const int xoffset = (bidx % nbinx) * bin_size_x;
  const int yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;
  const int zoffset = (bidx / (nbinx * nbiny)) * bin_size_z;

  const T ns_2f         = ns * T(.5);
  const auto ns_2       = (ns + 1) / 2;
  const auto rounded_ns = ns_2 * 2;

  const int N =
      (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns) * (bin_size_z + rounded_ns);

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }
  __syncthreads();
#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns * 3);
  auto *__restrict__ ker1 = ker;
  auto *__restrict__ ker2 = ker + ns;
  auto *__restrict__ ker3 = ker + ns + ns;
#else
  T ker1[MAX_NSPREAD];
  T ker2[MAX_NSPREAD];
  T ker3[MAX_NSPREAD];
#endif

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int nuptsidx    = idxnupts[ptstart + i];
    const auto x_rescaled = fold_rescale(x[nuptsidx], nf1);
    const auto y_rescaled = fold_rescale(y[nuptsidx], nf2);
    const auto z_rescaled = fold_rescale(z[nuptsidx], nf3);
    const auto cnow       = c[nuptsidx];
    auto [xstart, xend]   = interval(ns, x_rescaled);
    auto [ystart, yend]   = interval(ns, y_rescaled);
    auto [zstart, zend]   = interval(ns, z_rescaled);

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
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
      eval_kernel_vec_horner(ker2, y1, ns, sigma);
      eval_kernel_vec_horner(ker3, z1, ns, sigma);
    } else {
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
      eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
      eval_kernel_vec(ker3, z1, ns, es_c, es_beta);
    }

    for (int zz = zstart; zz <= zend; zz++) {
      const T kervalue3 = ker3[zz - zstart];
      const int iz      = zz + ns_2;
      if (iz >= (bin_size_z + (int)rounded_ns) || iz < 0) break;
      for (int yy = ystart; yy <= yend; yy++) {
        const T kervalue2 = ker2[yy - ystart];
        const int iy      = yy + ns_2;
        if (iy >= (bin_size_y + (int)rounded_ns) || iy < 0) break;
        for (int xx = xstart; xx <= xend; xx++) {
          const int ix = xx + ns_2;
          if (ix >= (bin_size_x + (int)rounded_ns) || ix < 0) break;
          const int outidx = ix + iy * (bin_size_x + rounded_ns) +
                             iz * (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns);
          const auto kervalue = ker1[xx - xstart] * kervalue2 * kervalue3;
          const cuda_complex<T> res{cnow.x * kervalue, cnow.y * kervalue};
          atomicAddComplexShared<T>(fwshared + outidx, res);
        }
      }
    }
  }
  __syncthreads();

  /* write to global memory */
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    const int i = n % (bin_size_x + rounded_ns);
    const int j = (n / (bin_size_x + rounded_ns)) % (bin_size_y + rounded_ns);
    const int k = n / ((bin_size_x + rounded_ns) * (bin_size_y + rounded_ns));

    int ix = xoffset - ns_2 + i;
    int iy = yoffset - ns_2 + j;
    int iz = zoffset - ns_2 + k;

    if (ix < (nf1 + ns_2) && iy < (nf2 + ns_2) && iz < (nf3 + ns_2)) {
      ix                  = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy                  = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      iz                  = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
      const int outidx    = ix + iy * nf1 + iz * nf1 * nf2;
      const int sharedidx = i + j * (bin_size_x + rounded_ns) +
                            k * (bin_size_x + rounded_ns) * (bin_size_y + rounded_ns);
      atomicAddComplexGlobal<T>(fw + outidx, fwshared[sharedidx]);
    }
  }
}

/* Kernels for BlockGather Method */
template<typename T>
__global__ void locate_nupts_to_bins_ghost(
    int M, int bin_size_x, int bin_size_y, int bin_size_z, int nobinx, int nobiny,
    int nobinz, int binsperobinx, int binsperobiny, int binsperobinz, int *bin_size,
    const T *x, const T *y, const T *z, int *sortidx, int nf1, int nf2, int nf3) {
  int binidx, binx, biny, binz;
  int oldidx;
  T x_rescaled, y_rescaled, z_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    y_rescaled = fold_rescale(y[i], nf2);
    z_rescaled = fold_rescale(z[i], nf3);
    binx       = floor(x_rescaled / bin_size_x);
    biny       = floor(y_rescaled / bin_size_y);
    binz       = floor(z_rescaled / bin_size_z);
    binx = binx / (binsperobinx - 2) * binsperobinx + (binx % (binsperobinx - 2) + 1);
    biny = biny / (binsperobiny - 2) * binsperobiny + (biny % (binsperobiny - 2) + 1);
    binz = binz / (binsperobinz - 2) * binsperobinz + (binz % (binsperobinz - 2) + 1);

    binidx     = common::calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz,
                                           binsperobinx, binsperobiny, binsperobinz);
    oldidx     = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename T>
__global__ void calc_inverse_of_global_sort_index_ghost(
    int M, int bin_size_x, int bin_size_y, int bin_size_z, int nobinx, int nobiny,
    int nobinz, int binsperobinx, int binsperobiny, int binsperobinz, int *bin_startpts,
    const int *sortidx, const T *x, const T *y, const T *z, int *index, int nf1, int nf2,
    int nf3) {
  int binx, biny, binz;
  int binidx;
  T x_rescaled, y_rescaled, z_rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    x_rescaled = fold_rescale(x[i], nf1);
    y_rescaled = fold_rescale(y[i], nf2);
    z_rescaled = fold_rescale(z[i], nf3);
    binx       = floor(x_rescaled / bin_size_x);
    biny       = floor(y_rescaled / bin_size_y);
    binz       = floor(z_rescaled / bin_size_z);
    binx = binx / (binsperobinx - 2) * binsperobinx + (binx % (binsperobinx - 2) + 1);
    biny = biny / (binsperobiny - 2) * binsperobiny + (biny % (binsperobiny - 2) + 1);
    binz = binz / (binsperobinz - 2) * binsperobinz + (binz % (binsperobinz - 2) + 1);

    binidx = common::calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz,
                                       binsperobinx, binsperobiny, binsperobinz);

    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

template<typename T, int KEREVALMETH>
__global__ void spread_3d_block_gather(
    const T *x, const T *y, const T *z, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, int ns, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma,
    const int *binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
    int binsperobin, int *subprob_to_bin, const int *subprobstartpts, int maxsubprobsize,
    int nobinx, int nobiny, int nobinz, const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  cuda_complex<T> *fwshared = (cuda_complex<T> *)sharedbuf;
  const int subpidx         = blockIdx.x;
  const int obidx           = subprob_to_bin[subpidx];
  const int bidx            = obidx * binsperobin;

  const int obinsubp_idx = subpidx - subprobstartpts[obidx];
  const int ptstart      = binstartpts[bidx] + obinsubp_idx * maxsubprobsize;
  const int nupts =
      min(maxsubprobsize, binstartpts[bidx + binsperobin] - binstartpts[bidx] -
                              obinsubp_idx * maxsubprobsize);

  const int xoffset = (obidx % nobinx) * obin_size_x;
  const int yoffset = (obidx / nobinx) % nobiny * obin_size_y;
  const int zoffset = (obidx / (nobinx * nobiny)) * obin_size_z;

  const int N = obin_size_x * obin_size_y * obin_size_z;

#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns * 3);
  auto *__restrict__ ker1 = ker;
  auto *__restrict__ ker2 = ker + ns;
  auto *__restrict__ ker3 = ker + ns + ns;
#else
  T ker1[MAX_NSPREAD];
  T ker2[MAX_NSPREAD];
  T ker3[MAX_NSPREAD];
#endif
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    int nidx = idxnupts[ptstart + i];
    int b    = nidx / M;
    int box[3];
    for (int &d : box) {
      d = b % 3;
      if (d == 1) d = -1;
      if (d == 2) d = 1;
      b = b / 3;
    }
    const int ii          = nidx % M;
    const auto x_rescaled = fold_rescale(x[ii], nf1) + box[0] * nf1;
    const auto y_rescaled = fold_rescale(y[ii], nf2) + box[1] * nf2;
    const auto z_rescaled = fold_rescale(z[ii], nf3) + box[2] * nf3;
    const auto cnow       = c[ii];
    auto [xstart, xend]   = interval(ns, x_rescaled);
    auto [ystart, yend]   = interval(ns, y_rescaled);
    auto [zstart, zend]   = interval(ns, z_rescaled);

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
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
      eval_kernel_vec_horner(ker2, y1, ns, sigma);
      eval_kernel_vec_horner(ker3, z1, ns, sigma);
    } else {
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
      eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
      eval_kernel_vec(ker3, z1, ns, es_c, es_beta);
    }

    const auto xstartnew = xstart < 0 ? 0 : xstart;
    const auto ystartnew = ystart < 0 ? 0 : ystart;
    const auto zstartnew = zstart < 0 ? 0 : zstart;
    const auto xendnew   = xend >= obin_size_x ? obin_size_x - 1 : xend;
    const auto yendnew   = yend >= obin_size_y ? obin_size_y - 1 : yend;
    const auto zendnew   = zend >= obin_size_z ? obin_size_z - 1 : zend;

    for (int zz = zstartnew; zz <= zendnew; zz++) {
      const T kervalue3 = ker3[zz - zstart];
      for (int yy = ystartnew; yy <= yendnew; yy++) {
        const T kervalue2 = ker2[yy - ystart];
        for (int xx = xstartnew; xx <= xendnew; xx++) {
          const auto outidx = xx + yy * obin_size_x + zz * obin_size_y * obin_size_x;
          const T kervalue1 = ker1[xx - xstart];
          const cuda_complex<T> res{cnow.x * kervalue1 * kervalue2 * kervalue3,
                                    cnow.y * kervalue1 * kervalue2 * kervalue3};
          atomicAddComplexShared<T>(fwshared + outidx, res);
        }
      }
    }
  }
  __syncthreads();
  /* write to global memory */
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int i = n % obin_size_x;
    int j = (n / obin_size_x) % obin_size_y;
    int k = n / (obin_size_x * obin_size_y);

    const auto ix     = xoffset + i;
    const auto iy     = yoffset + j;
    const auto iz     = zoffset + k;
    const auto outidx = ix + iy * nf1 + iz * nf1 * nf2;
    atomicAdd(&fw[outidx].x, fwshared[n].x);
    atomicAdd(&fw[outidx].y, fwshared[n].y);
  }
}

/* ---------------------- 3d Interpolation Kernels ---------------------------*/
/* Kernels for NUptsdriven Method */
template<typename T, int KEREVALMETH>
__global__ void interp_3d_nupts_driven(
    const T *x, const T *y, const T *z, cuda_complex<T> *c, const cuda_complex<T> *fw,
    int M, int ns, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma, int *idxnupts) {
#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns * 3);
  auto *__restrict__ ker1 = ker;
  auto *__restrict__ ker2 = ker + ns;
  auto *__restrict__ ker3 = ker + ns + ns;
#else
  T ker1[MAX_NSPREAD];
  T ker2[MAX_NSPREAD];
  T ker3[MAX_NSPREAD];
#endif
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    const auto x_rescaled = fold_rescale(x[idxnupts[i]], nf1);
    const auto y_rescaled = fold_rescale(y[idxnupts[i]], nf2);
    const auto z_rescaled = fold_rescale(z[idxnupts[i]], nf3);

    const auto [xstart, xend] = interval(ns, x_rescaled);
    const auto [ystart, yend] = interval(ns, y_rescaled);
    const auto [zstart, zend] = interval(ns, z_rescaled);

    const auto x1 = T(xstart) - x_rescaled;
    const auto y1 = T(ystart) - y_rescaled;
    const auto z1 = T(zstart) - z_rescaled;

    cuda_complex<T> cnow{0, 0};

    if constexpr (KEREVALMETH == 1) {
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
      eval_kernel_vec_horner(ker2, y1, ns, sigma);
      eval_kernel_vec_horner(ker3, z1, ns, sigma);
    } else {
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
      eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
      eval_kernel_vec(ker3, z1, ns, es_c, es_beta);
    }

    for (int zz = zstart; zz <= zend; zz++) {
      const auto kervalue3 = ker3[zz - zstart];
      int iz               = zz < 0 ? zz + nf3 : (zz > nf3 - 1 ? zz - nf3 : zz);
      for (int yy = ystart; yy <= yend; yy++) {
        const auto kervalue2 = ker2[yy - ystart];
        int iy               = yy < 0 ? yy + nf2 : (yy > nf2 - 1 ? yy - nf2 : yy);
        for (int xx = xstart; xx <= xend; xx++) {
          const int ix         = xx < 0 ? xx + nf1 : (xx > nf1 - 1 ? xx - nf1 : xx);
          const int inidx      = ix + iy * nf1 + iz * nf2 * nf1;
          const auto kervalue1 = ker1[xx - xstart];
          cnow.x += fw[inidx].x * kervalue1 * kervalue2 * kervalue3;
          cnow.y += fw[inidx].y * kervalue1 * kervalue2 * kervalue3;
        }
      }
    }
    c[idxnupts[i]] = cnow;
  }
}

/* Kernels for SubProb Method */
template<typename T, int KEREVALMETH>
__global__ void interp_3d_subprob(
    const T *x, const T *y, const T *z, cuda_complex<T> *c, const cuda_complex<T> *fw,
    int M, int ns, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma,
    const int *binstartpts, const int *bin_size, int bin_size_x, int bin_size_y,
    int bin_size_z, const int *subprob_to_bin, const int *subprobstartpts,
    const int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int nbinz,
    const int *idxnupts) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

#if ALLOCA_SUPPORTED
  auto ker                = (T *)alloca(sizeof(T) * ns * 3);
  auto *__restrict__ ker1 = ker;
  auto *__restrict__ ker2 = ker + ns;
  auto *__restrict__ ker3 = ker + ns + ns;
#else
  T ker1[MAX_NSPREAD];
  T ker2[MAX_NSPREAD];
  T ker3[MAX_NSPREAD];
#endif

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
      eval_kernel_vec_horner(ker1, x1, ns, sigma);
      eval_kernel_vec_horner(ker2, y1, ns, sigma);
      eval_kernel_vec_horner(ker3, z1, ns, sigma);
    } else {
      eval_kernel_vec(ker1, x1, ns, es_c, es_beta);
      eval_kernel_vec(ker2, y1, ns, es_c, es_beta);
      eval_kernel_vec(ker3, z1, ns, es_c, es_beta);
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
          cnow.x += fwshared[outidx].x * kervalue1 * kervalue2 * kervalue3;
          cnow.y += fwshared[outidx].y * kervalue1 * kervalue2 * kervalue3;
        }
      }
    }
    c[idxnupts[idx]] = cnow;
  }
}

} // namespace spreadinterp
} // namespace cufinufft

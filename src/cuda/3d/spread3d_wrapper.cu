#include <cassert>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cufinufft/common.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>
#include <cufinufft/intrinsics.h>

using namespace cufinufft::common;
using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

using cuda::std::dextents;
using cuda::std::dynamic_extent;
using cuda::std::extents;
using cuda::std::mdspan;
using cuda::std::span;

template<typename T>
static __global__ void calc_bin_size_noghost_3d(int M, int nf1, int nf2, int nf3, int bin_size_x,
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
static __global__ void calc_inverse_of_global_sort_index_3d(
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

template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_3d_nupts_driven(
    const T *x, const T *y, const T *z, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma, const int *idxnupts) {
  T ker1[ns];
  T ker2[ns];
  T ker3[ns];
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
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
      eval_kernel_vec_horner<T, ns>(ker3, z1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker3, z1, es_c, es_beta);
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

/* Kernels for Output Driven method */
template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_3d_output_driven(
    T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1, int nf2,
    int nf3, T sigma, T es_c, T es_beta, int *binstartpts, int *bin_size, int bin_size_x,
    int bin_size_y, int bin_size_z, int *subprob_to_bin, int *subprobstartpts,
    int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int nbinz, int *idxnupts,
    int np) {
  extern __shared__ char sharedbuf[];

  static constexpr auto ns_2f      = T(ns * .5);
  static constexpr auto ns_2       = (ns + 1) / 2;
  static constexpr auto rounded_ns = ns_2 * 2;

  const auto padded_size_x = bin_size_x + rounded_ns;
  const auto padded_size_y = bin_size_y + rounded_ns;
  const auto padded_size_z = bin_size_z + rounded_ns;

  const int bidx        = subprob_to_bin[blockIdx.x];
  const int binsubp_idx = blockIdx.x - subprobstartpts[bidx];
  const int ptstart     = binstartpts[bidx] + binsubp_idx * maxsubprobsize;
  const int nupts = min(maxsubprobsize, bin_size[bidx] - binsubp_idx * maxsubprobsize);

  const int xoffset = (bidx % nbinx) * bin_size_x;
  const int yoffset = ((bidx / nbinx) % nbiny) * bin_size_y;
  const int zoffset = (bidx / (nbinx * nbiny)) * bin_size_z;

  auto kerevals = mdspan<T, extents<int, dynamic_extent, 3, ns>>((T *)sharedbuf, np);
  const auto c_kerevals =
      mdspan<const T, extents<int, dynamic_extent, 3, ns>>((T *)sharedbuf, np);
  // sharedbuf + size of kerevals in bytes
  // Offset pointer into sharedbuf after kerevals
  // Create span using pointer + size

  auto nupts_sm = span(
      reinterpret_cast<cuda_complex<T> *>(kerevals.data_handle() + kerevals.size()),
      np);

  auto shift = span(reinterpret_cast<int3 *>(nupts_sm.data() + nupts_sm.size()), np);

  auto local_subgrid = mdspan<cuda_complex<T>, dextents<int, 3>>(
      reinterpret_cast<cuda_complex<T> *>(shift.data() + shift.size()), padded_size_z,
      padded_size_y, padded_size_x);

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid.size(); i += blockDim.x) {
    local_subgrid.data_handle()[i] = {0, 0};
  }
  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx = loadReadOnly(idxnupts + ptstart + i + batch_begin);
      // index of the current point within the batch
      const auto x_rescaled = fold_rescale(loadReadOnly(x + nuptsidx), nf1);
      const auto y_rescaled = fold_rescale(loadReadOnly(y + nuptsidx), nf2);
      const auto z_rescaled = fold_rescale(loadReadOnly(z + nuptsidx), nf3);
      nupts_sm[i]           = loadCacheStreaming(c + nuptsidx);
      const auto xstart     = int(std::ceil(x_rescaled - ns_2f));
      const auto ystart     = int(std::ceil(y_rescaled - ns_2f));
      const auto zstart     = int(std::ceil(z_rescaled - ns_2f));
      const T x1            = T(xstart) - x_rescaled;
      const T y1            = T(ystart) - y_rescaled;
      const T z1            = T(zstart) - z_rescaled;

      shift[i] = {xstart - xoffset, ystart - yoffset, zstart - zoffset};

      if constexpr (KEREVALMETH == 1) {
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 0, 0), x1, sigma);
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 1, 0), y1, sigma);
        eval_kernel_vec_horner<T, ns>(&kerevals(i, 2, 0), z1, sigma);
      } else {
        eval_kernel_vec<T, ns>(&kerevals(i, 0, 0), x1, es_c, es_beta);
        eval_kernel_vec<T, ns>(&kerevals(i, 1, 0), y1, es_c, es_beta);
        eval_kernel_vec<T, ns>(&kerevals(i, 2, 0), z1, es_c, es_beta);
      }
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      // strength from shared memory
      static constexpr int sizex = ns;            // true span in X
      static constexpr int sizey = ns;            // true span in Y
      static constexpr int sizez = ns;            // true span in Z
      static constexpr int plane = sizex * sizey; // #cells per Z‐slice
      static constexpr int total = plane * sizez; // total #cells

      const auto cnow                     = nupts_sm[i];
      const auto [xstart, ystart, zstart] = shift[i];

      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // decompose idx using `plane`
        const int zz   = idx / plane;
        const int rem1 = idx - zz * plane;
        const int yy   = rem1 / sizex;
        const int xx   = rem1 - yy * sizex;

        // decompose idx using `plane`
        // recover global coords
        const int real_zz = zstart + zz;
        const int real_yy = ystart + yy;
        const int real_xx = xstart + xx;

        // padded indices
        const int iz = real_zz + ns_2;
        const int iy = real_yy + ns_2;
        const int ix = real_xx + ns_2;

        // separable window weights
        const auto kervalue =
            c_kerevals(i, 0, xx) * c_kerevals(i, 1, yy) * c_kerevals(i, 2, zz);
        // accumulate
        local_subgrid(iz, iy, ix) += {cnow * kervalue};
      }
      __syncthreads();
    }
  }
  for (int n = threadIdx.x; n < local_subgrid.size(); n += blockDim.x) {
    const int i = n % (padded_size_x);
    const int j = (n / (padded_size_x)) % (padded_size_y);
    const int k = n / ((padded_size_x) * (padded_size_y));

    int ix = xoffset - ns_2 + i;
    int iy = yoffset - ns_2 + j;
    int iz = zoffset - ns_2 + k;

    if (ix < (nf1 + ns_2) && iy < (nf2 + ns_2) && iz < (nf3 + ns_2)) {
      ix               = ix < 0 ? ix + nf1 : (ix > nf1 - 1 ? ix - nf1 : ix);
      iy               = iy < 0 ? iy + nf2 : (iy > nf2 - 1 ? iy - nf2 : iy);
      iz               = iz < 0 ? iz + nf3 : (iz > nf3 - 1 ? iz - nf3 : iz);
      const int outidx = ix + iy * nf1 + iz * nf1 * nf2;
      atomicAddComplexGlobal<T>(fw + outidx, local_subgrid(k, j, i));
    }
  }
}

/* Kernels for Subprob method */
template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_3d_subprob(
    T *x, T *y, T *z, cuda_complex<T> *c, cuda_complex<T> *fw, int M, int nf1, int nf2,
    int nf3, T sigma, T es_c, T es_beta, int *binstartpts, int *bin_size, int bin_size_x,
    int bin_size_y, int bin_size_z, int *subprob_to_bin, int *subprobstartpts,
    int *numsubprob, int maxsubprobsize, int nbinx, int nbiny, int nbinz, int *idxnupts) {
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

  T ker1[ns];
  T ker2[ns];
  T ker3[ns];

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
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
      eval_kernel_vec_horner<T, ns>(ker3, z1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker3, z1, es_c, es_beta);
    }

    for (int zz = zstart; zz <= zend; zz++) {
      const T kervalue3 = ker3[zz - zstart];
      const int iz      = zz + ns_2;
      if (iz >= (bin_size_z + rounded_ns) || iz < 0) break;
      for (int yy = ystart; yy <= yend; yy++) {
        const T kervalue2 = ker2[yy - ystart];
        const int iy      = yy + ns_2;
        if (iy >= (bin_size_y + rounded_ns) || iy < 0) break;
        for (int xx = xstart; xx <= xend; xx++) {
          const int ix = xx + ns_2;
          if (ix >= (bin_size_x + rounded_ns) || ix < 0) break;
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
template<typename T>
static __global__ void locate_nupts_to_bins_ghost(
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
static __global__ void calc_inverse_of_global_sort_index_ghost(
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

template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_3d_block_gather(
    const T *x, const T *y, const T *z, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma, const int *binstartpts,
    int obin_size_x, int obin_size_y, int obin_size_z, int binsperobin,
    int *subprob_to_bin, const int *subprobstartpts, int maxsubprobsize, int nobinx,
    int nobiny, int nobinz, const int *idxnupts) {
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

  T ker1[ns];
  T ker2[ns];
  T ker3[ns];

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
      eval_kernel_vec_horner<T, ns>(ker1, x1, sigma);
      eval_kernel_vec_horner<T, ns>(ker2, y1, sigma);
      eval_kernel_vec_horner<T, ns>(ker3, z1, sigma);
    } else {
      eval_kernel_vec<T, ns>(ker1, x1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker2, y1, es_c, es_beta);
      eval_kernel_vec<T, ns>(ker3, z1, es_c, es_beta);
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

template<typename T>
void cuspread3d_nuptsdriven_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;

  if (d_plan->opts.gpu_sort) {
    int bin_size_x = d_plan->opts.gpu_binsizex;
    int bin_size_y = d_plan->opts.gpu_binsizey;
    int bin_size_z = d_plan->opts.gpu_binsizez;
    if (bin_size_x < 0 || bin_size_y < 0 || bin_size_z < 0) {
      std::cerr << "[cuspread3d_nuptsdriven_prop] error: invalid binsize "
                   "(binsizex, binsizey, binsizez) = (";
      std::cerr << bin_size_x << "," << bin_size_y << "," << bin_size_z << ")"
                << std::endl;
      throw FINUFFT_ERR_BINSIZE_NOTVALID;
    }

    int numbins[3];
    numbins[0] = (nf1 + bin_size_x - 1) / bin_size_x;
    numbins[1] = (nf2 + bin_size_y - 1) / bin_size_y;
    numbins[2] = (nf3 + bin_size_z - 1) / bin_size_z;

    T *d_kx = d_plan->kxyz[0];
    T *d_ky = d_plan->kxyz[1];
    T *d_kz = d_plan->kxyz[2];

    int *d_binsize     = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx     = d_plan->sortidx;
    int *d_idxnupts    = d_plan->idxnupts;

    checkCudaErrors(cudaMemsetAsync(
             d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));
    calc_bin_size_noghost_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1],
        numbins[2], d_binsize, d_kx, d_ky, d_kz, d_sortidx);
    THROW_IF_CUDA_ERROR

    int n = numbins[0] * numbins[1] * numbins[2];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

    calc_inverse_of_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
        M, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2],
        d_binstartpts, d_sortidx, d_kx, d_ky, d_kz, d_idxnupts, nf1, nf2, nf3);
    THROW_IF_CUDA_ERROR
  } else {
    int *d_idxnupts = d_plan->idxnupts;
    thrust::sequence(thrust::cuda::par.on(stream), d_idxnupts, d_idxnupts + M);
    THROW_IF_CUDA_ERROR
  }
}
template void cuspread3d_nuptsdriven_prop<float>(int nf1, int nf2, int nf3, int M,
                                                 cufinufft_plan_t<float> *d_plan);
template void cuspread3d_nuptsdriven_prop<double>(int nf1, int nf2, int nf3, int M,
                                                  cufinufft_plan_t<double> *d_plan);

template<typename T, int ns>
static void cuspread3d_nuptsdriven(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize) {
  auto &stream = d_plan->stream;

  dim3 threadsPerBlock;
  dim3 blocks;

  T sigma   = d_plan->spopts.upsampfac;
  T es_c    = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta = d_plan->spopts.beta;

  int *d_idxnupts       = d_plan->idxnupts;
  T *d_kx               = d_plan->kxyz[0];
  T *d_ky               = d_plan->kxyz[1];
  T *d_kz               = d_plan->kxyz[2];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x          = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocks.y          = 1;

  if (d_plan->opts.gpu_kerevalmeth == 1) {
    for (int t = 0; t < blksize; t++) {
      spread_3d_nupts_driven<T, 1, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  } else {
    for (int t = 0; t < blksize; t++) {
      spread_3d_nupts_driven<T, 0, ns><<<blocks, threadsPerBlock, 0, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          es_c, es_beta, sigma, d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

template<typename T>
void cuspread3d_blockgather_prop(int nf1, int nf2, int nf3, int M,
                                cufinufft_plan_t<T> *d_plan) {
  auto &stream = d_plan->stream;

  dim3 threadsPerBlock;
  dim3 blocks;

  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
  int o_bin_size_x   = d_plan->opts.gpu_obinsizex;
  int o_bin_size_y   = d_plan->opts.gpu_obinsizey;
  int o_bin_size_z   = d_plan->opts.gpu_obinsizez;

  int numobins[3];
  if (nf1 % o_bin_size_x != 0 || nf2 % o_bin_size_y != 0 || nf3 % o_bin_size_z != 0) {
    std::cerr << "[cuspread3d_blockgather_prop] error:\n";
    std::cerr << "       mod(nf(1|2|3), opts.gpu_obinsize(x|y|z)) != 0" << std::endl;
    std::cerr << "       (nf1, nf2, nf3) = (" << nf1 << ", " << nf2 << ", " << nf3 << ")"
              << std::endl;
    std::cerr << "       (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", "
              << o_bin_size_y << ", " << o_bin_size_z << ")" << std::endl;
    throw FINUFFT_ERR_BINSIZE_NOTVALID;
  }

  numobins[0] = ceil((T)nf1 / o_bin_size_x);
  numobins[1] = ceil((T)nf2 / o_bin_size_y);
  numobins[2] = ceil((T)nf3 / o_bin_size_z);

  int bin_size_x = d_plan->opts.gpu_binsizex;
  int bin_size_y = d_plan->opts.gpu_binsizey;
  int bin_size_z = d_plan->opts.gpu_binsizez;
  if (o_bin_size_x % bin_size_x != 0 || o_bin_size_y % bin_size_y != 0 ||
      o_bin_size_z % bin_size_z != 0) {
    std::cerr << "[cuspread3d_blockgather_prop] error:\n";
    std::cerr << "      mod(ops.gpu_obinsize(x|y|z), opts.gpu_binsize(x|y|z)) != 0"
              << std::endl;
    std::cerr << "      (binsizex, binsizey, binsizez) = (" << bin_size_x << ", "
              << bin_size_y << ", " << bin_size_z << ")" << std::endl;
    std::cerr << "      (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", "
              << o_bin_size_y << ", " << o_bin_size_z << ")" << std::endl;
    throw FINUFFT_ERR_BINSIZE_NOTVALID;
  }

  int binsperobinx, binsperobiny, binsperobinz;
  int numbins[3];
  binsperobinx = o_bin_size_x / bin_size_x + 2;
  binsperobiny = o_bin_size_y / bin_size_y + 2;
  binsperobinz = o_bin_size_z / bin_size_z + 2;
  numbins[0]   = numobins[0] * (binsperobinx);
  numbins[1]   = numobins[1] * (binsperobiny);
  numbins[2]   = numobins[2] * (binsperobinz);

  T *d_kx = d_plan->kxyz[0];
  T *d_ky = d_plan->kxyz[1];
  T *d_kz = d_plan->kxyz[2];

  int *d_binsize         = d_plan->binsize;
  int *d_sortidx         = d_plan->sortidx;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_idxnupts        = NULL;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_subprob_to_bin  = NULL;

  checkCudaErrors(cudaMemsetAsync(
           d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));

  locate_nupts_to_bins_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2],
      binsperobinx, binsperobiny, binsperobinz, d_binsize, d_kx, d_ky, d_kz, d_sortidx,
      nf1, nf2, nf3);
  THROW_IF_CUDA_ERROR

  threadsPerBlock.x = 8;
  threadsPerBlock.y = 8;
  threadsPerBlock.z = 8;

  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  fill_ghost_bins<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobinx, binsperobiny, binsperobinz, numobins[0], numobins[1], numobins[2],
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
  checkCudaErrors(cudaMallocWrapper(&d_idxnupts, totalNUpts * sizeof(int),
                                               stream, d_plan->supports_pools));

  calc_inverse_of_global_sort_index_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2],
      binsperobinx, binsperobiny, binsperobinz, d_binstartpts, d_sortidx, d_kx, d_ky,
      d_kz, d_idxnupts, nf1, nf2, nf3);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_idxnupts);
    throw FINUFFT_ERR_CUDA_FAILURE;
  }

  threadsPerBlock.x = 2;
  threadsPerBlock.y = 2;
  threadsPerBlock.z = 2;

  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  ghost_bin_pts_index<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobinx, binsperobiny, binsperobinz, numobins[0], numobins[1], numobins[2],
      d_binsize, d_idxnupts, d_binstartpts, M);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_idxnupts);
    throw FINUFFT_ERR_CUDA_FAILURE;
  }

  cudaFree(d_plan->idxnupts);
  d_plan->idxnupts = d_idxnupts;

  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  n = numobins[0] * numobins[1] * numobins[2];
  calc_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
      binsperobinx, binsperobiny, binsperobinz, d_binsize, d_numsubprob, maxsubprobsize,
      numobins[0] * numobins[1] * numobins[2]);
  THROW_IF_CUDA_ERROR

  n        = numobins[0] * numobins[1] * numobins[2];
  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream));

  int totalnumsubprob;
  checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n],
                                           sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  checkCudaErrors(
           cudaMallocWrapper(&d_subprob_to_bin, totalnumsubprob * sizeof(int), stream,
                             d_plan->supports_pools));
  map_b_into_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_subprob_to_bin, d_subprobstartpts, d_numsubprob, n);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_subprob_to_bin);
    throw FINUFFT_ERR_CUDA_FAILURE;
  }

  assert(d_subprob_to_bin != NULL);
  cudaFree(d_plan->subprob_to_bin);
  d_plan->subprob_to_bin  = d_subprob_to_bin;
  d_plan->totalnumsubprob = totalnumsubprob;
}
template void cuspread3d_blockgather_prop<float>(int nf1, int nf2, int nf3, int M,
                                                 cufinufft_plan_t<float> *d_plan);
template void cuspread3d_blockgather_prop<double>(int nf1, int nf2, int nf3, int M,
                                                  cufinufft_plan_t<double> *d_plan);

template<typename T, int ns>
static void cuspread3d_blockgather(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                           int blksize) {
  auto &stream = d_plan->stream;

  T es_c             = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta          = d_plan->spopts.beta;
  T sigma            = d_plan->spopts.upsampfac;
  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;

  int obin_size_x = d_plan->opts.gpu_obinsizex;
  int obin_size_y = d_plan->opts.gpu_obinsizey;
  int obin_size_z = d_plan->opts.gpu_obinsizez;
  int bin_size_x  = d_plan->opts.gpu_binsizex;
  int bin_size_y  = d_plan->opts.gpu_binsizey;
  int bin_size_z  = d_plan->opts.gpu_binsizez;
  int numobins[3];
  numobins[0] = ceil((T)nf1 / obin_size_x);
  numobins[1] = ceil((T)nf2 / obin_size_y);
  numobins[2] = ceil((T)nf3 / obin_size_z);

  int binsperobinx, binsperobiny, binsperobinz;
  binsperobinx = obin_size_x / bin_size_x + 2;
  binsperobiny = obin_size_y / bin_size_y + 2;
  binsperobinz = obin_size_z / bin_size_z + 2;

  T *d_kx               = d_plan->kxyz[0];
  T *d_ky               = d_plan->kxyz[1];
  T *d_kz               = d_plan->kxyz[2];
  cuda_complex<T> *d_c  = d_plan->c;
  cuda_complex<T> *d_fw = d_plan->fw;

  int *d_binstartpts     = d_plan->binstartpts;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  size_t sharedplanorysize =
      obin_size_x * obin_size_y * obin_size_z * sizeof(cuda_complex<T>);
  if (sharedplanorysize > 49152) {
    std::cerr << "[cuspread3d_blockgather] error: not enough shared memory" << std::endl;
    throw FINUFFT_ERR_INSUFFICIENT_SHMEM;
  }

  for (int t = 0; t < blksize; t++) {
    if (d_plan->opts.gpu_kerevalmeth == 1) {
      spread_3d_block_gather<T, 1, ns>
          <<<totalnumsubprob, 64, sharedplanorysize, stream>>>(
              d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
              es_c, es_beta, sigma, d_binstartpts, obin_size_x, obin_size_y, obin_size_z,
              binsperobinx * binsperobiny * binsperobinz, d_subprob_to_bin,
              d_subprobstartpts, maxsubprobsize, numobins[0], numobins[1], numobins[2],
              d_idxnupts);
      THROW_IF_CUDA_ERROR
    } else {
      spread_3d_block_gather<T, 0, ns>
          <<<totalnumsubprob, 64, sharedplanorysize, stream>>>(
              d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
              es_c, es_beta, sigma, d_binstartpts, obin_size_x, obin_size_y, obin_size_z,
              binsperobinx * binsperobiny * binsperobinz, d_subprob_to_bin,
              d_subprobstartpts, maxsubprobsize, numobins[0], numobins[1], numobins[2],
              d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

template<typename T>
void cuspread3d_subprob_prop(int nf1, int nf2, int nf3, int M,
                            cufinufft_plan_t<T> *d_plan) {
  const auto stream = d_plan->stream;

  int maxsubprobsize = d_plan->opts.gpu_maxsubprobsize;
  int bin_size_x     = d_plan->opts.gpu_binsizex;
  int bin_size_y     = d_plan->opts.gpu_binsizey;
  int bin_size_z     = d_plan->opts.gpu_binsizez;
  if (bin_size_x < 0 || bin_size_y < 0 || bin_size_z < 0) {
    std::cerr << "error: invalid binsize (binsizex, binsizey, binsizez) = (";
    std::cerr << bin_size_x << "," << bin_size_y << "," << bin_size_z << ")" << std::endl;
    throw FINUFFT_ERR_BINSIZE_NOTVALID;
  }

  int numbins[3];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);
  numbins[2] = ceil((T)nf3 / bin_size_z);

  T *d_kx = d_plan->kxyz[0];
  T *d_ky = d_plan->kxyz[1];
  T *d_kz = d_plan->kxyz[2];

  int *d_binsize         = d_plan->binsize;
  int *d_binstartpts     = d_plan->binstartpts;
  int *d_sortidx         = d_plan->sortidx;
  int *d_numsubprob      = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts        = d_plan->idxnupts;

  int *d_subprob_to_bin = NULL;

  checkCudaErrors(cudaMemsetAsync(
           d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));
  calc_bin_size_noghost_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, nf1, nf2, nf3, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1],
      numbins[2], d_binsize, d_kx, d_ky, d_kz, d_sortidx);
  THROW_IF_CUDA_ERROR

  int n = numbins[0] * numbins[1] * numbins[2];
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);

  calc_inverse_of_global_sort_index_3d<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, bin_size_z, numbins[0], numbins[1], numbins[2],
      d_binstartpts, d_sortidx, d_kx, d_ky, d_kz, d_idxnupts, nf1, nf2, nf3);
  THROW_IF_CUDA_ERROR
  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  calc_subprob_3d_v2<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      d_binsize, d_numsubprob, maxsubprobsize, numbins[0] * numbins[1] * numbins[2]);
  THROW_IF_CUDA_ERROR

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts + 1);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), d_ptr, d_ptr + n, d_result);
  int totalnumsubprob;
  checkCudaErrors(cudaMemsetAsync(d_subprobstartpts, 0, sizeof(int), stream));
  checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n],
                                      sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  checkCudaErrors(cudaMallocWrapper(&d_subprob_to_bin, totalnumsubprob * sizeof(int),
                                        stream, d_plan->supports_pools));

  map_b_into_subprob_3d_v2<<<(numbins[0] * numbins[1] + 1024 - 1) / 1024, 1024, 0,
                             stream>>>(d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
                                       numbins[0] * numbins[1] * numbins[2]);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] Error: %s\n", __func__, cudaGetErrorString(err));
    cudaFree(d_subprob_to_bin);
    throw FINUFFT_ERR_CUDA_FAILURE;
  }

  assert(d_subprob_to_bin != NULL);
  if (d_plan->subprob_to_bin != NULL) cudaFree(d_plan->subprob_to_bin);
  d_plan->subprob_to_bin = d_subprob_to_bin;
  assert(d_plan->subprob_to_bin != nullptr);
  d_plan->totalnumsubprob = totalnumsubprob;
}
template void cuspread3d_subprob_prop<float>(int nf1, int nf2, int nf3, int M,
                                             cufinufft_plan_t<float> *d_plan);
template void cuspread3d_subprob_prop<double>(int nf1, int nf2, int nf3, int M,
                                              cufinufft_plan_t<double> *d_plan);

template<typename T, int ns>
static void cuspread3d_subprob(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
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

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  T sigma                      = d_plan->spopts.upsampfac;
  T es_c                       = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta                    = d_plan->spopts.beta;
  const auto sharedplanorysize = shared_memory_required<T>(
      3, d_plan->spopts.nspread, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);
  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_3d_subprob<T, 1, ns>, 3, *d_plan) != 0) {
      throw FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_3d_subprob<T, 1, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          sigma, es_c, es_beta, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
          bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
          numbins[0], numbins[1], numbins[2], d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_3d_subprob<T, 0, ns>, 3, *d_plan) != 0) {
      throw FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    for (int t = 0; t < blksize; t++) {
      spread_3d_subprob<T, 0, ns><<<totalnumsubprob, 256, sharedplanorysize, stream>>>(
          d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
          sigma, es_c, es_beta, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
          bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob, maxsubprobsize,
          numbins[0], numbins[1], numbins[2], d_idxnupts);
      THROW_IF_CUDA_ERROR
    }
  }
}

template<typename T, int ns>
static void cuspread3d_output_driven(int nf1, int nf2, int nf3, int M,
                              cufinufft_plan_t<T> *d_plan, int blksize) {
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

  int totalnumsubprob   = d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  const auto np = d_plan->opts.gpu_np;

  T sigma                      = d_plan->spopts.upsampfac;
  T es_c                       = 4.0 / T(d_plan->spopts.nspread * d_plan->spopts.nspread);
  T es_beta                    = d_plan->spopts.beta;
  const auto sharedplanorysize = shared_memory_required<T>(
      3, ns, d_plan->opts.gpu_binsizex, d_plan->opts.gpu_binsizey,
      d_plan->opts.gpu_binsizez, d_plan->opts.gpu_np);
  if (d_plan->opts.gpu_kerevalmeth) {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_3d_output_driven<T, 1, ns>, 3, *d_plan) !=
            0) {
      throw FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    cudaFuncSetSharedMemConfig(spread_3d_output_driven<T, 1, ns>,
                               cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      spread_3d_output_driven<T, 1, ns>
          <<<totalnumsubprob, std::min(256, std::max(ns * ns * ns, np)),
             sharedplanorysize, stream>>>(
              d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
              sigma, es_c, es_beta, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
              bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
              maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts, np);
      THROW_IF_CUDA_ERROR
    }
  } else {
    if (const auto finufft_err =
            cufinufft_set_shared_memory(spread_3d_output_driven<T, 0, ns>, 3, *d_plan) !=
            0) {
      throw FINUFFT_ERR_INSUFFICIENT_SHMEM;
    }
    cudaFuncSetSharedMemConfig(spread_3d_output_driven<T, 0, ns>,
                               cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      spread_3d_output_driven<T, 0, ns>
          <<<totalnumsubprob, std::min(256, std::max(ns * ns * ns, np)),
             sharedplanorysize, stream>>>(
              d_kx, d_ky, d_kz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, nf1, nf2, nf3,
              sigma, es_c, es_beta, d_binstartpts, d_binsize, bin_size_x, bin_size_y,
              bin_size_z, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
              maxsubprobsize, numbins[0], numbins[1], numbins[2], d_idxnupts,
              d_plan->opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  }
}

// Functor to handle function selection (nuptsdriven, subprob, blockgather)
struct Spread3DDispatcher {
  template<int ns, typename T>
  void operator()(int nf1, int nf2, int nf3, int M, cufinufft_plan_t<T> *d_plan,
                 int blksize) const {
    switch (d_plan->opts.gpu_method) {
    case 1:
      cuspread3d_nuptsdriven<T, ns>(nf1, nf2, nf3, M, d_plan, blksize);
    case 2:
      cuspread3d_subprob<T, ns>(nf1, nf2, nf3, M, d_plan, blksize);
    case 3:
      cuspread3d_output_driven<T, ns>(nf1, nf2, nf3, M, d_plan, blksize);
    case 4:
      cuspread3d_blockgather<T, ns>(nf1, nf2, nf3, M, d_plan, blksize);
    default:
      std::cerr << "[cuspread3d] error: invalid method " +
                       std::to_string(d_plan->opts.gpu_method) +
                       ", should be 1, 2, 3 or 4\n";
      throw FINUFFT_ERR_METHOD_NOTVALID;
    }
  }
};

// Updated cuspread3d using generic dispatch
template<typename T> void cuspread3d(cufinufft_plan_t<T> *d_plan, int blksize) {
  /*
    A wrapper for different spreading methods.

    Methods available:
        (1) Non-uniform points driven
        (2) Subproblem
        (4) Block gather

    Melody Shih 07/25/19

    Now the function is updated to dispatch based on ns. This is to avoid alloca which
    it seems slower according to the MRI community.
    Marco Barbone 01/30/25
  */
  launch_dispatch_ns<Spread3DDispatcher, T>(
      Spread3DDispatcher(), d_plan->spopts.nspread, d_plan->nf123[0], d_plan->nf123[1], d_plan->nf123[2],
      d_plan->M, d_plan, blksize);
}
template void cuspread3d<float>(cufinufft_plan_t<float> *d_plan, int blksize);
template void cuspread3d<double>(cufinufft_plan_t<double> *d_plan, int blksize);

} // namespace spreadinterp
} // namespace cufinufft

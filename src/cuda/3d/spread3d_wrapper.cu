#include <cassert>
#include <iostream>

#include <cuComplex.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/contrib/helper_math.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cufinufft/common.h>
#include <cufinufft/common_kernels.hpp>
#include <cufinufft/intrinsics.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

using namespace cufinufft::common;
using namespace cufinufft::utils;

namespace cufinufft {
namespace spreadinterp {

using cuda::std::dextents;
using cuda::std::dynamic_extent;
using cuda::std::extents;
using cuda::std::mdspan;
using cuda::std::span;

static __host__ __device__ int calc_global_index(
    int xidx, int yidx, int zidx, int onx, int ony, int onz, int bnx, int bny, int bnz) {
  int oix, oiy, oiz;
  oix = xidx / bnx;
  oiy = yidx / bny;
  oiz = zidx / bnz;
  return (oix + oiy * onx + oiz * ony * onx) * (bnx * bny * bnz) +
         (xidx % bnx + yidx % bny * bnx + zidx % bnz * bny * bnx);
}

static __global__ void calc_subprob_3d_v1(
    int binsperobinx, int binsperobiny, int binsperobinz, const int *bin_size,
    int *num_subprob, int maxsubprobsize, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    int numnupts    = 0;
    int binsperobin = binsperobinx * binsperobiny * binsperobinz;
    for (int b = 0; b < binsperobin; b++) {
      numnupts += bin_size[binsperobin * i + b];
    }
    num_subprob[i] = ceil(numnupts / (float)maxsubprobsize);
  }
}

static __global__ void map_b_into_subprob_3d_v1(int *d_subprob_to_obin,
                                                const int *d_subprobstartpts,
                                                const int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d_numsubprob[i]; j++) {
      d_subprob_to_obin[d_subprobstartpts[i] + j] = i;
    }
  }
}

static __global__ void fill_ghost_bins(int binsperobinx, int binsperobiny,
                                       int binsperobinz, int nobinx, int nobiny,
                                       int nobinz, int *binsize) {
  int binx = threadIdx.x + blockIdx.x * blockDim.x;
  int biny = threadIdx.y + blockIdx.y * blockDim.y;
  int binz = threadIdx.z + blockIdx.z * blockDim.z;

  int nbinx = nobinx * binsperobinx;
  int nbiny = nobiny * binsperobiny;
  int nbinz = nobinz * binsperobinz;

  if (binx < nbinx && biny < nbiny && binz < nbinz) {
    int binidx = calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx,
                                   binsperobiny, binsperobinz);
    int i, j, k;
    i = binx;
    j = biny;
    k = binz;
    if (binx % binsperobinx == 0) {
      i = binx - 2;
      i = i < 0 ? i + nbinx : i;
    }
    if (binx % binsperobinx == binsperobinx - 1) {
      i = binx + 2;
      i = (i >= nbinx) ? i - nbinx : i;
    }
    if (biny % binsperobiny == 0) {
      j = biny - 2;
      j = j < 0 ? j + nbiny : j;
    }
    if (biny % binsperobiny == binsperobiny - 1) {
      j = biny + 2;
      j = (j >= nbiny) ? j - nbiny : j;
    }
    if (binz % binsperobinz == 0) {
      k = binz - 2;
      k = k < 0 ? k + nbinz : k;
    }
    if (binz % binsperobinz == binsperobinz - 1) {
      k = binz + 2;
      k = (k >= nbinz) ? k - nbinz : k;
    }
    int idxtoupdate = calc_global_index(i, j, k, nobinx, nobiny, nobinz, binsperobinx,
                                        binsperobiny, binsperobinz);
    if (idxtoupdate != binidx) {
      binsize[binidx] = binsize[idxtoupdate];
    }
  }
}

static __global__ void ghost_bin_pts_index(
    int binsperobinx, int binsperobiny, int binsperobinz, int nobinx, int nobiny,
    int nobinz, const int *binsize, int *index, const int *binstartpts, int M) {
  int binx  = threadIdx.x + blockIdx.x * blockDim.x;
  int biny  = threadIdx.y + blockIdx.y * blockDim.y;
  int binz  = threadIdx.z + blockIdx.z * blockDim.z;
  int nbinx = nobinx * binsperobinx;
  int nbiny = nobiny * binsperobiny;
  int nbinz = nobinz * binsperobinz;

  int i, j, k;
  int w = 0;
  int box[3];
  if (binx < nbinx && biny < nbiny && binz < nbinz) {
    box[0] = box[1] = box[2] = 0;
    i                        = binx;
    j                        = biny;
    k                        = binz;
    int binidx = calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx,
                                   binsperobiny, binsperobinz);
    if (binx % binsperobinx == 0) {
      i      = binx - 2;
      box[0] = (i < 0);
      i      = i < 0 ? i + nbinx : i;
      w      = 1;
    }
    if (binx % binsperobinx == binsperobinx - 1) {
      i      = binx + 2;
      box[0] = (i > nbinx) * 2;
      i      = (i > nbinx) ? i - nbinx : i;
      w      = 1;
    }
    if (biny % binsperobiny == 0) {
      j      = biny - 2;
      box[1] = (j < 0);
      j      = j < 0 ? j + nbiny : j;
      w      = 1;
    }
    if (biny % binsperobiny == binsperobiny - 1) {
      j      = biny + 2;
      box[1] = (j > nbiny) * 2;
      j      = (j > nbiny) ? j - nbiny : j;
      w      = 1;
    }
    if (binz % binsperobinz == 0) {
      k      = binz - 2;
      box[2] = (k < 0);
      k      = k < 0 ? k + nbinz : k;
      w      = 1;
    }
    if (binz % binsperobinz == binsperobinz - 1) {
      k      = binz + 2;
      box[2] = (k > nbinz) * 2;
      k      = (k > nbinz) ? k - nbinz : k;
      w      = 1;
    }
    int corbinidx = calc_global_index(i, j, k, nobinx, nobiny, nobinz, binsperobinx,
                                      binsperobiny, binsperobinz);
    if (w == 1) {
      for (int n = 0; n < binsize[binidx]; n++) {
        index[binstartpts[binidx] + n] =
            M * (box[0] + box[1] * 3 + box[2] * 9) + index[binstartpts[corbinidx] + n];
      }
    }
  }
}

/* Kernels for Subprob method */
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

    binidx     = calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx,
                                   binsperobiny, binsperobinz);
    oldidx     = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename T>
static __global__ void calc_inverse_of_global_sort_index_ghost(
    int M, int bin_size_x, int bin_size_y, int bin_size_z, int nobinx, int nobiny,
    int nobinz, int binsperobinx, int binsperobiny, int binsperobinz,
    const int *bin_startpts, const int *sortidx, const T *x, const T *y, const T *z,
    int *index, int nf1, int nf2, int nf3) {
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

    binidx = calc_global_index(binx, biny, binz, nobinx, nobiny, nobinz, binsperobinx,
                               binsperobiny, binsperobinz);

    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

template<typename T, int KEREVALMETH, int ns>
static __global__ void spread_3d_block_gather(
    const T *x, const T *y, const T *z, const cuda_complex<T> *c, cuda_complex<T> *fw,
    int M, int nf1, int nf2, int nf3, T es_c, T es_beta, T sigma, const int *binstartpts,
    int obin_size_x, int obin_size_y, int obin_size_z, int binsperobin,
    const int *subprob_to_bin, const int *subprobstartpts, int maxsubprobsize, int nobinx,
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
static void cuspread3d_blockgather_prop(cufinufft_plan_t<T> &d_plan) {
  auto &stream = d_plan.stream;
  int M        = d_plan.M;
  int nf1      = d_plan.nf123[0];
  int nf2      = d_plan.nf123[1];
  int nf3      = d_plan.nf123[2];

  dim3 threadsPerBlock;
  dim3 blocks;

  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;
  int o_bin_size_x   = d_plan.opts.gpu_obinsizex;
  int o_bin_size_y   = d_plan.opts.gpu_obinsizey;
  int o_bin_size_z   = d_plan.opts.gpu_obinsizez;

  int numobins[3];
  if (nf1 % o_bin_size_x != 0 || nf2 % o_bin_size_y != 0 || nf3 % o_bin_size_z != 0) {
    std::cerr << "[cuspread3d_blockgather_prop] error:\n";
    std::cerr << "       mod(nf(1|2|3), opts.gpu_obinsize(x|y|z)) != 0" << std::endl;
    std::cerr << "       (nf1, nf2, nf3) = (" << nf1 << ", " << nf2 << ", " << nf3 << ")"
              << std::endl;
    std::cerr << "       (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", "
              << o_bin_size_y << ", " << o_bin_size_z << ")" << std::endl;
    throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
  }

  numobins[0] = ceil((T)nf1 / o_bin_size_x);
  numobins[1] = ceil((T)nf2 / o_bin_size_y);
  numobins[2] = ceil((T)nf3 / o_bin_size_z);

  int bin_size_x = d_plan.opts.gpu_binsizex;
  int bin_size_y = d_plan.opts.gpu_binsizey;
  int bin_size_z = d_plan.opts.gpu_binsizez;
  if (o_bin_size_x % bin_size_x != 0 || o_bin_size_y % bin_size_y != 0 ||
      o_bin_size_z % bin_size_z != 0) {
    std::cerr << "[cuspread3d_blockgather_prop] error:\n";
    std::cerr << "      mod(ops.gpu_obinsize(x|y|z), opts.gpu_binsize(x|y|z)) != 0"
              << std::endl;
    std::cerr << "      (binsizex, binsizey, binsizez) = (" << bin_size_x << ", "
              << bin_size_y << ", " << bin_size_z << ")" << std::endl;
    std::cerr << "      (obinsizex, obinsizey, obinsizez) = (" << o_bin_size_x << ", "
              << o_bin_size_y << ", " << o_bin_size_z << ")" << std::endl;
    throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
  }

  int binsperobinx, binsperobiny, binsperobinz;
  int numbins[3];
  binsperobinx = o_bin_size_x / bin_size_x + 2;
  binsperobiny = o_bin_size_y / bin_size_y + 2;
  binsperobinz = o_bin_size_z / bin_size_z + 2;
  numbins[0]   = numobins[0] * (binsperobinx);
  numbins[1]   = numobins[1] * (binsperobiny);
  numbins[2]   = numobins[2] * (binsperobinz);

  const T *d_kx = d_plan.kxyz[0];
  const T *d_ky = d_plan.kxyz[1];
  const T *d_kz = d_plan.kxyz[2];

  int *d_binsize         = dethrust(d_plan.binsize);
  int *d_sortidx         = dethrust(d_plan.sortidx);
  int *d_binstartpts     = dethrust(d_plan.binstartpts);
  int *d_numsubprob      = dethrust(d_plan.numsubprob);
  int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);

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
  gpu_array<int> d_idxnupts(totalNUpts, d_plan.alloc);

  calc_inverse_of_global_sort_index_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size_x, bin_size_y, bin_size_z, numobins[0], numobins[1], numobins[2],
      binsperobinx, binsperobiny, binsperobinz, d_binstartpts, d_sortidx, d_kx, d_ky,
      d_kz, dethrust(d_idxnupts), nf1, nf2, nf3);

  threadsPerBlock.x = 2;
  threadsPerBlock.y = 2;
  threadsPerBlock.z = 2;

  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  ghost_bin_pts_index<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobinx, binsperobiny, binsperobinz, numobins[0], numobins[1], numobins[2],
      d_binsize, dethrust(d_idxnupts), d_binstartpts, M);

  d_plan.idxnupts.clear();
  d_plan.idxnupts.swap(d_idxnupts);

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
  checkCudaErrors(cudaMemcpyAsync(&totalnumsubprob, &d_subprobstartpts[n], sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  gpu_array<int> d_subprob_to_bin(totalnumsubprob, d_plan.alloc);
  map_b_into_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
      dethrust(d_subprob_to_bin), d_subprobstartpts, d_numsubprob, n);

  d_plan.subprob_to_bin.clear();
  d_plan.subprob_to_bin.swap(d_subprob_to_bin);
  d_plan.totalnumsubprob = totalnumsubprob;
}

template<typename T, int ns>
static void cuspread3d_blockgather(int nf1, int nf2, int nf3, int M,
                                   const cufinufft_plan_t<T> &d_plan, int blksize) {
  auto &stream = d_plan.stream;

  T es_c             = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta          = d_plan.spopts.beta;
  T sigma            = d_plan.spopts.upsampfac;
  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;

  int obin_size_x = d_plan.opts.gpu_obinsizex;
  int obin_size_y = d_plan.opts.gpu_obinsizey;
  int obin_size_z = d_plan.opts.gpu_obinsizez;
  int bin_size_x  = d_plan.opts.gpu_binsizex;
  int bin_size_y  = d_plan.opts.gpu_binsizey;
  int bin_size_z  = d_plan.opts.gpu_binsizez;
  int numobins[3];
  numobins[0] = ceil((T)nf1 / obin_size_x);
  numobins[1] = ceil((T)nf2 / obin_size_y);
  numobins[2] = ceil((T)nf3 / obin_size_z);

  int binsperobinx, binsperobiny, binsperobinz;
  binsperobinx = obin_size_x / bin_size_x + 2;
  binsperobiny = obin_size_y / bin_size_y + 2;
  binsperobinz = obin_size_z / bin_size_z + 2;

  const T *d_kx              = d_plan.kxyz[0];
  const T *d_ky              = d_plan.kxyz[1];
  const T *d_kz              = d_plan.kxyz[2];
  const cuda_complex<T> *d_c = d_plan.c;
  cuda_complex<T> *d_fw      = d_plan.fw;

  const int *d_binstartpts     = dethrust(d_plan.binstartpts);
  const int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);
  const int *d_idxnupts        = dethrust(d_plan.idxnupts);

  int totalnumsubprob         = d_plan.totalnumsubprob;
  const int *d_subprob_to_bin = dethrust(d_plan.subprob_to_bin);

  size_t sharedplanorysize =
      obin_size_x * obin_size_y * obin_size_z * sizeof(cuda_complex<T>);
  if (sharedplanorysize > 49152) {
    std::cerr << "[cuspread3d_blockgather] error: not enough shared memory" << std::endl;
    throw int(FINUFFT_ERR_INSUFFICIENT_SHMEM);
  }

  for (int t = 0; t < blksize; t++) {
    if (d_plan.opts.gpu_kerevalmeth == 1) {
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
#if 0
template<typename T, int ns>
static void cuspread3d_output_driven(int nf1, int nf2, int nf3, int M,
                                     const cufinufft_plan_t<T> &d_plan, int blksize) {
  auto &stream = d_plan.stream;

  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;

  // assume that bin_size_x > ns/2;
  int bin_size_x = d_plan.opts.gpu_binsizex;
  int bin_size_y = d_plan.opts.gpu_binsizey;
  int bin_size_z = d_plan.opts.gpu_binsizez;
  int numbins[3];
  numbins[0] = ceil((T)nf1 / bin_size_x);
  numbins[1] = ceil((T)nf2 / bin_size_y);
  numbins[2] = ceil((T)nf3 / bin_size_z);

  const T *d_kx              = d_plan.kxyz[0];
  const T *d_ky              = d_plan.kxyz[1];
  const T *d_kz              = d_plan.kxyz[2];
  const cuda_complex<T> *d_c = d_plan.c;
  cuda_complex<T> *d_fw      = d_plan.fw;

  const int *d_binsize         = dethrust(d_plan.binsize);
  const int *d_binstartpts     = dethrust(d_plan.binstartpts);
  const int *d_numsubprob      = dethrust(d_plan.numsubprob);
  const int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);
  const int *d_idxnupts        = dethrust(d_plan.idxnupts);

  int totalnumsubprob         = d_plan.totalnumsubprob;
  const int *d_subprob_to_bin = dethrust(d_plan.subprob_to_bin);

  const auto np = d_plan.opts.gpu_np;

  T sigma   = d_plan.spopts.upsampfac;
  T es_c    = 4.0 / T(d_plan.spopts.nspread * d_plan.spopts.nspread);
  T es_beta = d_plan.spopts.beta;
  const auto sharedplanorysize =
      shared_memory_required<T>(3, ns, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
                                d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);
  if (d_plan.opts.gpu_kerevalmeth) {
    cufinufft_set_shared_memory(spread_output_driven<T, 1, 3, ns>, 3, d_plan);
    cudaFuncSetSharedMemConfig(spread_output_driven<T, 1, 3, ns>,
                               cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      spread_output_driven<T, 1, 3, ns>
          <<<totalnumsubprob, std::min(256, std::max(ns * ns * ns, np)),
             sharedplanorysize, stream>>>(
              d_plan.kxyz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, d_plan.nf123,
              sigma, es_c, es_beta, d_binstartpts, d_binsize, {bin_size_x, bin_size_y,
              bin_size_z}, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
              maxsubprobsize, {numbins[0], numbins[1], numbins[2]}, d_idxnupts, np);
      THROW_IF_CUDA_ERROR
    }
  } else {
    cufinufft_set_shared_memory(spread_output_driven<T, 0, 3, ns>, 3, d_plan);
    cudaFuncSetSharedMemConfig(spread_output_driven<T, 0, 3, ns>,
                               cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      spread_output_driven<T, 0, 3, ns>
          <<<totalnumsubprob, std::min(256, std::max(ns * ns * ns, np)),
             sharedplanorysize, stream>>>(
              d_plan.kxyz, d_c + t * M, d_fw + t * nf1 * nf2 * nf3, M, d_plan.nf123,
              sigma, es_c, es_beta, d_binstartpts, d_binsize, {bin_size_x, bin_size_y,
              bin_size_z}, d_subprob_to_bin, d_subprobstartpts, d_numsubprob,
              maxsubprobsize, {numbins[0], numbins[1], numbins[2]}, d_idxnupts, np);
      THROW_IF_CUDA_ERROR
    }
  }
}
#endif
// Functor to handle function selection (nuptsdriven, subprob, blockgather)
struct Spread3DDispatcher {
  template<int ns, typename T>
  void operator()(int nf1, int nf2, int nf3, int M, const cufinufft_plan_t<T> &d_plan,
                  int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuspread_nupts_driven<T, 3, ns>(d_plan, blksize);
    case 2:
      return cuspread_subprob<T, 3, ns>(d_plan, blksize);
    case 3:
      return cuspread_output_driven<T, 3, ns>(d_plan, blksize);
    case 4:
      return cuspread3d_blockgather<T, ns>(nf1, nf2, nf3, M, d_plan, blksize);
    default:
      std::cerr << "[cuspread3d] error: invalid method " +
                       std::to_string(d_plan.opts.gpu_method) +
                       ", should be 1, 2, 3 or 4\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Updated cuspread3d using generic dispatch
template<typename T> void cuspread3d(const cufinufft_plan_t<T> &d_plan, int blksize) {
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
  launch_dispatch_ns<Spread3DDispatcher, T>(Spread3DDispatcher(), d_plan.spopts.nspread,
                                            d_plan.nf123[0], d_plan.nf123[1],
                                            d_plan.nf123[2], d_plan.M, d_plan, blksize);
}
template void cuspread3d<float>(const cufinufft_plan_t<float> &d_plan, int blksize);
template void cuspread3d<double>(const cufinufft_plan_t<double> &d_plan, int blksize);

template<typename T> void cuspread3d_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_method == 1) cuspread_nuptsdriven_prop<T, 3>(d_plan);
  if (d_plan.opts.gpu_method == 2) cuspread_subprob_prop<T, 3>(d_plan);
  if (d_plan.opts.gpu_method == 3) cuspread_subprob_prop<T, 3>(d_plan);
  if (d_plan.opts.gpu_method == 4) cuspread3d_blockgather_prop<T>(d_plan);
}
template void cuspread3d_prop(cufinufft_plan_t<float> &d_plan);
template void cuspread3d_prop(cufinufft_plan_t<double> &d_plan);

} // namespace spreadinterp
} // namespace cufinufft

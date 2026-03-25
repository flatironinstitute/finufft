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

static __host__ __device__ int calc_global_index(
    int xidx, int yidx, int zidx, int onx, int ony, int onz, int bnx, int bny, int bnz) {
  int oix, oiy, oiz;
  oix = xidx / bnx;
  oiy = yidx / bny;
  oiz = zidx / bnz;
  return (oix + oiy * onx + oiz * ony * onx) * (bnx * bny * bnz) +
         (xidx % bnx + yidx % bny * bnx + zidx % bnz * bny * bnx);
}
static __host__ __device__ int calc_global_index(
    cuda::std::array<int,3> idx, cuda::std::array<int,3> on, cuda::std::array<int,3> bn) {
  cuda::std::array<int,3> oi{idx[0]/bn[0], idx[1]/bn[1], idx[2]/bn[2]};
  return (oi[0] + oi[1] * on[0] + oi[2] * on[1] * on[0]) * (bn[0] * bn[1] * bn[2]) +
         (idx[0] % bn[0] + idx[1] % bn[1] * bn[0] + idx[2] % bn[2] * bn[1] * bn[0]);
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

static __global__ void fill_ghost_bins(cuda::std::array<int,3> binsperobin,
                                       cuda::std::array<int,3> nobin, int *binsize) {
  constexpr int ndim=3;

  cuda::std::array<int,3> bin = {int(threadIdx.x + blockIdx.x * blockDim.x),
                                 int(threadIdx.y + blockIdx.y * blockDim.y),
                                 int(threadIdx.z + blockIdx.z * blockDim.z)};

  cuda::std::array<int,3> nbin;
  for (int idim=0; idim<ndim; ++idim)
    nbin[idim] = nobin[idim] * binsperobin[idim];

  if (bin[0] < nbin[0] && bin[1] < nbin[1] && bin[2] < nbin[2]) {
    int binidx = calc_global_index(bin, nobin, binsperobin);
    cuda::std::array<int,3> ijk;
    for (int idim=0; idim<ndim; ++idim) {
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
    int M, cuda::std::array<int,3> binsize, cuda::std::array<int,3> nobin,
    cuda::std::array<int,3> binsperobin, int *bin_size,
    cuda::std::array<const T *,3> xyz, int *sortidx, cuda::std::array<int,3> nf123) {
  int binidx;
  cuda::std::array<int,3> bin;
  int oldidx;
  cuda::std::array<T,3>  rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    rescaled[0] = fold_rescale(xyz[0][i], nf123[0]);
    rescaled[1] = fold_rescale(xyz[1][i], nf123[1]);
    rescaled[2] = fold_rescale(xyz[2][i], nf123[2]);
    bin[0]       = floor(rescaled[0] / binsize[0]);
    bin[1]       = floor(rescaled[1] / binsize[1]);
    bin[2]       = floor(rescaled[2] / binsize[2]);
    bin[0] = bin[0] / (binsperobin[0] - 2) * binsperobin[0] + (bin[0] % (binsperobin[0] - 2) + 1);
    bin[1] = bin[1] / (binsperobin[1] - 2) * binsperobin[1] + (bin[1] % (binsperobin[1] - 2) + 1);
    bin[2] = bin[2] / (binsperobin[2] - 2) * binsperobin[2] + (bin[2] % (binsperobin[2] - 2) + 1);

    binidx     = calc_global_index(bin, nobin, binsperobin);
    oldidx     = atomicAdd(&bin_size[binidx], 1);
    sortidx[i] = oldidx;
  }
}

template<typename T>
static __global__ void calc_inverse_of_global_sort_index_ghost(
    int M, cuda::std::array<int,3> binsize, cuda::std::array<int,3> nobin,
    cuda::std::array<int,3> binsperobin,
    const int *bin_startpts, const int *sortidx, cuda::std::array<const T *,3> xyz,
    int *index, cuda::std::array<int,3> nf123) {
  cuda::std::array<int,3> bin;
  cuda::std::array<T,3>  rescaled;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M;
       i += gridDim.x * blockDim.x) {
    rescaled[0] = fold_rescale(xyz[0][i], nf123[0]);
    rescaled[1] = fold_rescale(xyz[1][i], nf123[1]);
    rescaled[2] = fold_rescale(xyz[2][i], nf123[2]);
    bin[0]       = floor(rescaled[0] / binsize[0]);
    bin[1]       = floor(rescaled[1] / binsize[1]);
    bin[2]       = floor(rescaled[2] / binsize[2]);
    bin[0] = bin[0] / (binsperobin[0] - 2) * binsperobin[0] + (bin[0] % (binsperobin[0] - 2) + 1);
    bin[1] = bin[1] / (binsperobin[1] - 2) * binsperobin[1] + (bin[1] % (binsperobin[1] - 2) + 1);
    bin[2] = bin[2] / (binsperobin[2] - 2) * binsperobin[2] + (bin[2] % (binsperobin[2] - 2) + 1);

    int binidx = calc_global_index(bin, nobin, binsperobin);

    index[bin_startpts[binidx] + sortidx[i]] = i;
  }
}

template<typename T, int KEREVALMETH, int ndim, int ns>
static __global__ void spread_3d_block_gather(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw) {
  static_assert (ndim==3, "unsupported dimensionality");

  T es_c             = 4.0 / T(p.spopts.nspread * p.spopts.nspread);
  T es_beta          = p.spopts.beta;
  T sigma            = p.spopts.upsampfac;
  int maxsubprobsize = p.opts.gpu_maxsubprobsize;

  cuda::std::array<int,3> obin_size { p.opts.gpu_obinsizex, p.opts.gpu_obinsizey, p.opts.gpu_obinsizez };
  cuda::std::array<int,3> bin_size { p.opts.gpu_binsizex, p.opts.gpu_binsizey, p.opts.gpu_binsizez };
  cuda::std::array<int,ndim> nobin;
  for (size_t idim=0; idim<ndim; ++idim)
    nobin[idim] = ceil((T)p.nf123[idim] / obin_size[idim]);

  cuda::std::array<int,ndim> binsperobin;
  int binsperobin_tot = 1;
  for (size_t idim=0; idim<ndim; ++idim) {
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

  cuda::std::array<int,ndim> offset;
  offset[0] = (obidx % nobin[0]) * obin_size[0];
  offset[1] = (obidx / nobin[0]) % nobin[1] * obin_size[1];
  offset[2] = (obidx / (nobin[0] * nobin[1])) * obin_size[2];

  const int N = obin_size[0] * obin_size[1] * obin_size[2];

  T ker1[ns];
  T ker2[ns];
  T ker3[ns];

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

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
    const int ii          = nidx % p.M;
    const auto x_rescaled = fold_rescale(p.xyz[0][ii], p.nf123[0]) + box[0] * p.nf123[0];
    const auto y_rescaled = fold_rescale(p.xyz[1][ii], p.nf123[1]) + box[1] * p.nf123[1];
    const auto z_rescaled = fold_rescale(p.xyz[2][ii], p.nf123[2]) + box[2] * p.nf123[2];
    const auto cnow       = c[ii];
    auto [xstart, xend]   = interval(ns, x_rescaled);
    auto [ystart, yend]   = interval(ns, y_rescaled);
    auto [zstart, zend]   = interval(ns, z_rescaled);

    const T x1 = T(xstart) - x_rescaled;
    const T y1 = T(ystart) - y_rescaled;
    const T z1 = T(zstart) - z_rescaled;

    xstart -= offset[0];
    ystart -= offset[1];
    zstart -= offset[2];

    xend -= offset[0];
    yend -= offset[1];
    zend -= offset[2];

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
    const auto xendnew   = xend >= obin_size[0] ? obin_size[0] - 1 : xend;
    const auto yendnew   = yend >= obin_size[1] ? obin_size[1] - 1 : yend;
    const auto zendnew   = zend >= obin_size[2] ? obin_size[2] - 1 : zend;

    for (int zz = zstartnew; zz <= zendnew; zz++) {
      const T kervalue3 = ker3[zz - zstart];
      for (int yy = ystartnew; yy <= yendnew; yy++) {
        const T kervalue2 = ker2[yy - ystart];
        for (int xx = xstartnew; xx <= xendnew; xx++) {
          const auto outidx = xx + yy * obin_size[0] + zz * obin_size[1] * obin_size[0];
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
    int i = n % obin_size[0];
    int j = (n / obin_size[0]) % obin_size[1];
    int k = n / (obin_size[0] * obin_size[1]);

    const auto ix     = offset[0] + i;
    const auto iy     = offset[1] + j;
    const auto iz     = offset[2] + k;
    const auto outidx = ix + iy * p.nf123[0] + iz * p.nf123[0] * p.nf123[1];
    atomicAdd(&fw[outidx].x, fwshared[n].x);
    atomicAdd(&fw[outidx].y, fwshared[n].y);
  }
}

template<typename T, int ndim>
static void cuspread3d_blockgather_prop(cufinufft_plan_t<T> &d_plan) {
  static_assert (ndim==3, "unsupported dimensionality");

  auto &stream = d_plan.stream;
  int M        = d_plan.M;

  int maxsubprobsize = d_plan.opts.gpu_maxsubprobsize;
  cuda::std::array<int,3> o_bin_size;
  o_bin_size[0] = d_plan.opts.gpu_obinsizex;
  o_bin_size[1] = d_plan.opts.gpu_obinsizey;
  o_bin_size[2] = d_plan.opts.gpu_obinsizez;

  if (d_plan.nf123[0] % o_bin_size[0] != 0 || d_plan.nf123[1] % o_bin_size[1] != 0 || d_plan.nf123[2] % o_bin_size[2] != 0) {
    std::cerr << "[cuspread3d_blockgather_prop] error:\n";
    std::cerr << "       mod(nf(1|2|3), opts.gpu_obinsize(x|y|z)) != 0" << std::endl;
    std::cerr << "       (nf1, nf2, nf3) = (" << d_plan.nf123[0] << ", " << d_plan.nf123[1] << ", " << d_plan.nf123[2] << ")"
              << std::endl;
    std::cerr << "       (obinsizex, obinsizey, obinsizez) = (" << o_bin_size[0] << ", "
              << o_bin_size[1] << ", " << o_bin_size[2] << ")" << std::endl;
    throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
  }

  cuda::std::array<int,3> numobins;
  numobins[0] = ceil((T)d_plan.nf123[0] / o_bin_size[0]);
  numobins[1] = ceil((T)d_plan.nf123[1] / o_bin_size[1]);
  numobins[2] = ceil((T)d_plan.nf123[2] / o_bin_size[2]);

  cuda::std::array<int,3> bin_size = {d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey, d_plan.opts.gpu_binsizez};
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

  cuda::std::array<int,3> binsperobin;
  cuda::std::array<int,3> numbins;
  binsperobin[0] = o_bin_size[0] / bin_size[0] + 2;
  binsperobin[1] = o_bin_size[1] / bin_size[1] + 2;
  binsperobin[2] = o_bin_size[2] / bin_size[2] + 2;
  numbins[0]   = numobins[0] * binsperobin[0];
  numbins[1]   = numobins[1] * binsperobin[1];
  numbins[2]   = numobins[2] * binsperobin[2];

  int *d_binsize         = dethrust(d_plan.binsize);
  int *d_sortidx         = dethrust(d_plan.sortidx);
  int *d_binstartpts     = dethrust(d_plan.binstartpts);
  int *d_numsubprob      = dethrust(d_plan.numsubprob);
  int *d_subprobstartpts = dethrust(d_plan.subprobstartpts);

  checkCudaErrors(cudaMemsetAsync(
      d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));

  locate_nupts_to_bins_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size, numobins,
      binsperobin, d_binsize, d_plan.kxyz, d_sortidx,
      d_plan.nf123);
  THROW_IF_CUDA_ERROR

  dim3 threadsPerBlock = {8,8,8};

  dim3 blocks;
  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  fill_ghost_bins<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobin, numobins, d_binsize);
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
      M, bin_size, numobins, binsperobin, d_binstartpts, d_sortidx, d_plan.kxyz,
      dethrust(d_idxnupts), d_plan.nf123);

  threadsPerBlock = { 2,2,2 };

  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  ghost_bin_pts_index<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobin[0], binsperobin[1], binsperobin[2], numobins[0], numobins[1], numobins[2],
      d_binsize, dethrust(d_idxnupts), d_binstartpts, M);

  d_plan.idxnupts.clear();
  d_plan.idxnupts.swap(d_idxnupts);

  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  n = numobins[0] * numobins[1] * numobins[2];
  calc_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
      binsperobin[0], binsperobin[1], binsperobin[2], d_binsize, d_numsubprob, maxsubprobsize,
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

template<typename T, int ndim, int ns>
static void cuspread3d_blockgather(
    const cufinufft_plan_t<T> &d_plan,
    const cuda_complex<T> *c, cuda_complex<T> *fw, int blksize) {
  static_assert(ndim==3, "unsupported dimensionality");

  size_t sharedplanorysize =
       d_plan.opts.gpu_obinsizex * d_plan.opts.gpu_obinsizey * d_plan.opts.gpu_obinsizez * sizeof(cuda_complex<T>);
  if (sharedplanorysize > 49152) {
    std::cerr << "[cuspread3d_blockgather] error: not enough shared memory" << std::endl;
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
}

// Functor to handle function selection (nuptsdriven, subprob, blockgather)
struct Spread3DDispatcher {
  template<int ns, typename T>
  void operator()(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                  cuda_complex<T> *fw, int blksize) const {
    switch (d_plan.opts.gpu_method) {
    case 1:
      return cuspread_nupts_driven<T, 3, ns>(d_plan, c, fw, blksize);
    case 2:
      return cuspread_subprob<T, 3, ns>(d_plan, c, fw, blksize);
    case 3:
      return cuspread_output_driven<T, 3, ns>(d_plan, c, fw, blksize);
    case 4:
      return cuspread3d_blockgather<T, 3, ns>(d_plan, c, fw, blksize);
    default:
      std::cerr << "[cuspread3d] error: invalid method " +
                       std::to_string(d_plan.opts.gpu_method) +
                       ", should be 1, 2, 3 or 4\n";
      throw int(FINUFFT_ERR_METHOD_NOTVALID);
    }
  }
};

// Updated cuspread3d using generic dispatch
template<typename T>
void cuspread3d(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                cuda_complex<T> *fw, int blksize) {
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
                                            d_plan, c, fw, blksize);
}
template void cuspread3d<float>(const cufinufft_plan_t<float> &d_plan,
                                const cuda_complex<float> *c, cuda_complex<float> *fw,
                                int blksize);
template void cuspread3d<double>(const cufinufft_plan_t<double> &d_plan,
                                 const cuda_complex<double> *c, cuda_complex<double> *fw,
                                 int blksize);

template<typename T> void cuspread3d_prop(cufinufft_plan_t<T> &d_plan) {
  if (d_plan.opts.gpu_method == 1) cuspread_nuptsdriven_prop<T, 3>(d_plan);
  if (d_plan.opts.gpu_method == 2) cuspread_subprob_and_OD_prop<T, 3>(d_plan);
  if (d_plan.opts.gpu_method == 3) cuspread_subprob_and_OD_prop<T, 3>(d_plan);
  if (d_plan.opts.gpu_method == 4) cuspread3d_blockgather_prop<T, 3>(d_plan);
}
template void cuspread3d_prop(cufinufft_plan_t<float> &d_plan);
template void cuspread3d_prop(cufinufft_plan_t<double> &d_plan);

} // namespace spreadinterp
} // namespace cufinufft

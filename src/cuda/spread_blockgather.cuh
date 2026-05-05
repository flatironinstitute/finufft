// Method body: 3D block-gather spreading (gpu_method = 4). 3D-only.

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

// Functions for block-gather spreading
// Only implemented for 3D at the moment and potentially obsolete.

inline __host__ __device__ int calc_global_index(cuda::std::array<int, 3> idx,
                                                 cuda::std::array<int, 3> on,
                                                 cuda::std::array<int, 3> bn) {
  cuda::std::array<int, 3> oi{idx[0] / bn[0], idx[1] / bn[1], idx[2] / bn[2]};
  return (oi[0] + oi[1] * on[0] + oi[2] * on[1] * on[0]) * (bn[0] * bn[1] * bn[2]) +
         (idx[0] % bn[0] + idx[1] % bn[1] * bn[0] + idx[2] % bn[2] * bn[1] * bn[0]);
}

inline __global__ void calc_subprob_3d_v1(cuda::std::array<int, 3> binsperobin,
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

inline __global__ void map_b_into_subprob_3d_v1(int *d_subprob_to_obin,
                                                const int *d_subprobstartpts,
                                                const int *d_numsubprob, int numbins) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numbins;
       i += gridDim.x * blockDim.x)
    for (int j = 0; j < d_numsubprob[i]; j++)
      d_subprob_to_obin[d_subprobstartpts[i] + j] = i;
}

inline __global__ void fill_ghost_bins(cuda::std::array<int, 3> binsperobin,
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

inline __global__ void ghost_bin_pts_index(
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
__global__ void locate_nupts_to_bins_ghost(
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
__global__ void calc_inverse_of_global_sort_index_ghost(
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
__global__ void spread_3d_block_gather(cufinufft_gpu_data<T> p, const cuda_complex<T> *c,
                                       cuda_complex<T> *fw) {
  static_assert(ndim == 3, "unsupported dimensionality");

  T es_c             = p.es_c;
  T es_beta          = p.es_beta;
  T sigma            = p.sigma;
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

template<typename T, int ndim, int ns>
void spread_blockgather_3d_launch(const cufinufft_plan_t<T> &d_plan,
                                  const cuda_complex<T> *c, cuda_complex<T> *fw,
                                  int blksize) {
  if constexpr (ndim == 3) {
    size_t sharedplanorysize = d_plan.opts.gpu_obinsizex * d_plan.opts.gpu_obinsizey *
                               d_plan.opts.gpu_obinsizez * sizeof(cuda_complex<T>);
    if (sharedplanorysize > 49152) {
      std::cerr << "[spread_blockgather_3d_launch] error: not enough shared memory"
                << std::endl;
      throw int(FINUFFT_ERR_INSUFFICIENT_SHMEM);
    }

    const auto launch = [&](auto kernel) {
      //   cufinufft_set_shared_memory(kernel, d_plan);
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

template<typename T> struct SpreadBlockGatherCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    spread_blockgather_3d_launch<T, 3, Ns>(p, c, fw, blksize);
  }
};

template<typename T>
void do_spread_blockgather_3d(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                              cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadBlockGatherCaller<T> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T> void do_indexSort_blockgather_3d(cufinufft_plan_t<T> &p) {
  constexpr int ndim = 3;
  auto &stream       = p.stream;
  int M              = p.M;

  int maxsubprobsize                  = p.opts.gpu_maxsubprobsize;
  cuda::std::array<int, 3> o_bin_size = {p.opts.gpu_obinsizex, p.opts.gpu_obinsizey,
                                         p.opts.gpu_obinsizez};

  if (p.nf123[0] % o_bin_size[0] != 0 || p.nf123[1] % o_bin_size[1] != 0 ||
      p.nf123[2] % o_bin_size[2] != 0) {
    std::cerr << "[indexSort_blockgather_3d] error:\n";
    std::cerr << "       mod(nf(1|2|3), opts.gpu_obinsize(x|y|z)) != 0" << std::endl;
    std::cerr << "       (nf1, nf2, nf3) = (" << p.nf123[0] << ", " << p.nf123[1] << ", "
              << p.nf123[2] << ")" << std::endl;
    std::cerr << "       (obinsizex, obinsizey, obinsizez) = (" << o_bin_size[0] << ", "
              << o_bin_size[1] << ", " << o_bin_size[2] << ")" << std::endl;
    throw int(FINUFFT_ERR_BINSIZE_NOTVALID);
  }

  cuda::std::array<int, 3> numobins;
  for (int idim = 0; idim < ndim; ++idim)
    numobins[idim] = ceil((T)p.nf123[idim] / o_bin_size[idim]);

  cuda::std::array<int, 3> bin_size = {p.opts.gpu_binsizex, p.opts.gpu_binsizey,
                                       p.opts.gpu_binsizez};
  if (o_bin_size[0] % bin_size[0] != 0 || o_bin_size[1] % bin_size[1] != 0 ||
      o_bin_size[2] % bin_size[2] != 0) {
    std::cerr << "[indexSort_blockgather_3d] error:\n";
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

  int *d_binsize         = dethrust(p.binsize);
  int *d_sortidx         = dethrust(p.sortidx);
  int *d_binstartpts     = dethrust(p.binstartpts);
  int *d_numsubprob      = dethrust(p.numsubprob);
  int *d_subprobstartpts = dethrust(p.subprobstartpts);

  checkCudaErrors(cudaMemsetAsync(
      d_binsize, 0, numbins[0] * numbins[1] * numbins[2] * sizeof(int), stream));

  locate_nupts_to_bins_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size, numobins, binsperobin, d_binsize, p.kxyz, d_sortidx, p.nf123);
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
  p.idxnupts.resize(totalNUpts);

  calc_inverse_of_global_sort_index_ghost<<<(M + 1024 - 1) / 1024, 1024, 0, stream>>>(
      M, bin_size, numobins, binsperobin, d_binstartpts, d_sortidx, p.kxyz,
      dethrust(p.idxnupts), p.nf123);

  threadsPerBlock = {2, 2, 2};

  blocks.x = (threadsPerBlock.x + numbins[0] - 1) / threadsPerBlock.x;
  blocks.y = (threadsPerBlock.y + numbins[1] - 1) / threadsPerBlock.y;
  blocks.z = (threadsPerBlock.z + numbins[2] - 1) / threadsPerBlock.z;

  ghost_bin_pts_index<<<blocks, threadsPerBlock, 0, stream>>>(
      binsperobin, numobins, d_binsize, dethrust(p.idxnupts), d_binstartpts, M);

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
  p.subprob_to_bin.resize(totalnumsubprob);
  map_b_into_subprob_3d_v1<<<(n + 1024 - 1) / 1024, 1024, 0, stream>>>(
      dethrust(p.subprob_to_bin), d_subprobstartpts, d_numsubprob, n);

  p.totalnumsubprob = totalnumsubprob;
}

extern template void do_spread_blockgather_3d<float>(const cufinufft_plan_t<float> &,
                                                     const cuda_complex<float> *,
                                                     cuda_complex<float> *, int);
extern template void do_spread_blockgather_3d<double>(const cufinufft_plan_t<double> &,
                                                      const cuda_complex<double> *,
                                                      cuda_complex<double> *, int);
extern template void do_indexSort_blockgather_3d<float>(cufinufft_plan_t<float> &);
extern template void do_indexSort_blockgather_3d<double>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::spread_blockgather_3d(const cuda_complex<T> *c,
                                                cuda_complex<T> *fw, int blksize) const {
  if (this->dim != 3) throw int(FINUFFT_ERR_METHOD_NOTVALID);
  cufinufft::spreadinterp::do_spread_blockgather_3d<T>(*this, c, fw, blksize);
}

template<typename T> void cufinufft_plan_t<T>::indexSort_blockgather_3d() {
  if (this->dim != 3) throw int(FINUFFT_ERR_METHOD_NOTVALID);
  cufinufft::spreadinterp::do_indexSort_blockgather_3d<T>(*this);
}

// Method body: 3D block-gather spreading (gpu_method = 4). 3D-only.

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T> struct SpreadBlockGatherCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread3d_blockgather<T, 3, Ns>(p, c, fw, blksize);
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

template<typename T> void do_prep_blockgather_3d(cufinufft_plan_t<T> &p) {
  constexpr int ndim = 3;
  auto &stream       = p.stream;
  int M              = p.M;

  int maxsubprobsize                  = p.opts.gpu_maxsubprobsize;
  cuda::std::array<int, 3> o_bin_size = {p.opts.gpu_obinsizex, p.opts.gpu_obinsizey,
                                         p.opts.gpu_obinsizez};

  if (p.nf123[0] % o_bin_size[0] != 0 || p.nf123[1] % o_bin_size[1] != 0 ||
      p.nf123[2] % o_bin_size[2] != 0) {
    std::cerr << "[prep_blockgather_3d] error:\n";
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
    std::cerr << "[prep_blockgather_3d] error:\n";
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
extern template void do_prep_blockgather_3d<float>(cufinufft_plan_t<float> &);
extern template void do_prep_blockgather_3d<double>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::spread_blockgather_3d(const cuda_complex<T> *c,
                                                cuda_complex<T> *fw, int blksize) const {
  if (this->dim != 3) throw int(FINUFFT_ERR_METHOD_NOTVALID);
  cufinufft::spreadinterp::do_spread_blockgather_3d<T>(*this, c, fw, blksize);
}

template<typename T> void cufinufft_plan_t<T>::prep_blockgather_3d() {
  if (this->dim != 3) throw int(FINUFFT_ERR_METHOD_NOTVALID);
  cufinufft::spreadinterp::do_prep_blockgather_3d<T>(*this);
}

// Method body: subproblem spreading (gpu_method = 2). Also owns the prep
// shared with output-driven (gpu_method = 3) — both call the same prop.

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct SpreadSubprobCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread_subprob<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_spread_subprob(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                       cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadSubprobCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T, int Ndim> void do_prep_subprob_and_OD(cufinufft_plan_t<T> &p) {
  cuda::std::array<int, 3> binsizes = {p.opts.gpu_binsizex, p.opts.gpu_binsizey,
                                       p.opts.gpu_binsizez};

  auto nbins          = get_nbins<Ndim>(p.nf123, binsizes);
  const int nbins_tot = nbins_total(nbins);
  const cuda::std::array<T, 3> inv_binsizes{T(1) / binsizes[0], T(1) / binsizes[1],
                                            T(1) / binsizes[2]};

  checkCudaErrors(
      cudaMemsetAsync(dethrust(p.binsize), 0, nbins_tot * sizeof(int), p.stream));
  calc_bin_size_noghost<T, Ndim><<<(p.M + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
      p.M, p.nf123, inv_binsizes, nbins, dethrust(p.binsize), p.kxyz,
      dethrust(p.sortidx));
  THROW_IF_CUDA_ERROR
  thrust::exclusive_scan(thrust::cuda::par.on(p.stream), p.binsize.begin(),
                         p.binsize.end(), p.binstartpts.begin());
  THROW_IF_CUDA_ERROR
  calc_inverse_of_global_sort_idx<T, Ndim>
      <<<(p.M + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
          p.M, inv_binsizes, nbins, dethrust(p.binstartpts), dethrust(p.sortidx), p.kxyz,
          dethrust(p.idxnupts), p.nf123);
  THROW_IF_CUDA_ERROR

  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  calc_subprob<<<(p.M + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
      dethrust(p.binsize), dethrust(p.numsubprob), p.opts.gpu_maxsubprobsize, nbins_tot);
  THROW_IF_CUDA_ERROR

  thrust::inclusive_scan(thrust::cuda::par.on(p.stream), p.numsubprob.begin(),
                         p.numsubprob.begin() + nbins_tot, p.subprobstartpts.begin() + 1);
  THROW_IF_CUDA_ERROR

  int totalnumsubprob;
  checkCudaErrors(cudaMemsetAsync(dethrust(p.subprobstartpts), 0, sizeof(int), p.stream));
  checkCudaErrors(
      cudaMemcpyAsync(&totalnumsubprob, &(dethrust(p.subprobstartpts)[nbins_tot]),
                      sizeof(int), cudaMemcpyDeviceToHost, p.stream));
  cudaStreamSynchronize(p.stream);
  p.subprob_to_bin.resize(totalnumsubprob);

  map_b_into_subprob<<<(nbins[0] * nbins[1] + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
      dethrust(p.subprob_to_bin), dethrust(p.subprobstartpts), dethrust(p.numsubprob),
      nbins_tot);

  p.totalnumsubprob = totalnumsubprob;
}

extern template void do_spread_subprob<float, 1>(const cufinufft_plan_t<float> &,
                                                 const cuda_complex<float> *,
                                                 cuda_complex<float> *, int);
extern template void do_spread_subprob<float, 2>(const cufinufft_plan_t<float> &,
                                                 const cuda_complex<float> *,
                                                 cuda_complex<float> *, int);
extern template void do_spread_subprob<float, 3>(const cufinufft_plan_t<float> &,
                                                 const cuda_complex<float> *,
                                                 cuda_complex<float> *, int);
extern template void do_spread_subprob<double, 1>(const cufinufft_plan_t<double> &,
                                                  const cuda_complex<double> *,
                                                  cuda_complex<double> *, int);
extern template void do_spread_subprob<double, 2>(const cufinufft_plan_t<double> &,
                                                  const cuda_complex<double> *,
                                                  cuda_complex<double> *, int);
extern template void do_spread_subprob<double, 3>(const cufinufft_plan_t<double> &,
                                                  const cuda_complex<double> *,
                                                  cuda_complex<double> *, int);

extern template void do_prep_subprob_and_OD<float, 1>(cufinufft_plan_t<float> &);
extern template void do_prep_subprob_and_OD<float, 2>(cufinufft_plan_t<float> &);
extern template void do_prep_subprob_and_OD<float, 3>(cufinufft_plan_t<float> &);
extern template void do_prep_subprob_and_OD<double, 1>(cufinufft_plan_t<double> &);
extern template void do_prep_subprob_and_OD<double, 2>(cufinufft_plan_t<double> &);
extern template void do_prep_subprob_and_OD<double, 3>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::spread_subprob(const cuda_complex<T> *c, cuda_complex<T> *fw,
                                         int blksize) const {
  using cufinufft::spreadinterp::do_spread_subprob;
  switch (this->dim) {
  case 1:
    return do_spread_subprob<T, 1>(*this, c, fw, blksize);
  case 2:
    return do_spread_subprob<T, 2>(*this, c, fw, blksize);
  case 3:
    return do_spread_subprob<T, 3>(*this, c, fw, blksize);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

template<typename T> void cufinufft_plan_t<T>::prep_subprob_and_OD() {
  using cufinufft::spreadinterp::do_prep_subprob_and_OD;
  switch (this->dim) {
  case 1:
    return do_prep_subprob_and_OD<T, 1>(*this);
  case 2:
    return do_prep_subprob_and_OD<T, 2>(*this);
  case 3:
    return do_prep_subprob_and_OD<T, 3>(*this);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

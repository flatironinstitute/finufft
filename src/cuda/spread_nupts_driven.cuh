// Method body: nupts-driven spreading (gpu_method = 1).
//
// Defines the per-dim worker templates (do_spread_nupts_driven /
// do_prep_nupts_driven) and the plan-method bodies that dispatch on
// dim. Workers are extern-template declared here and instantiated in
// the per-dim TUs spread_nupts_driven_{1,2,3}d.cu.
//
// Mirrors the CPU split pattern: definitions in the header,
// per-dim explicit instantiation in TUs, so nvcc parallelism scales
// per (method, dim) instead of per method.

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct SpreadNuptsDrivenCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread_nupts_driven<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_spread_nupts_driven(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                            cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadNuptsDrivenCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T, int Ndim> void do_prep_nupts_driven(cufinufft_plan_t<T> &p) {
  if (p.opts.gpu_sort) {
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
            p.M, inv_binsizes, nbins, dethrust(p.binstartpts), dethrust(p.sortidx),
            p.kxyz, dethrust(p.idxnupts), p.nf123);
    THROW_IF_CUDA_ERROR
  } else {
    thrust::sequence(thrust::cuda::par.on(p.stream), p.idxnupts.begin(),
                     p.idxnupts.begin() + p.M);
    THROW_IF_CUDA_ERROR
  }
}

extern template void do_spread_nupts_driven<float, 1>(const cufinufft_plan_t<float> &,
                                                      const cuda_complex<float> *,
                                                      cuda_complex<float> *, int);
extern template void do_spread_nupts_driven<float, 2>(const cufinufft_plan_t<float> &,
                                                      const cuda_complex<float> *,
                                                      cuda_complex<float> *, int);
extern template void do_spread_nupts_driven<float, 3>(const cufinufft_plan_t<float> &,
                                                      const cuda_complex<float> *,
                                                      cuda_complex<float> *, int);
extern template void do_spread_nupts_driven<double, 1>(const cufinufft_plan_t<double> &,
                                                       const cuda_complex<double> *,
                                                       cuda_complex<double> *, int);
extern template void do_spread_nupts_driven<double, 2>(const cufinufft_plan_t<double> &,
                                                       const cuda_complex<double> *,
                                                       cuda_complex<double> *, int);
extern template void do_spread_nupts_driven<double, 3>(const cufinufft_plan_t<double> &,
                                                       const cuda_complex<double> *,
                                                       cuda_complex<double> *, int);

extern template void do_prep_nupts_driven<float, 1>(cufinufft_plan_t<float> &);
extern template void do_prep_nupts_driven<float, 2>(cufinufft_plan_t<float> &);
extern template void do_prep_nupts_driven<float, 3>(cufinufft_plan_t<float> &);
extern template void do_prep_nupts_driven<double, 1>(cufinufft_plan_t<double> &);
extern template void do_prep_nupts_driven<double, 2>(cufinufft_plan_t<double> &);
extern template void do_prep_nupts_driven<double, 3>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::spread_nupts_driven(const cuda_complex<T> *c,
                                              cuda_complex<T> *fw, int blksize) const {
  using cufinufft::spreadinterp::do_spread_nupts_driven;
  switch (this->dim) {
  case 1:
    return do_spread_nupts_driven<T, 1>(*this, c, fw, blksize);
  case 2:
    return do_spread_nupts_driven<T, 2>(*this, c, fw, blksize);
  case 3:
    return do_spread_nupts_driven<T, 3>(*this, c, fw, blksize);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

template<typename T> void cufinufft_plan_t<T>::prep_nupts_driven() {
  using cufinufft::spreadinterp::do_prep_nupts_driven;
  switch (this->dim) {
  case 1:
    return do_prep_nupts_driven<T, 1>(*this);
  case 2:
    return do_prep_nupts_driven<T, 2>(*this);
  case 3:
    return do_prep_nupts_driven<T, 3>(*this);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

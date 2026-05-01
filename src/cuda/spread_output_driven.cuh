// Method body: output-driven spreading (gpu_method = 3). Prep is shared
// with gpu_method=2 and provided by spread_subprob.cuh.

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct SpreadOutputDrivenCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread_output_driven<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_spread_output_driven(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                             cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadOutputDrivenCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

extern template void do_spread_output_driven<float, 1>(const cufinufft_plan_t<float> &,
                                                       const cuda_complex<float> *,
                                                       cuda_complex<float> *, int);
extern template void do_spread_output_driven<float, 2>(const cufinufft_plan_t<float> &,
                                                       const cuda_complex<float> *,
                                                       cuda_complex<float> *, int);
extern template void do_spread_output_driven<float, 3>(const cufinufft_plan_t<float> &,
                                                       const cuda_complex<float> *,
                                                       cuda_complex<float> *, int);
extern template void do_spread_output_driven<double, 1>(const cufinufft_plan_t<double> &,
                                                        const cuda_complex<double> *,
                                                        cuda_complex<double> *, int);
extern template void do_spread_output_driven<double, 2>(const cufinufft_plan_t<double> &,
                                                        const cuda_complex<double> *,
                                                        cuda_complex<double> *, int);
extern template void do_spread_output_driven<double, 3>(const cufinufft_plan_t<double> &,
                                                        const cuda_complex<double> *,
                                                        cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::spread_output_driven(const cuda_complex<T> *c,
                                               cuda_complex<T> *fw, int blksize) const {
  using cufinufft::spreadinterp::do_spread_output_driven;
  switch (this->dim) {
  case 1:
    return do_spread_output_driven<T, 1>(*this, c, fw, blksize);
  case 2:
    return do_spread_output_driven<T, 2>(*this, c, fw, blksize);
  case 3:
    return do_spread_output_driven<T, 3>(*this, c, fw, blksize);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

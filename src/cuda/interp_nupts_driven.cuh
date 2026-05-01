// Method body: nupts-driven interpolation (gpu_method = 1).

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct InterpNuptsDrivenCaller {
  const cufinufft_plan_t<T> &p;
  cuda_complex<T> *c;
  const cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuinterp_nuptsdriven<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_interp_nupts_driven(const cufinufft_plan_t<T> &p, cuda_complex<T> *c,
                            const cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  InterpNuptsDrivenCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

extern template void do_interp_nupts_driven<float, 1>(const cufinufft_plan_t<float> &,
                                                      cuda_complex<float> *,
                                                      const cuda_complex<float> *, int);
extern template void do_interp_nupts_driven<float, 2>(const cufinufft_plan_t<float> &,
                                                      cuda_complex<float> *,
                                                      const cuda_complex<float> *, int);
extern template void do_interp_nupts_driven<float, 3>(const cufinufft_plan_t<float> &,
                                                      cuda_complex<float> *,
                                                      const cuda_complex<float> *, int);
extern template void do_interp_nupts_driven<double, 1>(const cufinufft_plan_t<double> &,
                                                       cuda_complex<double> *,
                                                       const cuda_complex<double> *, int);
extern template void do_interp_nupts_driven<double, 2>(const cufinufft_plan_t<double> &,
                                                       cuda_complex<double> *,
                                                       const cuda_complex<double> *, int);
extern template void do_interp_nupts_driven<double, 3>(const cufinufft_plan_t<double> &,
                                                       cuda_complex<double> *,
                                                       const cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::interp_nupts_driven(
    cuda_complex<T> *c, const cuda_complex<T> *fw, int blksize) const {
  using cufinufft::spreadinterp::do_interp_nupts_driven;
  switch (this->dim) {
  case 1:
    return do_interp_nupts_driven<T, 1>(*this, c, fw, blksize);
  case 2:
    return do_interp_nupts_driven<T, 2>(*this, c, fw, blksize);
  case 3:
    return do_interp_nupts_driven<T, 3>(*this, c, fw, blksize);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

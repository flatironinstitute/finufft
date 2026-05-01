// Method body: subproblem interpolation (gpu_method = 2).

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct InterpSubprobCaller {
  const cufinufft_plan_t<T> &p;
  cuda_complex<T> *c;
  const cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuinterp_subprob<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_interp_subprob(const cufinufft_plan_t<T> &p, cuda_complex<T> *c,
                       const cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  InterpSubprobCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

extern template void do_interp_subprob<float, 1>(const cufinufft_plan_t<float> &,
                                                 cuda_complex<float> *,
                                                 const cuda_complex<float> *, int);
extern template void do_interp_subprob<float, 2>(const cufinufft_plan_t<float> &,
                                                 cuda_complex<float> *,
                                                 const cuda_complex<float> *, int);
extern template void do_interp_subprob<float, 3>(const cufinufft_plan_t<float> &,
                                                 cuda_complex<float> *,
                                                 const cuda_complex<float> *, int);
extern template void do_interp_subprob<double, 1>(const cufinufft_plan_t<double> &,
                                                  cuda_complex<double> *,
                                                  const cuda_complex<double> *, int);
extern template void do_interp_subprob<double, 2>(const cufinufft_plan_t<double> &,
                                                  cuda_complex<double> *,
                                                  const cuda_complex<double> *, int);
extern template void do_interp_subprob<double, 3>(const cufinufft_plan_t<double> &,
                                                  cuda_complex<double> *,
                                                  const cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::interp_subprob(cuda_complex<T> *c, const cuda_complex<T> *fw,
                                         int blksize) const {
  using cufinufft::spreadinterp::do_interp_subprob;
  switch (this->dim) {
  case 1:
    return do_interp_subprob<T, 1>(*this, c, fw, blksize);
  case 2:
    return do_interp_subprob<T, 2>(*this, c, fw, blksize);
  case 3:
    return do_interp_subprob<T, 3>(*this, c, fw, blksize);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

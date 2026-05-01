// Method TU: subproblem interpolation (gpu_method = 2).

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct SubprobInterpCaller {
  const cufinufft_plan_t<T> &p;
  cuda_complex<T> *c;
  const cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuinterp_subprob<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void cuinterp_subprob_op<T, Ndim>::exec(const cufinufft_plan_t<T> &p, cuda_complex<T> *c,
                                        const cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SubprobInterpCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template struct cuinterp_subprob_op<float, 1>;
template struct cuinterp_subprob_op<float, 2>;
template struct cuinterp_subprob_op<float, 3>;
template struct cuinterp_subprob_op<double, 1>;
template struct cuinterp_subprob_op<double, 2>;
template struct cuinterp_subprob_op<double, 3>;

} // namespace spreadinterp
} // namespace cufinufft

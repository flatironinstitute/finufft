// Method TU: subproblem spreading (gpu_method = 2). Also owns the prep entry
// shared with output-driven (gpu_method = 3) — both call the same prop.

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct SubprobSpreadCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread_subprob<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void cuspread_subprob_op<T, Ndim>::exec(const cufinufft_plan_t<T> &p,
                                        const cuda_complex<T> *c, cuda_complex<T> *fw,
                                        int blksize) {
  using namespace finufft::common;
  SubprobSpreadCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T, int Ndim>
void cuspread_subprob_op<T, Ndim>::prep(cufinufft_plan_t<T> &p) {
  cuspread_subprob_and_OD_prop<T, Ndim>(p);
}

template struct cuspread_subprob_op<float, 1>;
template struct cuspread_subprob_op<float, 2>;
template struct cuspread_subprob_op<float, 3>;
template struct cuspread_subprob_op<double, 1>;
template struct cuspread_subprob_op<double, 2>;
template struct cuspread_subprob_op<double, 3>;

} // namespace spreadinterp
} // namespace cufinufft

// Method TU: nupts-driven spreading (gpu_method = 1).
// Owns this method's spread entry and its bin-sort/index prep entry.

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

// CPU pattern: Caller struct captures runtime args; templated operator()<Ns>
// is invoked once per Ns by finufft::common::dispatch (compile-time matrix
// expansion). See include/finufft/spread.hpp::SpreadSubproblem1dCaller.
template<typename T, int Ndim> struct NuptsDrivenSpreadCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread_nupts_driven<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void cuspread_nupts_driven_op<T, Ndim>::exec(const cufinufft_plan_t<T> &p,
                                             const cuda_complex<T> *c,
                                             cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  NuptsDrivenSpreadCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T, int Ndim>
void cuspread_nupts_driven_op<T, Ndim>::prep(cufinufft_plan_t<T> &p) {
  cuspread_nuptsdriven_prop<T, Ndim>(p);
}

template struct cuspread_nupts_driven_op<float, 1>;
template struct cuspread_nupts_driven_op<float, 2>;
template struct cuspread_nupts_driven_op<float, 3>;
template struct cuspread_nupts_driven_op<double, 1>;
template struct cuspread_nupts_driven_op<double, 2>;
template struct cuspread_nupts_driven_op<double, 3>;

} // namespace spreadinterp
} // namespace cufinufft

// Method TU: output-driven spreading (gpu_method = 3). Prep is shared with
// gpu_method=2 and lives in spread_subprob.cu.

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct OutputDrivenSpreadCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread_output_driven<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void cuspread_output_driven_op<T, Ndim>::exec(const cufinufft_plan_t<T> &p,
                                              const cuda_complex<T> *c,
                                              cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  OutputDrivenSpreadCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template struct cuspread_output_driven_op<float, 1>;
template struct cuspread_output_driven_op<float, 2>;
template struct cuspread_output_driven_op<float, 3>;
template struct cuspread_output_driven_op<double, 1>;
template struct cuspread_output_driven_op<double, 2>;
template struct cuspread_output_driven_op<double, 3>;

} // namespace spreadinterp
} // namespace cufinufft

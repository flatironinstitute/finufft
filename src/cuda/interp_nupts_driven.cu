// Method TU: nupts-driven interpolation (gpu_method = 1).

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T, int Ndim> struct NuptsDrivenInterpCaller {
  const cufinufft_plan_t<T> &p;
  cuda_complex<T> *c;
  const cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuinterp_nuptsdriven<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void cuinterp_nupts_driven_op<T, Ndim>::exec(const cufinufft_plan_t<T> &p,
                                             cuda_complex<T> *c,
                                             const cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  NuptsDrivenInterpCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template struct cuinterp_nupts_driven_op<float, 1>;
template struct cuinterp_nupts_driven_op<float, 2>;
template struct cuinterp_nupts_driven_op<float, 3>;
template struct cuinterp_nupts_driven_op<double, 1>;
template struct cuinterp_nupts_driven_op<double, 2>;
template struct cuinterp_nupts_driven_op<double, 3>;

} // namespace spreadinterp
} // namespace cufinufft

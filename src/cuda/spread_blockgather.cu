// Method TU: 3D block-gather spreading (gpu_method = 4). 3D-only.

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

template<typename T> struct BlockGatherSpreadCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread3d_blockgather<T, 3, Ns>(p, c, fw, blksize);
  }
};

template<typename T>
void cuspread_3d_blockgather_op<T>::exec(const cufinufft_plan_t<T> &p,
                                         const cuda_complex<T> *c, cuda_complex<T> *fw,
                                         int blksize) {
  using namespace finufft::common;
  BlockGatherSpreadCaller<T> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T> void cuspread_3d_blockgather_op<T>::prep(cufinufft_plan_t<T> &p) {
  cuspread3d_blockgather_prop<T, 3>(p);
}

template struct cuspread_3d_blockgather_op<float>;
template struct cuspread_3d_blockgather_op<double>;

} // namespace spreadinterp
} // namespace cufinufft

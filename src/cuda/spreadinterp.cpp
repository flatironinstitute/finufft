// Plan-method dispatch entry points for spread / interp / their preparation.
//
// This TU is pure host C++: it dispatches on opts.gpu_method by calling
// other plan member methods. The per-method bodies (which contain
// __global__ kernels and __device__ helpers) live in the per-method
// (method, dim) instantiation TUs together with their member-method
// explicit instantiations.

#include <iostream>

#include <cufinufft/cufinufft_plan_t.hpp>
#include <finufft_errors.h>

template<typename T> void cufinufft_plan_t<T>::indexSort() {
  switch (opts.gpu_method) {
  case 1:
    indexSort_nupts_driven();
    return;
  case 2:
  case 3:
    indexSort_subprob_and_OD();
    return;
  case 4:
    indexSort_blockgather_3d();
    return;
  default:
    std::cerr << "[indexSort] error: incorrect gpu_method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
}

template<typename T>
void cufinufft_plan_t<T>::spread(const cuda_complex<T> *c, cuda_complex<T> *fw,
                                 int blksize) const {
  switch (opts.gpu_method) {
  case 1:
    return spread_nupts_driven(c, fw, blksize);
  case 2:
    return spread_subprob(c, fw, blksize);
  case 3:
    return spread_output_driven(c, fw, blksize);
  case 4:
    return spread_blockgather_3d(c, fw, blksize);
  default:
    std::cerr << "[spread] error: incorrect gpu_method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
}

template<typename T>
void cufinufft_plan_t<T>::interp(cuda_complex<T> *c, const cuda_complex<T> *fw,
                                 int blksize) const {
  switch (opts.gpu_method) {
  case 1:
    return interp_nupts_driven(c, fw, blksize);
  case 2:
    return interp_subprob(c, fw, blksize);
  default:
    std::cerr << "[interp] error: incorrect gpu_method\n";
    throw int(FINUFFT_ERR_METHOD_NOTVALID);
  }
}

template void cufinufft_plan_t<float>::indexSort();
template void cufinufft_plan_t<double>::indexSort();
template void cufinufft_plan_t<float>::spread(const cuda_complex<float> *,
                                              cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread(const cuda_complex<double> *,
                                               cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::interp(cuda_complex<float> *,
                                              const cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::interp(cuda_complex<double> *,
                                               const cuda_complex<double> *, int) const;

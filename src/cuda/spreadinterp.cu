// Plan-method dispatch entry points for spread / interp / their preparation.
//
// Each plan method body lives in a per-method .cuh; the heavy worker
// templates are extern-template declared there and instantiated per dim
// in the CMake-driven *_inst.cu translation units (one TU per (method,
// dim) via -DCUFINUFFT_DIM). This TU is the thin gpu_method dispatcher
// + a single home for all plan-method explicit instantiations (float
// and double).

#include "interp_nupts_driven.cuh"
#include "interp_subprob.cuh"
#include "spread_blockgather.cuh"
#include "spread_nupts_driven.cuh"
#include "spread_output_driven.cuh"
#include "spread_subprob.cuh"

template<typename T> void cufinufft_plan_t<T>::prep_spreadinterp() {
  switch (opts.gpu_method) {
  case 1:
    prep_nupts_driven();
    return;
  case 2:
  case 3:
    prep_subprob_and_OD();
    return;
  case 4:
    prep_blockgather_3d();
    return;
  default:
    std::cerr << "[prep_spreadinterp] error: incorrect gpu_method\n";
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

// gpu_method dispatchers
template void cufinufft_plan_t<float>::prep_spreadinterp();
template void cufinufft_plan_t<double>::prep_spreadinterp();
template void cufinufft_plan_t<float>::spread(const cuda_complex<float> *,
                                              cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread(const cuda_complex<double> *,
                                               cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::interp(cuda_complex<float> *,
                                              const cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::interp(cuda_complex<double> *,
                                               const cuda_complex<double> *, int) const;

// Per-method plan-method bodies — bodies are templates defined in the .cuh
// files included above; they call extern-template workers that are
// instantiated in the *_inst.cu TUs.
template void cufinufft_plan_t<float>::prep_nupts_driven();
template void cufinufft_plan_t<double>::prep_nupts_driven();
template void cufinufft_plan_t<float>::prep_subprob_and_OD();
template void cufinufft_plan_t<double>::prep_subprob_and_OD();
template void cufinufft_plan_t<float>::prep_blockgather_3d();
template void cufinufft_plan_t<double>::prep_blockgather_3d();

template void cufinufft_plan_t<float>::spread_nupts_driven(
    const cuda_complex<float> *, cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread_nupts_driven(
    const cuda_complex<double> *, cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::spread_subprob(const cuda_complex<float> *,
                                                      cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread_subprob(const cuda_complex<double> *,
                                                       cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::spread_output_driven(
    const cuda_complex<float> *, cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread_output_driven(
    const cuda_complex<double> *, cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::spread_blockgather_3d(
    const cuda_complex<float> *, cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::spread_blockgather_3d(
    const cuda_complex<double> *, cuda_complex<double> *, int) const;

template void cufinufft_plan_t<float>::interp_nupts_driven(
    cuda_complex<float> *, const cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::interp_nupts_driven(
    cuda_complex<double> *, const cuda_complex<double> *, int) const;
template void cufinufft_plan_t<float>::interp_subprob(
    cuda_complex<float> *, const cuda_complex<float> *, int) const;
template void cufinufft_plan_t<double>::interp_subprob(
    cuda_complex<double> *, const cuda_complex<double> *, int) const;

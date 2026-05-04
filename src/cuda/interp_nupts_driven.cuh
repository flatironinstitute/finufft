// Method body: nupts-driven interpolation (gpu_method = 1).

#pragma once

#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

// Nupts-driven interpolation kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void interp_nupts_driven(
    cufinufft_gpu_data<T> p, cuda_complex<T> *c, const cuda_complex<T> *fw) {
  T es_c    = p.es_c;
  T es_beta = p.es_beta;
  T sigma   = p.sigma;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < p.M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(p.idxnupts + i);

    auto [ker, start] = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, sigma, es_c, es_beta);

    cuda_complex<T> cnow{0, 0};
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        cnow += loadReadOnly(fw + ix) * ker[0][x0];
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
        const auto inidx0 = iy * p.nf123[0];
        cuda_complex<T> cnowx{0, 0};
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
          cnowx += loadReadOnly(fw + inidx0 + ix) * ker[0][x0];
        cnow += cnowx * ker[1][y0];
      }
    }
    if constexpr (ndim == 3) {
      cuda::std::array<int, ns> xidx;
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        xidx[x0] = ix;
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= p.nf123[2]) ? 0 : iz + 1) {
        const auto inidx0 = iz * p.nf123[1] * p.nf123[0];
        cuda_complex<T> cnowy{0, 0};
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
          const auto inidx1 = inidx0 + iy * p.nf123[0];
          cuda_complex<T> cnowx{0, 0};
          for (int x0 = 0; x0 < ns; ++x0)
            cnowx += loadReadOnly(fw + inidx1 + xidx[x0]) * ker[0][x0];
          cnowy += cnowx * ker[1][y0];
        }
        cnow += cnowy * ker[2][z0];
      }
    }
    storeCacheStreaming(c + nuptsidx, cnow);
  }
}

// Nupts-driven interpolation CPU driver
template<typename T, int ndim, int ns>
void cuinterp_nuptsdriven(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                          const cuda_complex<T> *fw, int blksize) {
  const dim3 threadsPerBlock{
      optimal_interp_block_threads(d_plan.opts.gpu_device_id, d_plan.M), 1u, 1u};
  const dim3 blocks{(d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1};

  const auto launch = [&](auto kernel) {
    for (int t = 0; t < blksize; t++) {
      kernel<<<blocks, threadsPerBlock, 0, d_plan.stream>>>(d_plan, c + t * d_plan.M,
                                                            fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(interp_nupts_driven<T, 1, ndim, ns>)
                                     : launch(interp_nupts_driven<T, 0, ndim, ns>);
}

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

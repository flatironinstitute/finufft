// Method body: nupts-driven spreading (gpu_method = 1).
//
// Defines the per-dim worker templates (do_spread_nupts_driven /
// do_indexSort_nupts_driven) and the plan-method bodies that dispatch on
// dim. Workers are extern-template declared here and instantiated in
// the per-dim TUs spread_nupts_driven_{1,2,3}d.cu.
//
// Mirrors the CPU split pattern: definitions in the header,
// per-dim explicit instantiation in TUs, so nvcc parallelism scales
// per (method, dim) instead of per method.

#pragma once

#include "spreadinterp_common.cuh"
#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

// Nupts-driven spreading kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void spread_nupts_driven(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw) {

  T sigma   = p.sigma;
  T es_c    = p.es_c;
  T es_beta = p.es_beta;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < p.M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(p.idxnupts + i);
    auto [ker, start]   = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, sigma, es_c, es_beta);

    const auto val = loadReadOnly(c + nuptsidx);
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        atomicAddComplexGlobal<T>(fw + ix, ker[0][x0] * val);
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
        const auto outidx0   = iy * p.nf123[0];
        cuda_complex<T> valy = val * ker[1][y0];
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
          atomicAddComplexGlobal<T>(fw + outidx0 + ix, ker[0][x0] * valy);
      }
    } else {
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= p.nf123[2]) ? 0 : iz + 1) {
        const auto outidx0   = iz * p.nf123[1] * p.nf123[0];
        cuda_complex<T> valz = val * ker[2][z0];
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
          const auto outidx1   = outidx0 + iy * p.nf123[0];
          cuda_complex<T> valy = valz * ker[1][y0];
          for (int x0 = 0, ix = start[0]; x0 < ns;
               ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1) {
            atomicAddComplexGlobal<T>(fw + outidx1 + ix, ker[0][x0] * valy);
          }
        }
      }
    }
  }
}

// Nupts-driven spreading CPU driver
template<typename T, int ndim, int ns>
void spread_nupts_driven_launch(const cufinufft_plan_t<T> &d_plan,
                                const cuda_complex<T> *c, cuda_complex<T> *fw,
                                int blksize) {
  const dim3 threadsPerBlock{16, 1, 1};
  const dim3 blocks{(unsigned(d_plan.M) + 15) / 16, 1, 1};

  const auto launch = [&](auto kernel) {
    for (int t = 0; t < blksize; t++) {
      kernel<<<blocks, threadsPerBlock, 0, d_plan.stream>>>(d_plan, c + t * d_plan.M,
                                                            fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(spread_nupts_driven<T, 1, ndim, ns>)
                                     : launch(spread_nupts_driven<T, 0, ndim, ns>);
}

template<typename T, int Ndim> struct SpreadNuptsDrivenCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    spread_nupts_driven_launch<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_spread_nupts_driven(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                            cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadNuptsDrivenCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T, int Ndim> void do_indexSort_nupts_driven(cufinufft_plan_t<T> &p) {
  if (p.opts.gpu_sort) {
    auto layout         = compute_bin_layout<T, Ndim>(p.opts, p.nf123);
    auto &nbins         = layout.nbins;
    const int nbins_tot = layout.nbins_tot;
    auto &inv_binsizes  = layout.inv_binsizes;

    checkCudaErrors(
        cudaMemsetAsync(dethrust(p.binsize), 0, nbins_tot * sizeof(int), p.stream));
    calc_bin_size_noghost<T, Ndim><<<(p.M + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
        p.M, p.nf123, inv_binsizes, nbins, dethrust(p.binsize), p.kxyz,
        dethrust(p.sortidx));
    THROW_IF_CUDA_ERROR

    thrust::exclusive_scan(thrust::cuda::par.on(p.stream), p.binsize.begin(),
                           p.binsize.end(), p.binstartpts.begin());
    THROW_IF_CUDA_ERROR

    calc_inverse_of_global_sort_idx<T, Ndim>
        <<<(p.M + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
            p.M, inv_binsizes, nbins, dethrust(p.binstartpts), dethrust(p.sortidx),
            p.kxyz, dethrust(p.idxnupts), p.nf123);
    THROW_IF_CUDA_ERROR
  } else {
    thrust::sequence(thrust::cuda::par.on(p.stream), p.idxnupts.begin(),
                     p.idxnupts.begin() + p.M);
    THROW_IF_CUDA_ERROR
  }
}

extern template void do_spread_nupts_driven<float, 1>(const cufinufft_plan_t<float> &,
                                                      const cuda_complex<float> *,
                                                      cuda_complex<float> *, int);
extern template void do_spread_nupts_driven<float, 2>(const cufinufft_plan_t<float> &,
                                                      const cuda_complex<float> *,
                                                      cuda_complex<float> *, int);
extern template void do_spread_nupts_driven<float, 3>(const cufinufft_plan_t<float> &,
                                                      const cuda_complex<float> *,
                                                      cuda_complex<float> *, int);
extern template void do_spread_nupts_driven<double, 1>(const cufinufft_plan_t<double> &,
                                                       const cuda_complex<double> *,
                                                       cuda_complex<double> *, int);
extern template void do_spread_nupts_driven<double, 2>(const cufinufft_plan_t<double> &,
                                                       const cuda_complex<double> *,
                                                       cuda_complex<double> *, int);
extern template void do_spread_nupts_driven<double, 3>(const cufinufft_plan_t<double> &,
                                                       const cuda_complex<double> *,
                                                       cuda_complex<double> *, int);

extern template void do_indexSort_nupts_driven<float, 1>(cufinufft_plan_t<float> &);
extern template void do_indexSort_nupts_driven<float, 2>(cufinufft_plan_t<float> &);
extern template void do_indexSort_nupts_driven<float, 3>(cufinufft_plan_t<float> &);
extern template void do_indexSort_nupts_driven<double, 1>(cufinufft_plan_t<double> &);
extern template void do_indexSort_nupts_driven<double, 2>(cufinufft_plan_t<double> &);
extern template void do_indexSort_nupts_driven<double, 3>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::spread_nupts_driven(const cuda_complex<T> *c,
                                              cuda_complex<T> *fw, int blksize) const {
  using cufinufft::spreadinterp::do_spread_nupts_driven;
  switch (this->dim) {
  case 1:
    return do_spread_nupts_driven<T, 1>(*this, c, fw, blksize);
  case 2:
    return do_spread_nupts_driven<T, 2>(*this, c, fw, blksize);
  case 3:
    return do_spread_nupts_driven<T, 3>(*this, c, fw, blksize);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

template<typename T> void cufinufft_plan_t<T>::indexSort_nupts_driven() {
  using cufinufft::spreadinterp::do_indexSort_nupts_driven;
  switch (this->dim) {
  case 1:
    return do_indexSort_nupts_driven<T, 1>(*this);
  case 2:
    return do_indexSort_nupts_driven<T, 2>(*this);
  case 3:
    return do_indexSort_nupts_driven<T, 3>(*this);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

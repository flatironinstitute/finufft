// Per-dim instantiation TU: nupts-driven spread (gpu_method = 1).
// Compiled three times by CMake (foreach dim), each invocation passing
// -DCUFINUFFT_DIM={1,2,3}; produces one object per dim.

#include "spreadinterp_common.cuh"
#include <cufinufft/spreadinterp.hpp>

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

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
      THROW_IF_CUDA_ERROR();
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
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD<T>>;
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
    THROW_IF_CUDA_ERROR();

    thrust::exclusive_scan(thrust::cuda::par.on(p.stream), p.binsize.begin(),
                           p.binsize.end(), p.binstartpts.begin());
    THROW_IF_CUDA_ERROR();

    calc_inverse_of_global_sort_idx<T, Ndim>
        <<<(p.M + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
            p.M, inv_binsizes, nbins, dethrust(p.binstartpts), dethrust(p.sortidx),
            p.kxyz, dethrust(p.idxnupts), p.nf123);
    THROW_IF_CUDA_ERROR();
  } else {
    thrust::sequence(thrust::cuda::par.on(p.stream), p.idxnupts.begin(),
                     p.idxnupts.begin() + p.M);
    THROW_IF_CUDA_ERROR();
  }
}

template void do_spread_nupts_driven<float, CUFINUFFT_DIM>(
    const cufinufft_plan_t<float> &, const cuda_complex<float> *, cuda_complex<float> *,
    int);
template void do_spread_nupts_driven<double, CUFINUFFT_DIM>(
    const cufinufft_plan_t<double> &, const cuda_complex<double> *,
    cuda_complex<double> *, int);

template void do_indexSort_nupts_driven<float, CUFINUFFT_DIM>(cufinufft_plan_t<float> &);
template void do_indexSort_nupts_driven<double, CUFINUFFT_DIM>(
    cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

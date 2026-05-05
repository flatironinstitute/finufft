// Method body: subproblem spreading (gpu_method = 2). Also owns the prep
// shared with output-driven (gpu_method = 3) — both call the same prop.

#pragma once

#include "spreadinterp_common.cuh"
#include <cufinufft/spreadinterp.hpp>

namespace cufinufft {
namespace spreadinterp {

// Subprob spreading kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void spread_subprob(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  T sigma   = p.sigma;
  T es_c    = p.es_c;
  T es_beta = p.es_beta;

  // assume that bin_size > ns/2;
  auto info         = compute_subprob_block_info<T, ndim>(p, blockIdx.x);
  auto &binsizes    = info.binsizes;
  auto &nbins       = info.nbins;
  auto &offset      = info.offset;
  const int ptstart = info.ptstart;
  const int nupts   = info.nupts;

  constexpr auto ns_2       = (ns + 1) / 2;
  constexpr auto rounded_ns = ns_2 * 2;

  int N = 1;
  for (int idim = 0; idim < ndim; ++idim) N *= binsizes[idim] + rounded_ns;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    fwshared[i] = {0, 0};
  }

  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx       = ptstart + i;
    const auto nuptsidx = loadReadOnly(p.idxnupts + idx);
    auto [ker, start]   = get_kerval_and_local_start<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, offset, sigma, es_c, es_beta);

    const auto cnow = loadReadOnly(c + nuptsidx);
    if constexpr (ndim == 1) {
      const auto ofs = start[0] + ns_2;
      for (int xx = 0; xx < ns; ++xx) {
        atomicAddComplexShared<T>(fwshared + xx + ofs, cnow * ker[0][xx]);
      }
    }
    if constexpr (ndim == 2) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto ofs0    = (start[1] + ns_2) * delta_y + start[0] + ns_2;
      for (int yy = 0; yy < ns; ++yy) {
        const auto ofs   = ofs0 + yy * delta_y;
        const auto cnowy = cnow * ker[1][yy];
        for (int xx = 0; xx < ns; ++xx) {
          atomicAddComplexShared<T>(fwshared + xx + ofs, cnowy * ker[0][xx]);
        }
      }
    }
    if constexpr (ndim == 3) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto delta_z = delta_y * (binsizes[1] + rounded_ns);
      const auto ofs0 =
          (start[2] + ns_2) * delta_z + (start[1] + ns_2) * delta_y + (start[0] + ns_2);
      for (int zz = 0; zz < ns; ++zz) {
        const auto cnowz = cnow * ker[2][zz];
        const auto ofs1  = ofs0 + zz * delta_z;
        for (int yy = 0; yy < ns; ++yy) {
          const auto cnowy = cnowz * ker[1][yy];
          const auto ofs   = ofs1 + yy * delta_y;
          for (int xx = 0; xx < ns; ++xx) {
            atomicAddComplexShared<T>(fwshared + xx + ofs, cnowy * ker[0][xx]);
          }
        }
      }
    }
  }

  __syncthreads();

  /* write to global memory */
  shared_mem_copy_helper<T, ndim, ns>(
      binsizes, offset, p.nf123, [fw, fwshared](int idx_shared, int idx_global) {
        atomicAddComplexGlobal<T>(fw + idx_global, fwshared[idx_shared]);
      });
}

// Subprob spreading CPU driver
template<typename T, int ndim, int ns>
void cuspread_subprob(const cufinufft_plan_t<T> &d_plan, const cuda_complex<T> *c,
                      cuda_complex<T> *fw, int blksize) {
  const auto sharedplanorysize = d_plan.shared_memory_required();

  const auto launch = [&](auto kernel) {
    cufinufft_set_shared_memory(kernel, d_plan);
    for (int t = 0; t < blksize; t++) {
      kernel<<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          d_plan, c + t * d_plan.M, fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(spread_subprob<T, 1, ndim, ns>)
                                     : launch(spread_subprob<T, 0, ndim, ns>);
}

template<typename T, int Ndim> struct SpreadSubprobCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    cuspread_subprob<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_spread_subprob(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                       cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadSubprobCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template<typename T, int Ndim> void do_indexSort_subprob_and_OD(cufinufft_plan_t<T> &p) {
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
          p.M, inv_binsizes, nbins, dethrust(p.binstartpts), dethrust(p.sortidx), p.kxyz,
          dethrust(p.idxnupts), p.nf123);
  THROW_IF_CUDA_ERROR

  /* --------------------------------------------- */
  //        Determining Subproblem properties      //
  /* --------------------------------------------- */
  calc_subprob<<<(p.M + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
      dethrust(p.binsize), dethrust(p.numsubprob), p.opts.gpu_maxsubprobsize, nbins_tot);
  THROW_IF_CUDA_ERROR

  thrust::inclusive_scan(thrust::cuda::par.on(p.stream), p.numsubprob.begin(),
                         p.numsubprob.begin() + nbins_tot, p.subprobstartpts.begin() + 1);
  THROW_IF_CUDA_ERROR

  int totalnumsubprob;
  checkCudaErrors(cudaMemsetAsync(dethrust(p.subprobstartpts), 0, sizeof(int), p.stream));
  checkCudaErrors(
      cudaMemcpyAsync(&totalnumsubprob, &(dethrust(p.subprobstartpts)[nbins_tot]),
                      sizeof(int), cudaMemcpyDeviceToHost, p.stream));
  cudaStreamSynchronize(p.stream);
  p.subprob_to_bin.resize(totalnumsubprob);

  map_b_into_subprob<<<(nbins[0] * nbins[1] + 1024 - 1) / 1024, 1024, 0, p.stream>>>(
      dethrust(p.subprob_to_bin), dethrust(p.subprobstartpts), dethrust(p.numsubprob),
      nbins_tot);

  p.totalnumsubprob = totalnumsubprob;
}

extern template void do_spread_subprob<float, 1>(const cufinufft_plan_t<float> &,
                                                 const cuda_complex<float> *,
                                                 cuda_complex<float> *, int);
extern template void do_spread_subprob<float, 2>(const cufinufft_plan_t<float> &,
                                                 const cuda_complex<float> *,
                                                 cuda_complex<float> *, int);
extern template void do_spread_subprob<float, 3>(const cufinufft_plan_t<float> &,
                                                 const cuda_complex<float> *,
                                                 cuda_complex<float> *, int);
extern template void do_spread_subprob<double, 1>(const cufinufft_plan_t<double> &,
                                                  const cuda_complex<double> *,
                                                  cuda_complex<double> *, int);
extern template void do_spread_subprob<double, 2>(const cufinufft_plan_t<double> &,
                                                  const cuda_complex<double> *,
                                                  cuda_complex<double> *, int);
extern template void do_spread_subprob<double, 3>(const cufinufft_plan_t<double> &,
                                                  const cuda_complex<double> *,
                                                  cuda_complex<double> *, int);

extern template void do_indexSort_subprob_and_OD<float, 1>(cufinufft_plan_t<float> &);
extern template void do_indexSort_subprob_and_OD<float, 2>(cufinufft_plan_t<float> &);
extern template void do_indexSort_subprob_and_OD<float, 3>(cufinufft_plan_t<float> &);
extern template void do_indexSort_subprob_and_OD<double, 1>(cufinufft_plan_t<double> &);
extern template void do_indexSort_subprob_and_OD<double, 2>(cufinufft_plan_t<double> &);
extern template void do_indexSort_subprob_and_OD<double, 3>(cufinufft_plan_t<double> &);

} // namespace spreadinterp
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::spread_subprob(const cuda_complex<T> *c, cuda_complex<T> *fw,
                                         int blksize) const {
  using cufinufft::spreadinterp::do_spread_subprob;
  switch (this->dim) {
  case 1:
    return do_spread_subprob<T, 1>(*this, c, fw, blksize);
  case 2:
    return do_spread_subprob<T, 2>(*this, c, fw, blksize);
  case 3:
    return do_spread_subprob<T, 3>(*this, c, fw, blksize);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

template<typename T> void cufinufft_plan_t<T>::indexSort_subprob_and_OD() {
  using cufinufft::spreadinterp::do_indexSort_subprob_and_OD;
  switch (this->dim) {
  case 1:
    return do_indexSort_subprob_and_OD<T, 1>(*this);
  case 2:
    return do_indexSort_subprob_and_OD<T, 2>(*this);
  case 3:
    return do_indexSort_subprob_and_OD<T, 3>(*this);
  default:
    throw int(FINUFFT_ERR_DIM_NOTVALID);
  }
}

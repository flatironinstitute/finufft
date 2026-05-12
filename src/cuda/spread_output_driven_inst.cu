// Per-dim instantiation TU: output-driven spread (gpu_method = 3).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.
// Prep is shared with gpu_method=2 and instantiated in spread_subprob_inst.cu.

#include "spreadinterp_common.cuh"
#include <cufinufft/spreadinterp.hpp>

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

// Output-driven spreading kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void spread_output_driven(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw, int np) {
  extern __shared__ char sharedbuf[];

  T sigma   = p.sigma;
  T es_c    = p.es_c;
  T es_beta = p.es_beta;

  // assume that bin_size > ns/2;
  auto info         = compute_subprob_block_info<T, ndim>(p, blockIdx.x);
  auto &binsizes    = info.binsizes;
  auto &offset      = info.offset;
  const int ptstart = info.ptstart;
  const int nupts   = info.nupts;

  static constexpr auto ns_2f = T(ns * .5);
  static constexpr auto ns_2  = (ns + 1) / 2;
  int total                   = 1;

  for (int idim = 0; idim < ndim; ++idim) total *= ns;

  auto [padded_size, local_subgrid_size] = get_padded_subgrid_info<ndim, ns>(binsizes);

  using kernel_data = cuda::std::array<cuda::std::array<T, ns>, ndim>;
  auto *kerevals    = reinterpret_cast<kernel_data *>(sharedbuf);
  // Offset pointer into sharedbuf after kerevals
  auto *nupts_sm =
      reinterpret_cast<cuda_complex<T> *>(sharedbuf + np * sizeof(kernel_data));
  auto *shift = reinterpret_cast<cuda::std::array<int, ndim> *>(
      sharedbuf + np * sizeof(kernel_data) + np * sizeof(cuda_complex<T>));

  auto *local_subgrid = reinterpret_cast<cuda_complex<T> *>(
      sharedbuf + np * sizeof(kernel_data) + np * sizeof(cuda_complex<T>) +
      np * sizeof(cuda::std::array<int, ndim>));

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid_size; i += blockDim.x) {
    local_subgrid[i] = {0, 0};
  }

  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx      = loadReadOnly(p.idxnupts + ptstart + i + batch_begin);
      nupts_sm[i]             = loadReadOnly(c + nuptsidx);
      auto [ker, local_shift] = get_kerval_and_local_start<T, KEREVALMETH, ndim, ns>(
          nuptsidx, p.xyz, p.nf123, offset, sigma, es_c, es_beta);
      kerevals[i] = ker;
      shift[i]    = local_shift;
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      const auto cnow  = nupts_sm[i];
      const auto start = shift[i];

      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // strength from shared memory
        int tmp       = idx;
        int idxout    = 0;
        T kervalue    = 1;
        int strideout = 1;
        for (int idim = 0; idim + 1 < ndim; ++idim) {
          int s = tmp % ns;
          kervalue *= kerevals[i][idim][s];
          idxout += strideout * (s + start[idim] + ns_2);
          strideout *= padded_size[idim];
          tmp /= ns;
        }
        // last dimension can be done more cheaply
        kervalue *= kerevals[i][ndim - 1][tmp];
        idxout += strideout * (tmp + start[ndim - 1] + ns_2);

        local_subgrid[idxout] += cnow * kervalue;
      }
      __syncthreads();
    }
  }
  /* write to global memory */
  for (int n = threadIdx.x; n < local_subgrid_size; n += blockDim.x) {
    const int outidx =
        output_index_from_flat_local_index<ndim, ns>(n, padded_size, offset, p.nf123);
    atomicAddComplexGlobal<T>(fw + outidx, local_subgrid[n]);
  }
}

// Output-driven spreading CPU driver
template<typename T, int ndim, int ns>
void spread_output_driven_launch(const cufinufft_plan_t<T> &d_plan,
                                 const cuda_complex<T> *c, cuda_complex<T> *fw,
                                 int blksize) {
  int bufsz = 1;
  for (int idim = 0; idim < ndim; ++idim) bufsz *= ns;

  const auto sharedplanorysize = d_plan.shared_memory_required();
  const int nthreads           = std::min(256, std::max(bufsz, d_plan.opts.gpu_np));

  const auto launch = [&](auto kernel) {
    cufinufft_set_shared_memory(kernel, d_plan);
    cudaFuncSetSharedMemConfig(kernel, cudaSharedMemBankSizeEightByte);
    THROW_IF_CUDA_ERROR
    for (int t = 0; t < blksize; t++) {
      kernel<<<d_plan.totalnumsubprob, std::min(256, std::max(bufsz, d_plan.opts.gpu_np)),
               sharedplanorysize, d_plan.stream>>>(
          d_plan, c + t * d_plan.M, fw + t * d_plan.nf, d_plan.opts.gpu_np);
      THROW_IF_CUDA_ERROR
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(spread_output_driven<T, 1, ndim, ns>)
                                     : launch(spread_output_driven<T, 0, ndim, ns>);
}

template<typename T, int Ndim> struct SpreadOutputDrivenCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    spread_output_driven_launch<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_spread_output_driven(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                             cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadOutputDrivenCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = make_range<MIN_NSPREAD, MAX_NSPREAD<T>>;
  dispatch(caller, std::make_tuple(DispatchParam<NsSeq>{p.spopts.nspread}));
}

template void do_spread_output_driven<float, CUFINUFFT_DIM>(
    const cufinufft_plan_t<float> &, const cuda_complex<float> *, cuda_complex<float> *,
    int);
template void do_spread_output_driven<double, CUFINUFFT_DIM>(
    const cufinufft_plan_t<double> &, const cuda_complex<double> *,
    cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft

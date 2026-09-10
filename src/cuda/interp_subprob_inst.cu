// Per-dim instantiation TU: subproblem interp (gpu_method = 2).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.

#include "spreadinterp_common.cuh"
#include <cufinufft/spreadinterp.hpp>

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

// Subprob interpolation kernel
template<typename T, int ndim, int ns>
__global__ FINUFFT_FLATTEN void interp_subprob(
    cufinufft_gpu_data<T> p, cuda_complex<T> *c, const cuda_complex<T> *fw) {
  extern __shared__ char sharedbuf[];
  auto fwshared = (cuda_complex<T> *)sharedbuf;

  // assume that bin_size > ns/2;
  auto info         = compute_subprob_block_info<T, ndim>(p, blockIdx.x);
  auto &binsizes    = info.binsizes;
  auto &offset      = info.offset;
  const int ptstart = info.ptstart;
  const int nupts   = info.nupts;

  constexpr auto ns_2       = (ns + 1) / 2;
  constexpr auto rounded_ns = ns_2 * 2;

  shared_mem_copy_helper<T, ndim, ns>(
      binsizes, offset, p.nf123, [fw, fwshared](int idx_shared, int idx_global) {
        fwshared[idx_shared] = loadReadOnly(fw + idx_global);
      });
  __syncthreads();

  for (int i = threadIdx.x; i < nupts; i += blockDim.x) {
    const int idx       = ptstart + i;
    const auto nuptsidx = loadReadOnly(p.idxnupts + idx);
    auto [ker, start] = get_kerval_and_local_start<T, ndim, ns>(nuptsidx, p.xyz, p.nf123,
                                                                offset, p.horner_coeffs);

    cuda_complex<T> cnow{0, 0};
    if constexpr (ndim == 1) {
      const auto ofs0 = start[0] + ns_2;
      for (int xx = 0; xx < ns; ++xx) cnow += fwshared[ofs0 + xx] * ker[0][xx];
    }
    if constexpr (ndim == 2) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto ofs0    = (start[1] + ns_2) * delta_y + (start[0] + ns_2);
      for (int yy = 0; yy < ns; ++yy) {
        cuda_complex<T> cnowy{0, 0};
        const auto ofs = ofs0 + yy * delta_y;
        for (int xx = 0; xx < ns; ++xx) {
          cnowy += fwshared[ofs + xx] * ker[0][xx];
        }
        cnow += cnowy * ker[1][yy];
      }
    }
    if constexpr (ndim == 3) {
      const auto delta_y = binsizes[0] + rounded_ns;
      const auto delta_z = delta_y * (binsizes[1] + rounded_ns);
      const auto ofs0 =
          (start[2] + ns_2) * delta_z + (start[1] + ns_2) * delta_y + (start[0] + ns_2);
      for (int zz = 0; zz < ns; ++zz) {
        cuda_complex<T> cnowz{0, 0};
        const auto ofs1 = ofs0 + zz * delta_z;
        for (int yy = 0; yy < ns; ++yy) {
          cuda_complex<T> cnowy{0, 0};
          const auto ofs = ofs1 + yy * delta_y;
          for (int xx = 0; xx < ns; ++xx) {
            cnowy += fwshared[ofs + xx] * ker[0][xx];
          }
          cnowz += cnowy * ker[1][yy];
        }
        cnow += cnowz * ker[2][zz];
      }
    }
    storeCacheStreaming(c + nuptsidx, cnow);
  }
}

// Subprob interpolation CPU driver
template<typename T, int ndim, int ns>
void interp_subprob_launch(const cufinufft_plan_t<T> &d_plan, cuda_complex<T> *c,
                           const cuda_complex<T> *fw, int blksize) {
  const auto sharedplanorysize = d_plan.shared_memory_required();

  const auto launch = [&](auto kernel) {
    cufinufft_set_shared_memory(kernel, d_plan);
    for (int t = 0; t < blksize; t++) {
      kernel<<<d_plan.totalnumsubprob, 256, sharedplanorysize, d_plan.stream>>>(
          d_plan, c + t * d_plan.M, fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR();
    }
  };
  launch(interp_subprob<T, ndim, ns>);
}

template<typename T, int Ndim> struct InterpSubprobCaller {
  const cufinufft_plan_t<T> &p;
  cuda_complex<T> *c;
  const cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    interp_subprob_launch<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_interp_subprob(const cufinufft_plan_t<T> &p, cuda_complex<T> *c,
                       const cuda_complex<T> *fw, int blksize) {
  InterpSubprobCaller<T, Ndim> caller{p, c, fw, blksize};
  utils::dispatch_kernel_shape<T>(caller, p.spopts.nspread);
}

template void do_interp_subprob<float, CUFINUFFT_DIM>(const cufinufft_plan_t<float> &,
                                                      cuda_complex<float> *,
                                                      const cuda_complex<float> *, int);
template void do_interp_subprob<double, CUFINUFFT_DIM>(const cufinufft_plan_t<double> &,
                                                       cuda_complex<double> *,
                                                       const cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft

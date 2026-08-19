#ifndef CUFINUFFT_HEURISTICS_HPP
#define CUFINUFFT_HEURISTICS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include <cufinufft/cufinufft_plan_t.hpp>
#include <cufinufft_opts.h>
#include <finufft_common/safe_call.h>
#include <finufft_errors.h>

namespace cufinufft::common {

// Per-launch shared-memory budget for spread/interp kernels. Wraps the
// account of grid tile + per-point scratch and is consulted by
// cufinufft_set_shared_memory() and the bin-size selector.
template<typename T>
std::size_t shared_memory_required(int dim, int ns, int bin_size_x, int bin_size_y,
                                   int bin_size_z, int np);

// Pick (bin sizes, np) for the requested gpu_method on the device named in
// opts->gpu_device_id. Mutates opts in place. Mirrors finufft::heuristics on
// the CPU side.
template<typename T>
void cufinufft_setup_binsize(const GpuCapabilities &gpu, int type, int ns, int dim,
                             cufinufft_opts *opts);

// How many transforms to batch into one FFT / spread pass, in [1, ntransf]. Needs the
// fine grid, so it is called once nf is known.
// opts.gpu_maxbatchsize == 0 means auto; > 0 is honored (capped at ntransf).
template<typename T>
int choose_batchsize(const GpuCapabilities &gpu, const cufinufft_opts &opts, int ntransf,
                     std::int64_t nf);

// Opt this kernel into the dynamic shared memory the plan needs, throwing
// FINUFFT_ERR_INSUFFICIENT_SHMEM if the device cannot satisfy the request.
// WARNING: does not handle CUDA errors; the caller must check them.
template<typename T, typename V>
void cufinufft_set_shared_memory(V *kernel, const cufinufft_plan_t<T> &d_plan) {
  const auto shared_mem_required = d_plan.shared_memory_required();
  const int shared_mem_per_block = d_plan.gpu.max_smem_per_block_optin;
  if (shared_mem_required > unsigned(shared_mem_per_block)) {
    fprintf(stderr,
            "Error: Shared memory required per block is %zu bytes, but the device "
            "supports only %d bytes.\n",
            shared_mem_required, shared_mem_per_block);
    throw finufft::exception(FINUFFT_ERR_INSUFFICIENT_SHMEM);
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shared_mem_required);
}

} // namespace cufinufft::common

#endif

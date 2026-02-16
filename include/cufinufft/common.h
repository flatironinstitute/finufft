#ifndef __COMMON_H__
#define __COMMON_H__

#include <cufft.h>
#include <cufinufft/types.h>
#include <cufinufft_opts.h>
#include <finufft_common/spread_opts.h>
#include <finufft_errors.h>

#include <complex>

namespace cufinufft {
namespace common {
template<typename T>
int fseries_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, T *d_f, T *d_phase,
                           cuda::std::array<T *,3> d_fwkerhalf, int ns,
                           cudaStream_t stream);
template<typename T>
int nuft_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, T *d_f, T *d_z,
                        cuda::std::array<T *,3> d_kxyz, cuda::std::array<T *,3>d_fwkerhalf,
                        int ns, cudaStream_t stream);
template<typename T>
int setup_spreader_for_nufft(finufft_spread_opts &spopts, T eps, cufinufft_opts opts);

void set_nf_type12(CUFINUFFT_BIGINT ms, cufinufft_opts opts, finufft_spread_opts spopts,
                   CUFINUFFT_BIGINT *nf, CUFINUFFT_BIGINT b);

template<typename T>
void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, T *f, T *a,
                                   finufft_spread_opts opts);
template<typename T>
void onedim_nuft_kernel_precomp(T *f, T *zout, finufft_spread_opts opts);

template<typename T>
std::size_t shared_memory_required(int dim, int ns, int bin_size_x, int bin_size_y,
                                   int bin_size_z, int np);

template<typename T>
void cufinufft_setup_binsize(int type, int ns, int dim, cufinufft_opts *opts);

template<typename T, typename V>
int cufinufft_set_shared_memory(V *kernel, const int dim,
                                const cufinufft_plan_t<T> &d_plan) {
  /**
   * WARNING: this function does not handle cuda errors. The caller should check them.
   */
  int shared_mem_per_block{};
  const auto shared_mem_required = shared_memory_required<T>(
      dim, d_plan.spopts.nspread, d_plan.opts.gpu_binsizex, d_plan.opts.gpu_binsizey,
      d_plan.opts.gpu_binsizez, d_plan.opts.gpu_np);
  cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         d_plan.opts.gpu_device_id);
  if (shared_mem_required > shared_mem_per_block) {
    fprintf(stderr,
            "Error: Shared memory required per block is %zu bytes, but the device "
            "supports only %d bytes.\n",
            shared_mem_required, shared_mem_per_block);
    return FINUFFT_ERR_INSUFFICIENT_SHMEM;
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shared_mem_required);
  return 0;
}

} // namespace common
} // namespace cufinufft
#endif

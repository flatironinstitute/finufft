#ifndef CUFINUFFT_FFT_HPP
#define CUFINUFFT_FFT_HPP

#include <cuda/std/array>
#include <cuda_runtime.h>

#include <cufinufft/cufinufft_plan_t.hpp>
#include <cufinufft/types.hpp>
#include <finufft_common/spread_opts.h>

namespace cufinufft::common {

// Compute Fourier-series approximations of the spreading kernel on the GPU.
// Equispaced freq case (type 1/2). Mirrors onedim_fseries_kernel on CPU.
template<typename T>
void fseries_kernel_compute(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const T *d_f, const T *d_phase,
    cuda::std::array<gpu_array<T>, 3> &d_fwkerhalf, int ns, cudaStream_t stream);

// Same idea for the arbitrary-frequency case (type 3). Mirrors KernelFSeries
// (a.k.a. onedim_nuft_kernel) on CPU.
template<typename T>
void nuft_kernel_compute(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const T *d_f, const T *d_z,
    cuda::std::array<const T *, 3> d_kxyz, cuda::std::array<gpu_array<T>, 3> &d_fwkerhalf,
    int ns, cudaStream_t stream);

// Host-side Gauss-Legendre precompute that supplies (z, f) to nuft_kernel_compute.
// Implements the first half of CPU's onedim_nuft_kernel.
template<typename T>
void onedim_nuft_kernel_precomp(T *f, T *zout, finufft_spread_opts opts);

} // namespace cufinufft::common

#endif

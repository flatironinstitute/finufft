// Fourier-series / NUFT-kernel compute. F-series for type 1/2 (equispaced
// frequencies) and NUFT for type 3 (arbitrary frequencies). Hosts the
// __global__ kernels, their host wrappers, and the plan helpers
// (set_nf_type12, precompute_fseries_nodes) that drive them. Mirrors CPU
// src/fft.cpp.

#include <algorithm>

#include <cuComplex.h>
#include <cuda.h>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/fft.hpp>
#include <cufinufft/spreadinterp.hpp>
#include <cufinufft/utils.hpp>

namespace cufinufft {
namespace common {
using namespace cufinufft::spreadinterp;
using namespace finufft::common;
using std::max;

/** Kernel for computing approximations of exact Fourier series coeffs of
 *  cnufftspread's real symmetric kernel.
 * phase, f are intermediate results from function precompute_fseries_nodes().
 * this is the equispaced frequency case, used by type 1 & 2, matching
 * onedim_fseries_kernel in CPU code. Used by functions below in this file.
 */
template<typename T>
static __global__ void cu_fseries_kernel_compute(
    cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const T *f, const T *phase,
    cuda::std::array<T *, 3> fwkerhalf, int ns) {
  T J2            = ns / 2.0;
  int q           = (int)(2 + 3.0 * J2);
  int nf          = nf123[threadIdx.y];
  const T *phaset = phase + threadIdx.y * MAX_NQUAD;
  const T *ft     = f + threadIdx.y * MAX_NQUAD;
  T *oarr         = fwkerhalf[threadIdx.y];

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nf / 2 + 1;
       i += blockDim.x * gridDim.x) {
    T x = 0.0;
    for (int n = 0; n < q; n++) {
      // in type 1/2 2*PI/nf -> k[i]
      x += ft[n] * T(2) * std::cos(T(i) * phaset[n]);
    }
    oarr[i] = x * T(i % 2 ? -1 : 1); // signflip for the kernel origin being at PI
  }
}

/** Kernel for computing approximations of exact Fourier series coeffs of
 *  cnufftspread's real symmetric kernel.
 * a , f are intermediate results from function precompute_fseries_nodes().
 * this is the arbitrary frequency case (hence the extra kx, ky, kx arguments), used by
 * type 3, matching KernelFSeries in CPU code. Used by functions below in this file.
 */
template<typename T>
static __global__ void cu_nuft_kernel_compute(
    cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const T *f, const T *z,
    cuda::std::array<const T *, 3> kxyz, cuda::std::array<T *, 3> fwkerhalf, int ns) {
  T J2        = ns / 2.0;
  int q       = (int)(2 + 2.0 * J2);
  int nf      = nf123[threadIdx.y];
  const T *at = z + threadIdx.y * MAX_NQUAD;
  const T *ft = f + threadIdx.y * MAX_NQUAD;
  T *oarr     = fwkerhalf[threadIdx.y];
  const T *k  = kxyz[threadIdx.y];
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nf;
       i += blockDim.x * gridDim.x) {
    T x = 0.0;
    for (int n = 0; n < q; n++) {
      x += ft[n] * T(2) * std::cos(k[i] * at[n]);
    }
    oarr[i] = x;
  }
}

template<typename T>
void fseries_kernel_compute(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const T *d_f, const T *d_phase,
    cuda::std::array<gpu_array<T>, 3> &d_fwkerhalf, int ns, cudaStream_t stream)
/*
    wrapper for approximation of Fourier series of real symmetric spreading
    kernel.

    Melody Shih 2/20/22
*/
{
  int nout = max(max(nf123[0] / 2 + 1, nf123[1] / 2 + 1), nf123[2] / 2 + 1);

  dim3 threadsPerBlock(16, dim);
  dim3 numBlocks((nout + 16 - 1) / 16, 1);

  cu_fseries_kernel_compute<<<numBlocks, threadsPerBlock, 0, stream>>>(
      nf123, d_f, d_phase,
      {dethrust(d_fwkerhalf[0]), dethrust(d_fwkerhalf[1]), dethrust(d_fwkerhalf[2])}, ns);
  THROW_IF_CUDA_ERROR
}
template void fseries_kernel_compute<float>(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const float *d_f,
    const float *d_phase, cuda::std::array<gpu_array<float>, 3> &d_fwkerhalf, int ns,
    cudaStream_t stream);
template void fseries_kernel_compute<double>(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const double *d_f,
    const double *d_phase, cuda::std::array<gpu_array<double>, 3> &d_fwkerhalf, int ns,
    cudaStream_t stream);

template<typename T>
void nuft_kernel_compute(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const T *d_f, const T *d_z,
    cuda::std::array<const T *, 3> d_kxyz, cuda::std::array<gpu_array<T>, 3> &d_fwkerhalf,
    int ns, cudaStream_t stream)
/*
    Approximates exact Fourier transform of cnufftspread's real symmetric
    kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
    narrowness of kernel. Evaluates at set of arbitrary freqs k in [-pi, pi),
    for a kernel with x measured in grid-spacings. (See previous routine for
    FT definition).
    It implements onedim_nuft_kernel in CPU code. Except it combines up to three
    onedimensional kernel evaluations at once (for efficiency).

    Marco Barbone 08/28/2024
*/
{
  int nout = max(max(nf123[0], nf123[1]), nf123[2]);

  dim3 threadsPerBlock(16, dim);
  dim3 numBlocks((nout + 16 - 1) / 16, 1);

  cu_nuft_kernel_compute<<<numBlocks, threadsPerBlock, 0, stream>>>(
      nf123, d_f, d_z, d_kxyz,
      {dethrust(d_fwkerhalf[0]), dethrust(d_fwkerhalf[1]), dethrust(d_fwkerhalf[2])}, ns);
  THROW_IF_CUDA_ERROR
}
template void nuft_kernel_compute(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const float *d_f,
    const float *d_z, cuda::std::array<const float *, 3> d_kxyz,
    cuda::std::array<gpu_array<float>, 3> &d_fwkerhalf, int ns, cudaStream_t stream);
template void nuft_kernel_compute(
    int dim, cuda::std::array<CUFINUFFT_BIGINT, 3> nf123, const double *d_f,
    const double *d_z, cuda::std::array<const double *, 3> d_kxyz,
    cuda::std::array<gpu_array<double>, 3> &d_fwkerhalf, int ns, cudaStream_t stream);

template<typename T>
void onedim_nuft_kernel_precomp(T *f, T *z, finufft_spread_opts opts) {
  // it implements the first half of onedim_nuft_kernel in CPU code
  T J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 2.0 * J2); // matches CPU code
  double z_local[2 * MAX_NQUAD];
  double w_local[2 * MAX_NQUAD];
  gaussquad(2 * q, z_local, w_local);                     // half the nodes, (0,1)
  for (int n = 0; n < q; ++n) {                           // set up nodes z_n and vals f_n
    z[n] = J2 * T(z_local[n]);                            // rescale nodes
    f[n] = J2 * w_local[n] * evaluate_kernel(z[n], opts); // vals & quadr wei
  }
}
template void onedim_nuft_kernel_precomp<float>(float *f, float *a,
                                                finufft_spread_opts opts);
template void onedim_nuft_kernel_precomp<double>(double *f, double *a,
                                                 finufft_spread_opts opts);

} // namespace common
} // namespace cufinufft

template<typename T>
void cufinufft_plan_t<T>::set_nf_type12(CUFINUFFT_BIGINT ms, CUFINUFFT_BIGINT *nf,
                                        CUFINUFFT_BIGINT bs) const
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
  // round up to handle small cases
  *nf = static_cast<CUFINUFFT_BIGINT>(std::ceil(opts.upsampfac * ms));
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread; // otherwise spread fails
  if (*nf < MAX_NF) {                                     // otherwise will fail anyway
    if (bs & 1) bs *= 2; // make sure that bs is even
    *nf = finufft::common::next235(*nf, opts.gpu_method == 4 ? bs : 2);
  }
}
template void cufinufft_plan_t<float>::set_nf_type12(CUFINUFFT_BIGINT, CUFINUFFT_BIGINT *,
                                                     CUFINUFFT_BIGINT) const;
template void cufinufft_plan_t<double>::set_nf_type12(
    CUFINUFFT_BIGINT, CUFINUFFT_BIGINT *, CUFINUFFT_BIGINT) const;

template<typename T>
void cufinufft_plan_t<T>::precompute_fseries_nodes(CUFINUFFT_BIGINT nf_, T *f,
                                                   T *phase) const {
  using cufinufft::spreadinterp::evaluate_kernel;
  using finufft::common::gaussquad;
  using finufft::common::MAX_NQUAD;
  using finufft::common::PI;
  T J2         = spopts.nspread / 2.0; // J/2, half-width of ker z-support
  const auto q = (int)(2 + 3.0 * J2);  // matches CPU code
  double z[2 * MAX_NQUAD];
  double w[2 * MAX_NQUAD];
  gaussquad(2 * q, z, w);       // only half the nodes used, for (0,1)
  for (int n = 0; n < q; ++n) { // set up nodes z_n and vals f_n
    z[n] *= J2;                 // rescale nodes
    f[n]     = J2 * w[n] * evaluate_kernel((T)z[n], spopts); // vals & quadr wei
    phase[n] = T(2.0 * PI * z[n] / T(nf_));                  // phase winding rates
  }
}
template void cufinufft_plan_t<float>::precompute_fseries_nodes(CUFINUFFT_BIGINT, float *,
                                                                float *) const;
template void cufinufft_plan_t<double>::precompute_fseries_nodes(
    CUFINUFFT_BIGINT, double *, double *) const;

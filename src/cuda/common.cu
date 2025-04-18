#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cuComplex.h>
#include <cuda.h>

#include <cufinufft/common.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/defs.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

#include <legendre_rule_fast.h>

namespace cufinufft {
namespace common {
using namespace cufinufft::spreadinterp;
using std::max;

/** Kernel for computing approximations of exact Fourier series coeffs of
 *  cnufftspread's real symmetric kernel.
 * phase, f are intermediate results from function onedim_fseries_kernel_precomp()
 * (see cufinufft/contrib/common.cpp for description)
 * this is the equispaced frequency case, used by type 1 & 2, matching
 * onedim_fseries_kernel in CPU code
 */
template<typename T>
__global__ void cu_fseries_kernel_compute(int nf1, int nf2, int nf3, T *f, T *phase,
                                          T *fwkerhalf1, T *fwkerhalf2, T *fwkerhalf3,
                                          int ns) {
  T J2  = ns / 2.0;
  int q = (int)(2 + 3.0 * J2);
  int nf;
  T *phaset = phase + threadIdx.y * MAX_NQUAD;
  T *ft     = f + threadIdx.y * MAX_NQUAD;
  T *oarr;
  // standard parallelism pattern in cuda. using a 2D grid, this allows to leverage more
  // threads as the parallelism is x*y*z
  // each thread check the y index to determine which array to use
  if (threadIdx.y == 0) {
    oarr = fwkerhalf1;
    nf   = nf1;
  } else if (threadIdx.y == 1) {
    oarr = fwkerhalf2;
    nf   = nf2;
  } else {
    oarr = fwkerhalf3;
    nf   = nf3;
  }

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
 * a , f are intermediate results from function onedim_fseries_kernel_precomp()
 * (see cufinufft/contrib/common.cpp for description)
 * this is the arbitrary frequency case (hence the extra kx, ky, kx arguments), used by
 * type 3, matching onedim_nuft_kernel in CPU code
 */
template<typename T>
__global__ void cu_nuft_kernel_compute(int nf1, int nf2, int nf3, T *f, T *z, T *kx,
                                       T *ky, T *kz, T *fwkerhalf1, T *fwkerhalf2,
                                       T *fwkerhalf3, int ns) {
  T J2  = ns / 2.0;
  int q = (int)(2 + 2.0 * J2);
  int nf;
  T *at = z + threadIdx.y * MAX_NQUAD;
  T *ft = f + threadIdx.y * MAX_NQUAD;
  T *oarr, *k;
  // standard parallelism pattern in cuda. using a 2D grid, this allows to leverage more
  // threads as the parallelism is x*y*z
  // each thread check the y index to determine which array to use
  if (threadIdx.y == 0) {
    k    = kx;
    oarr = fwkerhalf1;
    nf   = nf1;
  } else if (threadIdx.y == 1) {
    k    = ky;
    oarr = fwkerhalf2;
    nf   = nf2;
  } else {
    k    = kz;
    oarr = fwkerhalf3;
    nf   = nf3;
  }
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
int fseries_kernel_compute(int dim, int nf1, int nf2, int nf3, T *d_f, T *d_phase,
                           T *d_fwkerhalf1, T *d_fwkerhalf2, T *d_fwkerhalf3, int ns,
                           cudaStream_t stream)
/*
    wrapper for approximation of Fourier series of real symmetric spreading
    kernel.

    Melody Shih 2/20/22
*/
{
  int nout = max(max(nf1 / 2 + 1, nf2 / 2 + 1), nf3 / 2 + 1);

  dim3 threadsPerBlock(16, dim);
  dim3 numBlocks((nout + 16 - 1) / 16, 1);

  cu_fseries_kernel_compute<<<numBlocks, threadsPerBlock, 0, stream>>>(
      nf1, nf2, nf3, d_f, d_phase, d_fwkerhalf1, d_fwkerhalf2, d_fwkerhalf3, ns);
  RETURN_IF_CUDA_ERROR

  return 0;
}

template<typename T>
int nuft_kernel_compute(int dim, int nf1, int nf2, int nf3, T *d_f, T *d_z, T *d_kx,
                        T *d_ky, T *d_kz, T *d_fwkerhalf1, T *d_fwkerhalf2,
                        T *d_fwkerhalf3, int ns, cudaStream_t stream)
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
  int nout = max(max(nf1, nf2), nf3);

  dim3 threadsPerBlock(16, dim);
  dim3 numBlocks((nout + 16 - 1) / 16, 1);

  cu_nuft_kernel_compute<<<numBlocks, threadsPerBlock, 0, stream>>>(
      nf1, nf2, nf3, d_f, d_z, d_kx, d_ky, d_kz, d_fwkerhalf1, d_fwkerhalf2, d_fwkerhalf3,
      ns);
  RETURN_IF_CUDA_ERROR

  return 0;
}

template<typename T>
int setup_spreader_for_nufft(finufft_spread_opts &spopts, T eps, cufinufft_opts opts)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Report status of setup_spreader. Just a wrapper following the CPU code.
{
  int ier = setup_spreader(spopts, eps, (T)opts.upsampfac, opts.gpu_kerevalmeth,
                           opts.debug, opts.gpu_spreadinterponly);
  return ier;
}

void set_nf_type12(CUFINUFFT_BIGINT ms, cufinufft_opts opts, finufft_spread_opts spopts,
                   CUFINUFFT_BIGINT *nf, CUFINUFFT_BIGINT bs)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
  *nf = (CUFINUFFT_BIGINT)(opts.upsampfac * ms);
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread; // otherwise spread fails
  if (*nf < MAX_NF) {                                     // otherwise will fail anyway
    if (opts.gpu_method == 4)                             // expensive at huge nf
      *nf = utils::next235beven(*nf, bs);
    else
      *nf = utils::next235beven(*nf, 1);
  }
}

/*
  Precomputation of approximations of exact Fourier series coeffs of cnufftspread's
  real symmetric kernel.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)
  phase_winding - if true (type 1-2), scaling for the equispaced case else (type 3)
                  scaling for the general kx,ky,kz case

  Outputs:
  a - vector of phases to be used for cosines on the GPU;
  f - function values at quadrature nodes multiplied with quadrature weights (a, f are
      provided as the inputs of onedim_fseries_kernel_compute() defined below)
*/

template<typename T>
void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, T *f, T *phase,
                                   finufft_spread_opts opts) {
  T J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  const auto q = (int)(2 + 3.0 * J2); // matches CPU code
  double z[2 * MAX_NQUAD];
  double w[2 * MAX_NQUAD];
  finufft::quadrature::legendre_compute_glr(2 * q, z, w); // only half the nodes used,
  // eg on (0,1)
  for (int n = 0; n < q; ++n) { // set up nodes z_n and vals f_n
    z[n] *= J2;                 // rescale nodes
    f[n]     = J2 * w[n] * evaluate_kernel((T)z[n], opts); // vals & quadr wei
    phase[n] = T(2.0 * M_PI * z[n] / T(nf));               // phase winding rates
  }
}

template<typename T>
void onedim_nuft_kernel_precomp(T *f, T *z, finufft_spread_opts opts) {
  // it implements the first half of onedim_nuft_kernel in CPU code
  T J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
  // # quadr nodes in z (from 0 to J/2; reflections will be added)...
  int q = (int)(2 + 2.0 * J2); // matches CPU code
  double z_local[2 * MAX_NQUAD];
  double w_local[2 * MAX_NQUAD];
  finufft::quadrature::legendre_compute_glr(2 * q, z_local, w_local); // only half the
                                                                      // nodes used, eg on
                                                                      // (0,1)
  for (int n = 0; n < q; ++n) {                           // set up nodes z_n and vals f_n
    z[n] = J2 * T(z_local[n]);                            // rescale nodes
    f[n] = J2 * w_local[n] * evaluate_kernel(z[n], opts); // vals & quadr wei
  }
}

template<typename T> std::size_t shared_memory_per_point(int dim, int ns) {
  return ns * sizeof(T) * dim       // kernel evaluations
         + sizeof(int) * dim        // indexes
         + sizeof(cuda_complex<T>); // strength
}

// Marco: 4/18/25 not 100% happy of having np here, but the alternatives seem worse to me
template<typename T>
std::size_t shared_memory_required(int dim, int ns, int bin_size_x, int bin_size_y,
                                   int bin_size_z, int np) {
  const auto shmem_per_point = shared_memory_per_point<T>(dim, ns);
  const int ns_2             = (ns + 1) / 2;
  std::size_t grid_size      = bin_size_x + 2 * ns_2;
  if (dim >= 2) grid_size *= bin_size_y + 2 * ns_2;
  if (dim == 3) grid_size *= bin_size_z + 2 * ns_2;
  return grid_size * sizeof(cuda_complex<T>) + shmem_per_point * np;
}

// Function to find bin_size_x == bin_size_y
// where bin_size_x * bin_size_y * bin_size_z < mem_size
template<typename T> int find_bin_size(std::size_t mem_size, int dim, int ns) {
  const auto elements        = float(mem_size) / sizeof(cuda_complex<T>);
  const auto padded_bin_size = std::floor(std::pow(elements, 1.0 / dim));
  const auto bin_size        = static_cast<int>(padded_bin_size) - ns - ns % 2;
  // TODO: over one dimension we could increase this a bit
  //       maybe the shape should not be uniform
  return bin_size - 1;
}

template<typename T>
void cufinufft_setup_binsize(int type, int ns, int dim, cufinufft_opts *opts) {
  // Marco Barbone 07/26/24. Using the shared memory available on the device, to
  // determine the optimal binsize for the spreader.
  // WARNING: This function does not check for CUDA errors, the caller should check and
  // handle them.
  int shared_mem_per_block{}, device_id{};
  cudaGetDevice(&device_id);
  cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                         device_id);
  auto binsize = find_bin_size<T>(shared_mem_per_block, dim, ns);
  switch (opts->gpu_method) {
  case 1:
    cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device_id);
    binsize = find_bin_size<T>(shared_mem_per_block, dim, ns);
  case 2:
  case 0: {
    opts->gpu_binsizex = opts->gpu_binsizex == 0 ? binsize : opts->gpu_binsizex;
    opts->gpu_binsizey = opts->gpu_binsizey == 0 ? binsize : opts->gpu_binsizey;
    opts->gpu_binsizez = opts->gpu_binsizez == 0 ? binsize : opts->gpu_binsizez;
    opts->gpu_binsizey = dim > 1 ? opts->gpu_binsizey : 1;
    opts->gpu_binsizez = dim > 2 ? opts->gpu_binsizez : 1;
    break;
  }
  case 3: {
    // opts->gpu_np this way is at least 16.
    const auto shmem_per_point = shared_memory_per_point<T>(dim, ns);
    const auto min_np_shmem    = shared_memory_per_point<T>(dim, ns) * opts->gpu_np;
    binsize = find_bin_size<T>(shared_mem_per_block - min_np_shmem, dim, ns);
    const auto shmem_required =
        shared_memory_required<T>(dim, ns, binsize, binsize, binsize, 0);
    const auto shmem_left = shared_mem_per_block - shmem_required;
    const auto max_np     = ((shmem_left) / shmem_per_point) & -16;
    if (opts->debug) {
      const int required_shmem =
          shared_memory_required<T>(dim, ns, binsize, binsize, binsize, max_np);
      printf("[cufinufft] Shared memory required: %d bytes (limit: %d bytes)\n",
             required_shmem, shared_mem_per_block);

      printf("[cufinufft]   min_np_shmem     = %d\n", min_np_shmem);
      printf("[cufinufft]   shmem_per_point  = %d\n", shmem_per_point);
      printf("[cufinufft]   shmem_required   = %d\n", shmem_required);
      printf("[cufinufft]   shmem_left       = %d\n", shmem_left);
      printf("[cufinufft]   max_np           = %d\n", max_np);
      printf("[cufinufft]   binsize          = %d\n", binsize);

      if (required_shmem > shared_mem_per_block) {
        throw std::runtime_error("[cufinufft] ERROR: Requested too much shared memory.");
      }
    }

    opts->gpu_np       = max_np;
    opts->gpu_binsizex = binsize;
    opts->gpu_binsizey = binsize;
    opts->gpu_binsizez = binsize;
  } break;
  case 4: {
    opts->gpu_obinsizex = (opts->gpu_obinsizex == 0) ? 8 : opts->gpu_obinsizex;
    opts->gpu_obinsizey = (opts->gpu_obinsizey == 0) ? 8 : opts->gpu_obinsizey;
    opts->gpu_obinsizez = (opts->gpu_obinsizez == 0) ? 8 : opts->gpu_obinsizez;
    opts->gpu_binsizex  = (opts->gpu_binsizex == 0) ? 4 : opts->gpu_binsizex;
    opts->gpu_binsizey  = (opts->gpu_binsizey == 0) ? 4 : opts->gpu_binsizey;
    opts->gpu_binsizez  = (opts->gpu_binsizez == 0) ? 4 : opts->gpu_binsizez;
    break;
  }
  }
}

template int setup_spreader_for_nufft(finufft_spread_opts &spopts, float eps,
                                      cufinufft_opts opts);
template int setup_spreader_for_nufft(finufft_spread_opts &spopts, double eps,
                                      cufinufft_opts opts);
template void onedim_fseries_kernel_precomp<float>(CUFINUFFT_BIGINT nf, float *f,
                                                   float *a, finufft_spread_opts opts);
template void onedim_fseries_kernel_precomp<double>(CUFINUFFT_BIGINT nf, double *f,
                                                    double *a, finufft_spread_opts opts);
template void onedim_nuft_kernel_precomp<float>(float *f, float *a,
                                                finufft_spread_opts opts);
template void onedim_nuft_kernel_precomp<double>(double *f, double *a,
                                                 finufft_spread_opts opts);
template int fseries_kernel_compute(int dim, int nf1, int nf2, int nf3, float *d_f,
                                    float *d_a, float *d_fwkerhalf1, float *d_fwkerhalf2,
                                    float *d_fwkerhalf3, int ns, cudaStream_t stream);
template int fseries_kernel_compute(
    int dim, int nf1, int nf2, int nf3, double *d_f, double *d_a, double *d_fwkerhalf1,
    double *d_fwkerhalf2, double *d_fwkerhalf3, int ns, cudaStream_t stream);
template int nuft_kernel_compute<float>(int dim, int nf1, int nf2, int nf3, float *d_f,
                                        float *d_a, float *d_kx, float *d_ky, float *d_kz,
                                        float *d_fwkerhalf1, float *d_fwkerhalf2,
                                        float *d_fwkerhalf3, int ns, cudaStream_t stream);
template int nuft_kernel_compute<double>(
    int dim, int nf1, int nf2, int nf3, double *d_f, double *d_a, double *d_kx,
    double *d_ky, double *d_kz, double *d_fwkerhalf1, double *d_fwkerhalf2,
    double *d_fwkerhalf3, int ns, cudaStream_t stream);

template std::size_t shared_memory_required<float>(
    int dim, int ns, int bin_size_x, int bin_size_y, int bin_size_z, int np);
template std::size_t shared_memory_required<double>(
    int dim, int ns, int bin_size_x, int bin_size_y, int bin_size_z, int np);
template std::size_t shared_memory_per_point<float>(int dim, int ns);
template std::size_t shared_memory_per_point<double>(int dim, int ns);
template void cufinufft_setup_binsize<float>(int type, int ns, int dim,
                                             cufinufft_opts *opts);
template void cufinufft_setup_binsize<double>(int type, int ns, int dim,
                                              cufinufft_opts *opts);
} // namespace common
} // namespace cufinufft

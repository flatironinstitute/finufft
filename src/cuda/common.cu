#include <algorithm>
#include <iomanip>
#include <iostream>

#include <cuComplex.h>
#include <cuda.h>

#include <cufinufft/common.h>
#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/defs.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

namespace cufinufft {
namespace common {
using namespace cufinufft::spreadinterp;
using namespace finufft::common;
using std::max;

/** Kernel for computing approximations of exact Fourier series coeffs of
 *  cnufftspread's real symmetric kernel.
 * phase, f are intermediate results from function onedim_fseries_kernel_precomp().
 * this is the equispaced frequency case, used by type 1 & 2, matching
 * onedim_fseries_kernel in CPU code. Used by functions below in this file.
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
 * a , f are intermediate results from function onedim_fseries_kernel_precomp().
 * this is the arbitrary frequency case (hence the extra kx, ky, kx arguments), used by
 * type 3, matching KernelFSeries in CPU code. Used by functions below in this file.
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
  // round up to handle small cases
  *nf = static_cast<CUFINUFFT_BIGINT>(std::ceil(opts.upsampfac * ms));
  if (*nf < 2 * spopts.nspread) *nf = 2 * spopts.nspread; // otherwise spread fails
  if (*nf < MAX_NF) {                                     // otherwise will fail anyway
    *nf = utils::next235beven(*nf, opts.gpu_method == 4 ? bs : 1); // expensive at huge nf
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
  gaussquad(2 * q, z, w);       // only half the nodes used, for (0,1)
  for (int n = 0; n < q; ++n) { // set up nodes z_n and vals f_n
    z[n] *= J2;                 // rescale nodes
    f[n]     = J2 * w[n] * evaluate_kernel((T)z[n], opts); // vals & quadr wei
    phase[n] = T(2.0 * PI * z[n] / T(nf));                 // phase winding rates
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
  gaussquad(2 * q, z_local, w_local);                     // half the nodes, (0,1)
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
  if (dim > 1) grid_size *= bin_size_y + 2 * ns_2;
  if (dim > 2) grid_size *= bin_size_z + 2 * ns_2;
  return grid_size * sizeof(cuda_complex<T>) + shmem_per_point * np;
}

// Function to find bin_size_x == bin_size_y
// where bin_size_x * bin_size_y * bin_size_z < mem_size
template<typename T> int find_bin_size(std::size_t mem_size, int dim, int ns) {
  const auto elements        = mem_size / sizeof(cuda_complex<T>);
  const auto padded_bin_size = int(std::floor(std::pow(elements, 1.0 / dim)));
  const auto bin_size        = padded_bin_size - (2 * (ns + 1) / 2);
  // TODO: over one dimension we could increase this a bit
  //       maybe the shape should not be uniform
  return bin_size;
}
template<typename T>
void cufinufft_setup_binsize(int type, int ns, int dim, cufinufft_opts *opts) {
  int shared_mem_per_block{}, device_id{opts->gpu_device_id};
  cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device_id);

  // Helper: Calculate bin size from available shared memory
  auto calc_binsize = [&](int shmem) -> int {
    return find_bin_size<T>(shmem, dim, ns);
  };

  // Helper: Set bin sizes respecting dimensionality
  auto set_binsizes = [&](int binsize_x, int binsize_y, int binsize_z) {
    opts->gpu_binsizex = binsize_x;
    opts->gpu_binsizey = (dim >= 2) ? binsize_y : 1;
    opts->gpu_binsizez = (dim >= 3) ? binsize_z : 1;
  };

  // Helper: Set bin sizes only if user hasn't specified them
  auto set_binsizes_if_unset = [&](int binsize) {
    int x = (opts->gpu_binsizex == 0) ? binsize : opts->gpu_binsizex;
    int y = (opts->gpu_binsizey == 0) ? binsize : opts->gpu_binsizey;
    int z = (opts->gpu_binsizez == 0) ? binsize : opts->gpu_binsizez;
    set_binsizes(x, y, z);
  };

  auto print_method_setup = [&](int method, int ns, int np, const char *suffix) {
    if (opts->debug < 1) return;
    if (method == 3) {
      printf("[cufinufft] Method 3 setup: dim=%d, ns=%d, bin=%dx%dx%d, np=%d%s\n",
             dim, ns, opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, np,
             suffix);
    } else {
      printf("[cufinufft] Method %d setup: dim=%d, ns=%d, bin=%dx%dx%d%s\n", method, dim,
             ns, opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, suffix);
    }
  };

  auto print_shmem_usage = [&](const char *label, int shmem_used, int target_percent) {
    if (opts->debug < 2) return;
    if (target_percent >= 0) {
      printf("  %s: Target %d%%, Used %d/%d bytes (%.1f%%)\n", label, target_percent,
             shmem_used, shared_mem_per_block,
             100.0 * shmem_used / shared_mem_per_block);
    } else {
      printf("  %s: Used %d/%d bytes (%.1f%%)\n", label, shmem_used, shared_mem_per_block,
             100.0 * shmem_used / shared_mem_per_block);
    }
  };

  switch (opts->gpu_method) {
  case 1: {
    // Method 1 (Global Memory): Smaller bins improve performance via better cache behavior
    const double load_factor = (dim <= 2) ? 0.5 : 0.75;
    int target_shmem = static_cast<int>(shared_mem_per_block * load_factor);
    int binsize = calc_binsize(target_shmem);

    // Fallback to full memory for extreme tolerances (large ns)
    if (binsize < 1) {
      binsize = calc_binsize(shared_mem_per_block);
    }

    // If still can't fit, throw error - will be caught by impl.h retry logic
    if (binsize < 1) {
      throw std::runtime_error(
          std::string("[cufinufft] ERROR: Insufficient shared memory for Method 1 (Global Memory).\n"
          "           Available: ") +
          std::to_string(shared_mem_per_block) + " bytes, kernel width ns=" + std::to_string(ns) +
          ".\n"
          "           GPU has insufficient shared memory for this tolerance.");
    }

    set_binsizes_if_unset(binsize);

    print_method_setup(1, ns, 0, "");
    int shmem_used = shared_memory_required<T>(
        dim, ns, opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, 0);
    print_shmem_usage("Shmem", shmem_used, (dim <= 2) ? 50 : 75);
    break;
  }
  case 2: {
    // Method 2 (Shared Memory): Maximize bin size for optimal performance
    const double load_factor = 1.0;
    int target_shmem = static_cast<int>(shared_mem_per_block * load_factor);
    int binsize = calc_binsize(target_shmem);

    // Method 2 requires shared memory - check if sufficient
    if (binsize < 1) {
      throw std::runtime_error(
          std::string("[cufinufft] ERROR: Insufficient shared memory for Method 2 (Shared Memory).\n"
          "           Available: ") +
          std::to_string(shared_mem_per_block) + " bytes, kernel width ns=" + std::to_string(ns) +
          ".\n"
          "           Try Method 1 (Global Memory) or set method=0 for auto-select.");
    }

    set_binsizes_if_unset(binsize);

    print_method_setup(2, ns, 0, "");
    int shmem_used = shared_memory_required<T>(
        dim, ns, opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, 0);
    print_shmem_usage("Shmem", shmem_used, -1);
    break;
  }
  case 3: {
    // Method 3 (Output-Driven): Partition shmem between np and bins using ratio
    const double load_factor = 1.0;
    const int shmem_per_point = shared_memory_per_point<T>(dim, ns);
    int target_shmem = static_cast<int>(shared_mem_per_block * load_factor);

    // Dimension-dependent np allocation ratio (from benchmark analysis)
    // 1D: High parallelism (more np), 3D: More memory for bins (less np)
    double np_ratio;
    if (dim == 1) {
      np_ratio = 0.15;  // 15% shmem for np (benchmark-optimal: high parallelism)
    } else if (dim == 2) {
      np_ratio = 0.10;  // 10% shmem for np (balanced)
    } else {
      np_ratio = 0.05;  // 5% shmem for np (3D needs more bin memory)
    }

    // Calculate target np from ratio (round to multiple of 16, minimum 16)
    int np_shmem = static_cast<int>(target_shmem * np_ratio);
    int target_np = std::max(16, (np_shmem / shmem_per_point) & ~15);

    // Calculate binsize from remaining memory
    np_shmem = target_np * shmem_per_point;  // Recalc with rounded np
    int binsize = calc_binsize(target_shmem - np_shmem);

    // Fallback: If bins don't fit, reduce to minimum np=16
    if (binsize < 1 && target_np > 16) {
      target_np = 16;
      np_shmem  = shmem_per_point * 16;
      binsize   = calc_binsize(target_shmem - np_shmem);
    }

    if (binsize < 1) {
      throw std::runtime_error(
          std::string("[cufinufft] ERROR: Insufficient shared memory for Method 3 (Output-Driven).\n"
          "           Available: ") +
          std::to_string(shared_mem_per_block) + " bytes, kernel width ns=" + std::to_string(ns) +
          ".\n"
          "           Try Method 1 (Global Memory) or set method=0 for auto-select.");
    }

    set_binsizes_if_unset(binsize);

    // Calculate max np that fits with the chosen bin size (round down to multiple of 16)
    binsize            = opts->gpu_binsizex;
    int shmem_required = shared_memory_required<T>(dim, ns, binsize, opts->gpu_binsizey,
                                                   opts->gpu_binsizez, 0);
    int shmem_left     = shared_mem_per_block - shmem_required;
    int max_np = (shmem_left / shmem_per_point) & ~15; // Round down to multiple of 16

    // If can't fit minimum np=16, reduce bin size until it fits
    if (max_np < 16) {
      max_np = 16;
      while (binsize > 1) {
        int required = shared_memory_required<T>(dim, ns, binsize, binsize, binsize, 16);
        if (required <= shared_mem_per_block) break;
        binsize--;
      }
      if (binsize < 1) {
        throw std::runtime_error(
            std::string("[cufinufft] ERROR: Insufficient shared memory for Method 3 (Output-Driven).\n"
            "           Cannot fit minimum np=16 with valid bin sizes.\n"
            "           Available: ") +
            std::to_string(shared_mem_per_block) +
            " bytes.\n"
            "           Try Method 1 (Global Memory) or set method=0 for auto-select.");
      }

      set_binsizes(binsize, binsize, binsize);
      shmem_required = shared_memory_required<T>(dim, ns, binsize, binsize, binsize, 0);
      max_np =
          std::max(16, ((shared_mem_per_block - shmem_required) / shmem_per_point) & ~15);
    }

    print_method_setup(3, ns, max_np, "");
    int total_shmem = shared_memory_required<T>(
        dim, ns, opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, max_np);
    if (opts->debug >= 2) {
      double actual_np_ratio = (double)(shmem_per_point * max_np) / total_shmem;
      printf(
          "  Shmem: %d/%d bytes (%.1f%%), per_point=%d, grid_shmem=%d, np_shmem=%d\n",
          total_shmem, shared_mem_per_block, 100.0 * total_shmem / shared_mem_per_block,
          shmem_per_point, shmem_required, shmem_per_point * max_np);
      printf("  np/bins allocation ratio: %.3f (target=%.2f, np gets %.1f%% of shmem)\n",
             actual_np_ratio, np_ratio, 100.0 * actual_np_ratio);
    }
    assert(total_shmem <= shared_mem_per_block);
    opts->gpu_np = max_np;
    break;
  }
  case 4: {
    opts->gpu_obinsizex = opts->gpu_obinsizex == 0 ? 8 : opts->gpu_obinsizex;
    opts->gpu_obinsizey = opts->gpu_obinsizey == 0 ? 8 : opts->gpu_obinsizey;
    opts->gpu_obinsizez = opts->gpu_obinsizez == 0 ? 8 : opts->gpu_obinsizez;
    opts->gpu_binsizex  = opts->gpu_binsizex == 0 ? 4 : opts->gpu_binsizex;
    opts->gpu_binsizey  = opts->gpu_binsizey == 0 ? 4 : opts->gpu_binsizey;
    opts->gpu_binsizez  = opts->gpu_binsizez == 0 ? 4 : opts->gpu_binsizez;
    break;
  }
  default: {
    throw std::runtime_error(std::string("[cufinufft] ERROR: Invalid gpu_method=") +
                             std::to_string(opts->gpu_method) +
                             ". Valid methods: 0 (auto), 1 (GM), 2 (SM), 3 (OD), 4.");
  }
  }

  // Final safety guard: binsize should never be 0 (would cause segfault)
  if (opts->gpu_binsizex < 1 || opts->gpu_binsizey < 1 || opts->gpu_binsizez < 1) {
    throw std::runtime_error(
        std::string("[cufinufft] ERROR: Invalid bin sizes after setup: ") +
        std::to_string(opts->gpu_binsizex) + "x" + std::to_string(opts->gpu_binsizey) + "x" +
        std::to_string(opts->gpu_binsizez) +
        ".\n"
        "           This is a bug - bin sizes should never be < 1.\n"
        "           Shared memory available: " +
        std::to_string(shared_mem_per_block) +
        " bytes, method: " + std::to_string(opts->gpu_method) + ", ns: " + std::to_string(ns));
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

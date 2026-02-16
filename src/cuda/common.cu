#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

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
static __global__ void cu_fseries_kernel_compute(cuda::std::array<CUFINUFFT_BIGINT,3> nf123, const T *f, const T *phase,
                                          cuda::std::array<T *,3> fwkerhalf,
                                          int ns) {
  T J2  = ns / 2.0;
  int q = (int)(2 + 3.0 * J2);
  int nf = nf123[threadIdx.y];
  const T *phaset = phase + threadIdx.y * MAX_NQUAD;
  const T *ft     = f + threadIdx.y * MAX_NQUAD;
  T *oarr=fwkerhalf[threadIdx.y];

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
static __global__ void cu_nuft_kernel_compute(cuda::std::array<CUFINUFFT_BIGINT,3> nf123, T *f, T *z,
                                       cuda::std::array<T *,3> kxyz, cuda::std::array<T *,3> fwkerhalf, int ns) {
  T J2  = ns / 2.0;
  int q = (int)(2 + 2.0 * J2);
  int nf = nf123[threadIdx.y];
  T *at = z + threadIdx.y * MAX_NQUAD;
  T *ft = f + threadIdx.y * MAX_NQUAD;
  T *oarr = fwkerhalf[threadIdx.y], *k = kxyz[threadIdx.y];
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
int fseries_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, T *d_f, T *d_phase,
                           cuda::std::array<T *,3> d_fwkerhalf, int ns,
                           cudaStream_t stream)
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
      nf123, d_f, d_phase, d_fwkerhalf, ns);
  RETURN_IF_CUDA_ERROR

  return 0;
}
template int fseries_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, float *d_f, float *d_phase,
                           cuda::std::array<float *,3> d_fwkerhalf, int ns,
                           cudaStream_t stream);
template int fseries_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, double *d_f, double *d_phase,
                           cuda::std::array<double *,3> d_fwkerhalf, int ns,
                           cudaStream_t stream);

template<typename T>
int nuft_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, T *d_f, T *d_z,
                        cuda::std::array<T *,3> d_kxyz, cuda::std::array<T *,3> d_fwkerhalf,
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
      nf123, d_f, d_z, d_kxyz, d_fwkerhalf,
      ns);
  RETURN_IF_CUDA_ERROR

  return 0;
}
template int nuft_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, float *d_f, float *d_z,
                        cuda::std::array<float *,3> d_kxyz, cuda::std::array<float *,3> d_fwkerhalf,
                        int ns, cudaStream_t stream);
template int nuft_kernel_compute(int dim, cuda::std::array<CUFINUFFT_BIGINT,3> nf123, double *d_f, double *d_z,
                        cuda::std::array<double *,3> d_kxyz, cuda::std::array<double *,3> d_fwkerhalf,
                        int ns, cudaStream_t stream);

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
template int setup_spreader_for_nufft(finufft_spread_opts &spopts, float eps,
                                      cufinufft_opts opts);
template int setup_spreader_for_nufft(finufft_spread_opts &spopts, double eps,
                                      cufinufft_opts opts);

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
template void onedim_fseries_kernel_precomp<float>(CUFINUFFT_BIGINT nf, float *f,
                                                   float *a, finufft_spread_opts opts);
template void onedim_fseries_kernel_precomp<double>(CUFINUFFT_BIGINT nf, double *f,
                                                    double *a, finufft_spread_opts opts);

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

template<typename T> std::size_t shared_memory_per_point(int dim, int ns) {
  return ns * sizeof(T) * dim       // kernel evaluations
         + sizeof(int) * dim        // indexes
         + sizeof(cuda_complex<T>); // strength
}
template std::size_t shared_memory_per_point<float>(int dim, int ns);
template std::size_t shared_memory_per_point<double>(int dim, int ns);

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
template std::size_t shared_memory_required<float>(
    int dim, int ns, int bin_size_x, int bin_size_y, int bin_size_z, int np);
template std::size_t shared_memory_required<double>(
    int dim, int ns, int bin_size_x, int bin_size_y, int bin_size_z, int np);

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

namespace {
// ============================================================================
// GPU Classification for Binsize Selection
// ============================================================================
//
// Categorizes GPUs based on runtime-queryable device attributes
// to select optimal bin sizes for Methods 2 & 3.
//
// Background: Benchmark sweeps across 8 GPUs (A100-40GB, A100-80GB, H100-80GB,
// H100-94GB, H200, RTX 6000 Ada, RTX 4070 Mobile, RTX Blackwell) revealed:
//
// METHOD 2: Requires 2-4 GPU groups depending on dimension
// METHOD 3: Requires 4-5 GPU groups (highly architecture-sensitive)
//
//   Group 1 (Ampere Large):    A100 (CC 8.0, 164 KB/block)
//   Group 2 (Hopper):          H100/H200 (CC 9.0, 228 KB/block)
//   Group 3 (Small-SMEM Desk): Ada/Blackwell workstation-class parts
//                              (~100 KB/block, high SM count)
//   Group 4 (Small-SMEM Mob):  Mobile parts with low SM count (need very large np)
//
// Key insight: Method 3 couples shared memory footprint with work granularity
// in ways that make different GPU architectures prefer vastly different configs.
// ============================================================================

// Method 3 GPU categories for granular heuristic selection
enum class Method3Category {
  AMPERE_LARGE, // A100: CC 8.0, prefers low shmem, small np
  HOPPER,       // H100/H200: CC 9.0, can use larger np
  ADA_DESKTOP,  // Small-SMEM desktop/workstation: Ada/Blackwell, high SM count
  ADA_MOBILE,   // Small-SMEM mobile: low SM count
  UNKNOWN       // Fallback
};

struct GpuCharacteristics {
  int cc_major, cc_minor;
  int max_smem_per_block_optin; // bytes
  int max_smem_per_sm;          // bytes
  int max_threads_per_sm;
  int max_warps_per_sm;         // derived: max_threads_per_sm / 32
  int multiprocessor_count;     // SM count
  std::string gpu_name;

  static GpuCharacteristics query(int device_id) {
    GpuCharacteristics gpu{};

    // Query compute capability
    cudaDeviceGetAttribute(&gpu.cc_major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&gpu.cc_minor, cudaDevAttrComputeCapabilityMinor, device_id);

    // Query shared memory limits (primary classification signals)
    cudaDeviceGetAttribute(&gpu.max_smem_per_block_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    cudaDeviceGetAttribute(&gpu.max_smem_per_sm,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id);

    // Query occupancy limiters
    cudaDeviceGetAttribute(&gpu.max_threads_per_sm,
                           cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
    gpu.max_warps_per_sm = gpu.max_threads_per_sm / 32;

    // Query SM count (for Ada desktop vs mobile classification)
    cudaDeviceGetAttribute(&gpu.multiprocessor_count, cudaDevAttrMultiProcessorCount,
                           device_id);

    // Get GPU name
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);
    gpu.gpu_name = std::string(prop.name);

    return gpu;
  }

  // Check if this GPU is "Hopper-like" for Method 2/3 binsize selection.
  // "Hopper-like" = large shared memory (≥200 KB/block) AND high occupancy (≥64 warps/SM)
  // Matches: H100 (9.0), H200 (9.0), Blackwell datacenter (10.0)
  bool is_hopper_like() const {
    return (max_smem_per_block_optin >= 200 * 1024) && (max_warps_per_sm >= 64);
  }

  // Check if this GPU has "small SMEM" constraints (≤110 KB/block).
  // Matches: Ada (8.9), Ampere 8.6, Blackwell workstation (12.0)
  bool is_small_smem() const { return (max_smem_per_block_optin <= 110 * 1024); }

  // Classify GPU for Method 3 heuristics (5-category system)
  Method3Category get_method3_category() const {
    // CC 8.0 = Ampere large (A100, A30)
    if (cc_major == 8 && cc_minor == 0) {
      return Method3Category::AMPERE_LARGE;
    }

    // CC 9.0 = Hopper (H100, H200)
    if (cc_major == 9 && cc_minor == 0) {
      return Method3Category::HOPPER;
    }

    // CC 8.9 = Ada Lovelace (need SM count to distinguish desktop vs mobile)
    // Desktop (RTX 6000 Ada): 142 SMs
    // Mobile (RTX 4070): typically 36-46 SMs
    // Threshold: 80 SMs
    if (cc_major == 8 && cc_minor == 9) {
      return (multiprocessor_count >= 80) ? Method3Category::ADA_DESKTOP
                                          : Method3Category::ADA_MOBILE;
    }

    // CC 12.0 = Blackwell workstation (RTX 6000/5000 Blackwell)
    // Reuse the same binsize table as Ada desktop (both are ~100KB/block parts).
    if (cc_major == 12 && cc_minor == 0) return Method3Category::ADA_DESKTOP;

    // CC 10.0 = Blackwell datacenter (B200) - treat as Hopper-like
    if (cc_major == 10 && cc_minor == 0) {
      return Method3Category::HOPPER;
    }

    // Unknown / future architectures: fallback to simple classification
    return Method3Category::UNKNOWN;
  }

  const char *method3_category_name() const {
    switch (get_method3_category()) {
    case Method3Category::AMPERE_LARGE:
      return "Ampere-Large";
    case Method3Category::HOPPER:
      return "Hopper";
    case Method3Category::ADA_DESKTOP:
      return "Small-SMEM-Desktop";
    case Method3Category::ADA_MOBILE:
      return "Small-SMEM-Mobile";
    case Method3Category::UNKNOWN:
      return "Unknown";
    }
    return "Unknown";
  }

  void print_classification(int debug_level) const {
    if (debug_level < 2) return;

    printf("[cufinufft] GPU Classification:\n");
    printf("  Name: %s\n", gpu_name.c_str());
    printf("  Compute Capability: %d.%d\n", cc_major, cc_minor);
    printf("  Multiprocessor Count: %d SMs\n", multiprocessor_count);
    printf("  Shared Memory:\n");
    printf("    Per-block (opt-in): %.1f KB\n", max_smem_per_block_optin / 1024.0);
    printf("    Per-SM: %.1f KB\n", max_smem_per_sm / 1024.0);
    printf("  Occupancy:\n");
    printf("    Max warps/SM: %d\n", max_warps_per_sm);
    printf("    Max threads/SM: %d\n", max_threads_per_sm);

    if (debug_level >= 3) {
      printf("  Binsize Categories:\n");
      printf("    Method 2/3: Hopper-like (≥200 KB/block, ≥64 warps): %s\n",
             is_hopper_like() ? "YES" : "NO");
      printf("    Method 2/3: Small SMEM (≤110 KB/block): %s\n",
             is_small_smem() ? "YES" : "NO");
      printf("    Method 3: Category: %s\n", method3_category_name());
    }
  }
};

// ============================================================================
// Method 3 Heuristic (Benchmark-Validated Tables)
// ============================================================================
// Returns (bin,np) for Method 3 based on GPU category, dimension, and ns.
// Tables derived from multi-GPU sweeps; achieves ~90% of optimal throughput.
// ============================================================================
struct Method3Config {
  int bin;
  int np;
};

inline int ns_bucket(int ns) {
  return (ns <= 4) ? 0 : (ns <= 7) ? 1 : (ns <= 10) ? 2 : 3;
}

template<typename T>
Method3Config get_method3_config(Method3Category category, int dim, int ns,
                                 int shmem_limit_bytes, int shmem_per_point_bytes) {
  const int complex_bytes = sizeof(cuda_complex<T>);
  // For unknown GPUs: compute bins dynamically based on available SMEM
  if (category == Method3Category::UNKNOWN) {
    // Iteratively find (bin, np) satisfying: bin >= 8, np >= 16
    // Start conservative (60% to grid), increase by 10% each iteration up to 100%
    double load_factor = 0.60;
    int bin            = 0;
    int np             = 0;

    while (load_factor <= 1.0) {
      const int grid_budget = static_cast<int>(shmem_limit_bytes * load_factor);
      const int elements    = grid_budget / complex_bytes;
      const int padded_bin  = static_cast<int>(std::floor(std::pow(elements, 1.0 / dim)));
      bin                   = padded_bin - (2 * (ns + 1) / 2);

      // Calculate actual grid memory and check remaining space for np
      const size_t grid_mem = shared_memory_required<T>(dim, ns, bin, bin, bin, 0);
      const int shmem_rem   = shmem_limit_bytes - static_cast<int>(grid_mem);
      np                    = (shmem_rem / shmem_per_point_bytes) & ~15;

      // Accept if both bin reasonable AND np sufficient (or we've exhausted options)
      if ((bin >= 8 && np >= 16) || load_factor >= 1.0) break;

      load_factor += 0.10; // Try 10% more aggressive
    }

    // Ensure minimums
    bin = std::max(4, bin);
    np  = std::max(16, std::min(2048, np));

    return {bin, np};
  }

  // Known GPUs: use benchmark-validated tables
  // Tables: [category][dim-1][ns_bucket] for 4 GPU types × 3 dims × 4 ns ranges
  static constexpr int BIN[4][3][4] = {
      {{362, 351, 753, 285}, {29, 25, 25, 23}, {8, 8, 7, 7}},    // AMPERE_LARGE
      {{511, 532, 441, 446}, {17, 14, 12, 21}, {10, 8, 8, 9}},   // HOPPER
      {{212, 169, 222, 123}, {23, 16, 13, 23}, {9, 9, 7, 4}},    // SMALL-SMEM DESKTOP
      {{199, 423, 191, 356}, {34, 39, 40, 40}, {12, 10, 7, 4}}}; // SMALL-SMEM MOBILE

  static constexpr int NP[4][3][4] = {
      {{208, 144, 128, 96}, {16, 16, 64, 48}, {96, 64, 16, 48}},      // AMPERE_LARGE
      {{288, 192, 160, 128}, {176, 112, 80, 112}, {240, 16, 64, 64}}, // HOPPER
      {{128, 96, 64, 64}, {80, 80, 112, 32}, {112, 112, 16, 16}}, // SMALL-SMEM DESKTOP
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};                // MOBILE (computed)

  const int category_idx = (category == Method3Category::AMPERE_LARGE)  ? 0
                           : (category == Method3Category::HOPPER)      ? 1
                           : (category == Method3Category::ADA_DESKTOP) ? 2
                                                                        : 3; // ADA_MOBILE

  const int dim_idx = std::max(0, std::min(2, dim - 1));
  const int ns_idx  = ns_bucket(ns);

  Method3Config cfg{BIN[category_idx][dim_idx][ns_idx],
                    NP[category_idx][dim_idx][ns_idx]};

  // Mobile GPUs: compute np to fill remaining shmem after grid tile
  // Constraint: np must be at least 16 for reasonable performance
  if (cfg.np == 0) {
    // Start with table bin, reduce if necessary to ensure np >= 16
    int bin = cfg.bin;

    while (bin >= 1) {
      const size_t grid_mem = shared_memory_required<T>(dim, ns, bin, bin, bin, 0);
      const int shmem_rem   = shmem_limit_bytes - static_cast<int>(grid_mem);
      const int np          = shmem_rem / shmem_per_point_bytes;

      if (np >= 16) {
        cfg.bin = bin;
        cfg.np  = np;
        break;
      }

      bin--; // Table bin too large, try smaller
    }

    if (cfg.np == 0) {
      throw std::runtime_error("[cufinufft] Mobile GPU: insufficient SMEM for Method 3 "
                               "(cannot satisfy np≥16 even with bin=1)");
    }
  }

  cfg.np = std::max(16, std::min(2048, cfg.np & ~15));
  return cfg;
}

} // anonymous namespace

template<typename T>
void cufinufft_setup_binsize([[maybe_unused]] int type, int ns, int dim,
                             cufinufft_opts *opts) {
  const int device_id = opts->gpu_device_id;
  int shmem_limit{};
  cudaDeviceGetAttribute(&shmem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         device_id);

  const auto gpu         = GpuCharacteristics::query(device_id);
  const int shmem_per_pt = static_cast<int>(shared_memory_per_point<T>(dim, ns));

  gpu.print_classification(opts->debug);

  auto set_bins = [&](int bin) {
    if (opts->gpu_binsizex == 0) opts->gpu_binsizex = bin;
    if (opts->gpu_binsizey == 0) opts->gpu_binsizey = (dim >= 2) ? bin : 1;
    if (opts->gpu_binsizez == 0) opts->gpu_binsizez = (dim >= 3) ? bin : 1;
  };

  auto validate_fit = [&](int np) {
    size_t need = shared_memory_required<T>(dim, ns, opts->gpu_binsizex,
                                            opts->gpu_binsizey, opts->gpu_binsizez, np);
    if (need > static_cast<size_t>(shmem_limit)) {
      throw std::runtime_error("[cufinufft] Config exceeds " +
                               std::to_string(shmem_limit) + " bytes available (needs " +
                               std::to_string(need) + " bytes)");
    }
  };

  auto debug_print = [&](int method, int np, const char *note) {
    if (opts->debug < 1) return;
    printf("[cufinufft] Method %d: dim=%d, ns=%d, bin=%dx%dx%d", method, dim, ns,
           opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez);
    if (np > 0) printf(", np=%d", np);
    if (note[0])
      printf(" %s", note);
    else if (method == 3 && np > 0)
      printf(" [%s]", gpu.method3_category_name());
    printf("\n");
    if (opts->debug >= 2) {
      size_t use = shared_memory_required<T>(dim, ns, opts->gpu_binsizex,
                                             opts->gpu_binsizey, opts->gpu_binsizez, np);
      printf("  Shmem: %zu/%d bytes (%.1f%%)\n", use, shmem_limit,
             100.0 * use / shmem_limit);
    }
  };

  switch (opts->gpu_method) {
  case 1:
    set_bins(dim == 1 ? 1024 : dim == 2 ? 40 : 8);
    debug_print(1, 0, "");
    break;

  case 2: {
    int bin = (dim == 1) ? 1024 : (dim == 2) ? 40 : 0;
    if (bin == 0) {
      double load = (ns <= 6) ? 0.50 : (ns <= 10 && !gpu.is_small_smem()) ? 0.90 : 1.0;
      int target  = static_cast<int>(shmem_limit * load);
      bin         = find_bin_size<T>(target, dim, ns);
      if (bin < 1) bin = find_bin_size<T>(shmem_limit, dim, ns);
    }
    if (bin < 1)
      throw std::runtime_error("[cufinufft] Insufficient shmem for Method 2 (ns=" +
                               std::to_string(ns) + "). Try Method 1.");
    set_bins(bin);
    validate_fit(0);
    debug_print(2, 0, "");
    break;
  }

  case 3: {
    const bool user_np  = (opts->gpu_np != 0);
    const bool user_bin = (opts->gpu_binsizex | opts->gpu_binsizey | opts->gpu_binsizez);

    if (user_np && !user_bin) {
      int avail = shmem_limit - opts->gpu_np * shmem_per_pt;
      if (avail <= 0)
        throw std::runtime_error(
            "[cufinufft] gpu_np=" + std::to_string(opts->gpu_np) + " too large");
      set_bins(find_bin_size<T>(avail, dim, ns));
      validate_fit(opts->gpu_np);
      debug_print(3, opts->gpu_np, "(user np)");
    } else if (user_bin) {
      int grid_mem = static_cast<int>(shared_memory_required<T>(
          dim, ns, opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, 0));
      int rem      = shmem_limit - grid_mem;
      if (rem < shmem_per_pt * 16)
        throw std::runtime_error("[cufinufft] User bin too large (no room for np≥16)");
      opts->gpu_np = std::max(16, (rem / shmem_per_pt) & ~15);
      validate_fit(opts->gpu_np);
      debug_print(3, opts->gpu_np, user_np ? "(user)" : "(user bin)");
    } else {
      auto cfg = get_method3_config<T>(gpu.get_method3_category(), dim, ns, shmem_limit,
                                       shmem_per_pt);
      set_bins(cfg.bin);
      opts->gpu_np = cfg.np;
      validate_fit(cfg.np);
      debug_print(3, cfg.np, "");
    }
    break;
  }

  case 4:
    if (opts->gpu_obinsizex == 0) opts->gpu_obinsizex = 8;
    if (opts->gpu_obinsizey == 0) opts->gpu_obinsizey = 8;
    if (opts->gpu_obinsizez == 0) opts->gpu_obinsizez = 8;
    if (opts->gpu_binsizex == 0) opts->gpu_binsizex = 4;
    if (opts->gpu_binsizey == 0) opts->gpu_binsizey = 4;
    if (opts->gpu_binsizez == 0) opts->gpu_binsizez = 4;
    break;

  default:
    throw std::runtime_error(
        "[cufinufft] Invalid gpu_method=" + std::to_string(opts->gpu_method));
  }

  if (opts->gpu_binsizex < 1 || opts->gpu_binsizey < 1 || opts->gpu_binsizez < 1)
    throw std::runtime_error(
        "[cufinufft] BUG: Invalid bin sizes (method=" + std::to_string(opts->gpu_method) +
        ", ns=" + std::to_string(ns) + ")");
}

template void cufinufft_setup_binsize<float>(int type, int ns, int dim,
                                             cufinufft_opts *opts);
template void cufinufft_setup_binsize<double>(int type, int ns, int dim,
                                              cufinufft_opts *opts);
} // namespace common
} // namespace cufinufft

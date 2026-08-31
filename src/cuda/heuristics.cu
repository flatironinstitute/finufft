// GPU bin-size heuristics, derived from device attributes (L2, bus width, shared
// memory) rather than per-architecture tables. Owns the shared-memory accounting
// used to validate that a chosen bin/np combination fits.

#include <algorithm>
#include <array>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <cuda.h>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/heuristics.hpp>
#include <cufinufft/spreadinterp.hpp>
#include <cufinufft/utils.hpp>
#include <finufft_common/kernel.h>

// GpuCapabilities member declared in types.hpp; defined here to keep <cstdio>
// out of the header.
void GpuCapabilities::print_classification(int debug_level) const {
  if (debug_level < 2) return;

  // Only for the name; every other field comes from cudaDeviceGetAttribute.
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device_id);
  printf("[cufinufft] GPU Classification:\n");
  printf("  Name: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", cc_major, cc_minor);
  printf("  Multiprocessor Count: %d SMs\n", multiprocessor_count);
  printf("  Shared Memory:\n");
  printf("    Per-block (opt-in): %.1f KB\n", max_smem_per_block_optin / 1024.0);
  printf("    Per-SM: %.1f KB\n", max_smem_per_sm / 1024.0);
  printf("  Occupancy:\n");
  printf("    Max warps/SM: %d\n", max_warps_per_sm());
  printf("    Max threads/SM: %d\n", max_threads_per_sm);
  printf("  L2: %.1f MB\n", l2_cache_size / 1048576.0);
  printf("  Memory bus: %d-bit (%s)\n", memory_bus_width,
         wide_memory_bus() ? "wide" : "narrow");
}

namespace cufinufft {
namespace common {
using std::max;

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

// Largest r with r^dim <= value. std::pow alone is short by one ulp on exact powers
// (pow(64, 1/3.) < 4 in IEEE 754), so retry the next integer.
static int integer_root(double value, int dim) {
  auto r = int(std::floor(std::pow(value, 1.0 / dim)));
  if (std::pow(double(r + 1), dim) <= value) ++r;
  return r;
}

// Function to find bin_size_x == bin_size_y
// where bin_size_x * bin_size_y * bin_size_z < mem_size
template<typename T> int find_bin_size(std::size_t mem_size, int dim, int ns) {
  const auto elements        = mem_size / sizeof(cuda_complex<T>);
  const auto padded_bin_size = integer_root(double(elements), dim);
  const auto bin_size        = padded_bin_size - 2 * ((ns + 1) / 2);
  // TODO: one dimension could take a larger bin; the tile shape need not be uniform
  return bin_size;
}

template<typename T>
void cufinufft_setup_binsize(const GpuCapabilities &gpu, int type, int ns, int dim,
                             const CUFINUFFT_BIGINT *mstu, cufinufft_opts *opts) {
  const int shmem_limit = gpu.max_smem_per_block_optin;
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
    if (note[0]) printf(" %s", note);
    printf("\n");
    if (opts->debug >= 2) {
      size_t use = shared_memory_required<T>(dim, ns, opts->gpu_binsizex,
                                             opts->gpu_binsizey, opts->gpu_binsizez, np);
      printf("  Shmem: %zu/%d bytes (%.1f%%)\n", use, shmem_limit,
             100.0 * use / shmem_limit);
    }
  };

  switch (opts->gpu_method) {
  case 1: {
    // The GM kernel stages no tile in shared memory; the bin width only sets setpts'
    // sort granularity, which trades locality of the scattered grid atomics (small
    // bins) against their spread over L2 sectors (large bins). bin = nf makes one bin
    // per dim and skips the sort entirely.

    // The pre-#807 fixed bins, for every case no L2 rule covers.
    const auto sized_bins = [&] {
      return dim == 1 ? 1024 : dim == 2 ? 40 : 8;
    };

    // Fine grid per dim, empty when the modes are not known at plan time.
    const auto fine_grid = [&]() -> std::optional<std::array<CUFINUFFT_BIGINT, 3>> {
      if (type == 3) return std::nullopt; // outer plan: modes unset
      std::array<CUFINUFFT_BIGINT, 3> nf{1, 1, 1};
      for (int d = 0; d < dim; ++d) {
        if (mstu[d] <= 0) return std::nullopt; // unset: sub-plan sized later
        nf[d] = opts->gpu_spreadinterponly
                    ? mstu[d]
                    : CUFINUFFT_BIGINT(
                          finufft::common::fine_grid_len(opts->upsampfac, mstu[d], ns));
      }
      return nf;
    }();

    if (!fine_grid) {
      set_bins(sized_bins());
      debug_print(1, 0, "");
      break;
    }

    const auto &nf_est         = *fine_grid;
    const auto cells           = std::int64_t(nf_est[0]) * nf_est[1] * nf_est[2];
    const auto budget          = gpu.l2_complex_budget<T>();
    const bool wide            = gpu.wide_memory_bus();

    // Halve each dim `shift` times: bin = ceil(nf / 2^shift).
    const auto set_bins_halved = [&](int shift) {
      const auto halved = [&](int d) {
        const auto v = (nf_est[d] + (CUFINUFFT_BIGINT(1) << shift) - 1) >> shift;
        return int(std::clamp<CUFINUFFT_BIGINT>(v, 1, std::numeric_limits<int>::max()));
      };
      if (opts->gpu_binsizex == 0) opts->gpu_binsizex = halved(0);
      if (opts->gpu_binsizey == 0) opts->gpu_binsizey = dim >= 2 ? halved(1) : 1;
      if (opts->gpu_binsizez == 0) opts->gpu_binsizez = dim >= 3 ? halved(2) : 1;
    };

    // t2 gather reads the grid through the texture path: 1D never wants a sort, 2D
    // only pays once the grid leaves ~2 budgets, 3D always sorts.
    const auto gather_shift = [&]() -> std::optional<int> {
      if (dim == 1 || (dim == 2 && cells <= 2 * budget)) return 0;
      return std::nullopt;
    };

    // t1 scatter keeps one bin while the grid stays L2-resident, then halves per dim
    // while each 2^(k*dim) tile still fits K, inside a per-dim spill band. Deeper
    // grids fall back to the sized bins.
    const auto scatter_shift = [&]() -> std::optional<int> {
      // 2 budgets on wide-bus parts and in 1D, 1 budget on 384-bit GDDR.
      const std::int64_t resident_cap = (wide || dim == 1) ? 2 * budget : budget;
      if (cells <= resident_cap) return 0;
      if (dim == 1) return std::nullopt;
      // K: 1 budget on Hopper/Blackwell, 1/2 on split-L2 sm_80 and on GDDR.
      const std::int64_t tile_cap = (wide && gpu.cc_major != 8) ? budget : budget / 2;
      const std::int64_t band =
          dim == 2 ? (wide ? std::numeric_limits<std::int64_t>::max() : 4 * budget)
                   : (wide ? 8 * budget : 2 * budget);
      if (cells > band) return std::nullopt;
      for (int k = 1; k <= 2; ++k)
        if ((cells >> (k * dim)) <= tile_cap) return k;
      return std::nullopt;
    };

    const auto shift = (type == 2) ? gather_shift() : scatter_shift();
    if (shift.has_value())
      set_bins_halved(*shift);
    else
      set_bins(sized_bins());
    debug_print(1, 0, "");
    break;
  }

  case 2: {
    // One subproblem is one block's batch of <= maxsubprobsize points, so size the
    // bin to fill a subproblem at the ~1 point/cell design density: bin^dim = msub,
    // capped by the largest padded tile that fits shared memory. Dense f64 3D on
    // large-smem parts wants a smaller bin, but its optimum is device-jagged and
    // method 3 serves those cells faster, so no table is worth reintroducing.
    const int bin = std::min(integer_root(double(opts->gpu_maxsubprobsize), dim),
                             find_bin_size<T>(shmem_limit, dim, ns));
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

    // Points the derived pick stages per batch: one warp. Larger np costs shared
    // memory and occupancy without adding work. The tile is capped against the
    // memory left after them, not against the whole limit, because the block
    // allocates tile + derived_np * shmem_per_pt.
    constexpr int derived_np = 32;

    // The bin response is a broad valley: its lower edge (halo amplification,
    // (1+2*ceil(ns/2)/bin)^dim work per point) is device-independent, its upper edge
    // (resident blocks/SM = smem/tile) only rises with more shared memory. A
    // low-valley pick therefore transfers across devices: bin 256/16/6, capped by the
    // largest tile that fits.
    const auto valley_bin = [&] {
      const int target = dim == 1 ? 256 : dim == 2 ? 16 : 6;
      const int tile_mem = std::max(0, shmem_limit - derived_np * shmem_per_pt);
      return std::min(target, find_bin_size<T>(tile_mem, dim, ns));
    };

    // np filling the shared memory left after the tile the bins imply.
    const auto np_for_rest = [&] {
      const auto grid_mem = int(shared_memory_required<T>(
          dim, ns, opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, 0));
      const int rem       = shmem_limit - grid_mem;
      if (rem < shmem_per_pt * 16)
        throw std::runtime_error("[cufinufft] User bin too large (no room for np≥16)");
      return std::max(16, (rem / shmem_per_pt) & ~15);
    };

    const char *note = "";
    if (user_np && !user_bin) {
      const int avail = shmem_limit - opts->gpu_np * shmem_per_pt;
      if (avail <= 0)
        throw std::runtime_error(
            "[cufinufft] gpu_np=" + std::to_string(opts->gpu_np) + " too large");
      set_bins(find_bin_size<T>(avail, dim, ns));
      note = "(user np)";
    } else if (user_bin) {
      // A partial user bin leaves dims at 0 (zero-size tile, nbins div-by-zero
      // downstream): fill the unset dims with the derived pick.
      set_bins(std::max(1, valley_bin()));
      if (!user_np) opts->gpu_np = np_for_rest();
      note = user_np ? "(user)" : "(user bin)";
    } else {
      const int bin = valley_bin();
      if (bin < 1)
        throw std::runtime_error("[cufinufft] Insufficient shmem for Method 3 (ns=" +
                                 std::to_string(ns) + "). Try Method 1.");
      set_bins(bin);
      opts->gpu_np = derived_np;
    }
    validate_fit(opts->gpu_np);
    debug_print(3, opts->gpu_np, note);
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

template<typename T>
int choose_batchsize(const GpuCapabilities &gpu, const cufinufft_opts &opts, int ntransf,
                     std::int64_t nf) {
  // Cap at ntransf: a larger batch would make cuFFT transform grids that are then
  // discarded.
  if (opts.gpu_maxbatchsize) return std::min(opts.gpu_maxbatchsize, ntransf);

  // No FFT to amortize a batch against, so it would only widen the working set. For a
  // type 3's outer plan it would also evict the L2 that the inner type-2's FFT needs.
  // Timings: https://github.com/flatironinstitute/finufft/pull/873
  if (opts.gpu_spreadinterponly) return 1;

  // Keep nf*batchsize inside the L2 budget, up to 32 to fill the SMs at small nf. Past
  // the budget a batch only adds FFT work the grid cannot hold. Multi-GPU timings:
  // https://github.com/flatironinstitute/finufft/pull/873
  const std::int64_t l2_elems = gpu.l2_complex_budget<T>();
  const int cap = int(std::clamp<std::int64_t>(l2_elems / nf, 1, std::min(ntransf, 32)));

  // Spread out the batches evenly: the cuFFT plan is fixed at batchsize, and cufft_ex has
  // no per-call count, so the last batch always transforms batchsize grids even when only
  // blksize of them hold data. ntransf=9 with cap 8 would do 2x8 grid FFTs for 9
  // transforms; balancing gives 2x5 instead. Never above cap, so the L2 bound still
  // holds.
  const int nbatch = 1 + (ntransf - 1) / cap;
  return 1 + (ntransf - 1) / nbatch;
}

template void cufinufft_setup_binsize<float>(const GpuCapabilities &, int type, int ns,
                                             int dim, const CUFINUFFT_BIGINT *mstu,
                                             cufinufft_opts *opts);
template void cufinufft_setup_binsize<double>(const GpuCapabilities &, int type, int ns,
                                              int dim, const CUFINUFFT_BIGINT *mstu,
                                              cufinufft_opts *opts);
template int choose_batchsize<float>(const GpuCapabilities &, const cufinufft_opts &, int,
                                     std::int64_t);
template int choose_batchsize<double>(const GpuCapabilities &, const cufinufft_opts &,
                                      int, std::int64_t);
} // namespace common
} // namespace cufinufft

template<typename T> std::size_t cufinufft_plan_t<T>::shared_memory_required() const {
  return cufinufft::common::shared_memory_required<T>(
      dim, spopts.nspread, opts.gpu_binsizex, opts.gpu_binsizey, opts.gpu_binsizez,
      opts.gpu_np);
}
template std::size_t cufinufft_plan_t<float>::shared_memory_required() const;
template std::size_t cufinufft_plan_t<double>::shared_memory_required() const;

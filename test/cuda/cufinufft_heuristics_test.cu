// Host-side checks of cufinufft_setup_binsize: every auto pick stays inside the
// shared-memory budget, explicit user (bin, np) survive verbatim, a partial user
// bin gets its unset dims filled, and the msub / smem roots floor exactly.
//
// cufinufft_setup_binsize is pure and takes GpuCapabilities by value, so the whole
// budget sweep runs against a table of synthetic devices as well as the live one.
// A runner holds one card; the rule it exercises is a function of L2, bus width and
// shared memory, so one card validates one point of it.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>

#include <cufinufft.h>
#include <cufinufft/heuristics.hpp>
#include <finufft_common/constants.h>

using cufinufft::common::cufinufft_setup_binsize;
using cufinufft::common::shared_memory_required;

namespace {

// cudaDeviceGetAttribute values, from the cards themselves or from the occupancy
// limits in the CUDA C Programming Guide: cc, SMs, smem/block opt-in, smem/SM,
// threads/SM, L2 bytes, bus bits. Spans both bus classes and both smem classes.
struct DeviceProfile {
  const char *name;
  int cc_major, cc_minor, sms, smem_block, smem_sm, threads_sm, l2, bus;
};
constexpr DeviceProfile kProfiles[] = {
    {"V100", 7, 0, 80, 98304, 98304, 2048, 6291456, 4096},
    {"A100", 8, 0, 108, 166912, 167936, 2048, 41943040, 5120},
    {"H100", 9, 0, 132, 232448, 233472, 2048, 52428800, 5120},
    {"L40S", 8, 9, 142, 101376, 102400, 1536, 100663296, 384},
    {"RTX 4070 Laptop", 8, 9, 36, 101376, 102400, 1536, 33554432, 128},
};

GpuCapabilities synthetic(const DeviceProfile &p) {
  GpuCapabilities g{};
  g.cc_major                 = p.cc_major;
  g.cc_minor                 = p.cc_minor;
  g.multiprocessor_count     = p.sms;
  g.max_smem_per_block_optin = p.smem_block;
  g.max_smem_per_sm          = p.smem_sm;
  g.max_threads_per_sm       = p.threads_sm;
  g.l2_cache_size            = p.l2;
  g.memory_bus_width         = p.bus;
  g.memory_pools_supported   = 1;
  return g;
}

cufinufft_opts base_opts(int method) {
  cufinufft_opts o;
  cufinufft_default_opts(&o);
  o.gpu_method = method;
  o.upsampfac  = 2.0;
  return o;
}

// Every auto pick over the full accepted kernel width, on one device. No pick may
// throw and none may ask for more shared memory than the device grants: bin sizes
// index that tile, so an over-budget pick is a memory bug before it is a slow one.
template<typename T> void budget_sweep(const GpuCapabilities &gpu, const char *device) {
  const CUFINUFFT_BIGINT mstu[3] = {128, 128, 128};
  const auto limit               = std::size_t(gpu.max_smem_per_block_optin);

  for (int method = 1; method <= 3; ++method)
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 3; ++type)
        for (int ns = finufft::common::MIN_NSPREAD; ns <= finufft::common::MAX_NSPREAD<T>;
             ++ns) {
          auto o = base_opts(method);
          try {
            cufinufft_setup_binsize<T>(gpu, type, ns, dim, mstu, &o);
          } catch (const std::exception &e) {
            printf("%s: method %d dim %d type %d ns %d threw: %s\n", device, method, dim,
                   type, ns, e.what());
            assert(false && "an auto pick must not throw");
          }
          assert(o.gpu_binsizex >= 1);
          assert(o.gpu_binsizey >= 1 && (dim >= 2 || o.gpu_binsizey == 1));
          assert(o.gpu_binsizez >= 1 && (dim >= 3 || o.gpu_binsizez == 1));
          if (method >= 2) {
            const auto need = shared_memory_required<T>(
                dim, ns, o.gpu_binsizex, o.gpu_binsizey, o.gpu_binsizez, o.gpu_np);
            if (need > limit)
              printf("%s: method %d dim %d type %d ns %d needs %zu of %zu bytes\n",
                     device, method, dim, type, ns, need, limit);
            assert(need <= limit);
          }
          if (method == 3) assert(o.gpu_np >= 16);
        }
}

template<typename T> void run(const GpuCapabilities &gpu) {
  const CUFINUFFT_BIGINT mstu[3] = {128, 128, 128};
  const auto limit               = std::size_t(gpu.max_smem_per_block_optin);

  budget_sweep<T>(gpu, "live device");
  for (const auto &p : kProfiles) budget_sweep<T>(synthetic(p), p.name);

  { // method 3: explicit (bin, np) is honored verbatim
    auto o         = base_opts(3);
    o.gpu_binsizex = o.gpu_binsizey = o.gpu_binsizez = 4;
    o.gpu_np                                         = 64;
    cufinufft_setup_binsize<T>(gpu, 1, 4, 3, mstu, &o);
    assert(o.gpu_binsizex == 4 && o.gpu_binsizey == 4 && o.gpu_binsizez == 4);
    assert(o.gpu_np == 64);
  }
  { // method 3: user np alone is honored, bins derived
    auto o   = base_opts(3);
    o.gpu_np = 48;
    cufinufft_setup_binsize<T>(gpu, 1, 7, 2, mstu, &o);
    assert(o.gpu_np == 48 && o.gpu_binsizex >= 1 && o.gpu_binsizey >= 1);
  }
  { // method 3: a partial user bin fills the unset dims
    auto o         = base_opts(3);
    o.gpu_binsizex = 8;
    cufinufft_setup_binsize<T>(gpu, 1, 7, 3, mstu, &o);
    assert(o.gpu_binsizex == 8 && o.gpu_binsizey >= 1 && o.gpu_binsizez >= 1);
    assert(o.gpu_np >= 16);
  }
  { // method 3: partial user bin with explicit np — both kept, tile validated
    auto o         = base_opts(3);
    o.gpu_binsizex = 8;
    o.gpu_np       = 128;
    cufinufft_setup_binsize<T>(gpu, 1, 7, 3, mstu, &o);
    assert(o.gpu_binsizex == 8 && o.gpu_binsizey >= 1 && o.gpu_binsizez >= 1);
    assert(o.gpu_np == 128);
    assert(shared_memory_required<T>(3, 7, o.gpu_binsizex, o.gpu_binsizey, o.gpu_binsizez,
                                     o.gpu_np) <= limit);
  }
  { // method 2: floor(msub^(1/dim)) must be exact for perfect cubes
    auto o               = base_opts(2);
    o.gpu_maxsubprobsize = 64;
    cufinufft_setup_binsize<T>(gpu, 1, 4, 3, mstu, &o);
    assert(o.gpu_binsizex == 4);
  }
  { // smem cap is maximal: the picked bin fits, one more per dim does not
    auto o               = base_opts(2);
    o.gpu_maxsubprobsize = 1 << 30; // bin comes from the smem cap alone
    const int ns         = 10;      // even ns: the case the old precedence bug shrank
    cufinufft_setup_binsize<T>(gpu, 1, ns, 3, mstu, &o);
    const int b = o.gpu_binsizex;
    assert(shared_memory_required<T>(3, ns, b, b, b, 0) <= limit);
    assert(shared_memory_required<T>(3, ns, b + 1, b + 1, b + 1, 0) > limit);
  }
}

} // namespace

int main() {
  const auto gpu = GpuCapabilities::query(0);
  run<float>(gpu);
  run<double>(gpu);
  printf("cufinufft_heuristics_test: all assertions passed\n");
  return 0;
}

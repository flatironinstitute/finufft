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
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>

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
// throw where the device can host the smallest tile, and none may ask for more
// shared memory than the device grants: bin sizes index that tile, so an over-budget
// pick is a memory bug before it is a slow one. A device too small for the smallest
// tile must reject the pick.
template<typename T> void budget_sweep(const GpuCapabilities &gpu, const char *device) {
  const CUFINUFFT_BIGINT mstu[3] = {128, 128, 128};
  const auto limit               = std::size_t(gpu.max_smem_per_block_optin);

  for (int method = 1; method <= 3; ++method)
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 3; ++type)
        for (int ns = finufft::common::MIN_NSPREAD; ns <= finufft::common::MAX_NSPREAD<T>;
             ++ns) {
          auto o = base_opts(method);
          // smallest tile the pick can land on: method 1 stages no tile, and the
          // method-3 auto pick stages 32 points
          const auto need_min =
              shared_memory_required<T>(dim, ns, 1, 1, 1, method == 3 ? 32 : 0);
          const bool fits = method == 1 || need_min <= limit;
          bool threw      = false;
          try {
            cufinufft_setup_binsize<T>(gpu, type, ns, dim, mstu, &o);
          } catch (const std::exception &e) {
            threw = true;
            // stderr, unbuffered: the assert below aborts before a buffered stdout
            // flushes, and the failing cell is the whole diagnostic.
            fprintf(stderr, "%s: method %d dim %d type %d ns %d threw: %s\n", device,
                    method, dim, type, ns, e.what());
          }
          if (!fits) {
            // too small for the smallest tile: rejection is the correct outcome
            assert(threw && "an unfit device must reject the auto pick");
            continue;
          }
          assert(!threw && "an auto pick must not throw");
          assert(o.gpu_binsizex >= 1);
          assert(o.gpu_binsizey >= 1 && (dim >= 2 || o.gpu_binsizey == 1));
          assert(o.gpu_binsizez >= 1 && (dim >= 3 || o.gpu_binsizez == 1));
          if (method >= 2) {
            const auto need = shared_memory_required<T>(
                dim, ns, o.gpu_binsizex, o.gpu_binsizey, o.gpu_binsizez, o.gpu_np);
            if (need > limit)
              fprintf(stderr,
                      "%s: method %d dim %d type %d ns %d needs %zu of %zu bytes\n",
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

  // The method-3 fill takes each unset dim to at most the valley cap (6 in 3D).
  // The fixed-value cases run only when that worst-case tile fits the device.
  constexpr int kValley3  = 6;
  const std::size_t tile8 = shared_memory_required<T>(3, 7, 8, kValley3, kValley3, 0);
  // per-point cost: the tile formula is linear in np
  const std::size_t per_pt_3_7 =
      shared_memory_required<T>(3, 7, 8, kValley3, kValley3, 1) - tile8;
  { // method 3: explicit (bin, np) is honored verbatim
    const auto need = shared_memory_required<T>(3, 4, 4, 4, 4, 64);
    if (need > limit) {
      fprintf(stderr, "live device: skip explicit (bin, np): needs %zu of %zu bytes\n",
              need, limit);
    } else {
      auto o         = base_opts(3);
      o.gpu_binsizex = o.gpu_binsizey = o.gpu_binsizez = 4;
      o.gpu_np                                         = 64;
      cufinufft_setup_binsize<T>(gpu, 1, 4, 3, mstu, &o);
      assert(o.gpu_binsizex == 4 && o.gpu_binsizey == 4 && o.gpu_binsizez == 4);
      assert(o.gpu_np == 64);
    }
  }
  { // method 3: user np alone is honored, bins derived
    const auto need = shared_memory_required<T>(2, 7, 1, 1, 1, 48); // minimal tile + np
    if (need > limit) {
      fprintf(stderr, "live device: skip user np: needs %zu of %zu bytes\n", need, limit);
    } else {
      auto o   = base_opts(3);
      o.gpu_np = 48;
      cufinufft_setup_binsize<T>(gpu, 1, 7, 2, mstu, &o);
      assert(o.gpu_np == 48 && o.gpu_binsizex >= 1 && o.gpu_binsizey >= 1);
    }
  }
  { // method 3: a partial user bin fills the unset dims
    const std::size_t need = tile8 + 16 * per_pt_3_7; // derived np needs 16 points
    if (need > limit) {
      fprintf(stderr, "live device: skip partial user bin: needs %zu of %zu bytes\n",
              need, limit);
    } else {
      auto o         = base_opts(3);
      o.gpu_binsizex = 8;
      cufinufft_setup_binsize<T>(gpu, 1, 7, 3, mstu, &o);
      assert(o.gpu_binsizex == 8 && o.gpu_binsizey >= 1 && o.gpu_binsizez >= 1);
      assert(o.gpu_np >= 16);
    }
  }
  { // method 3: partial user bin with explicit np — both kept, tile validated
    int np = 0;
    if (limit > tile8) np = int((limit - tile8) / per_pt_3_7); // np from the budget
    if (np > 128) np = 128;
    if (np < 1) {
      fprintf(stderr,
              "live device: skip partial bin with explicit np: needs %zu of %zu bytes\n",
              tile8 + per_pt_3_7, limit);
    } else {
      auto o         = base_opts(3);
      o.gpu_binsizex = 8;
      o.gpu_np       = np;
      cufinufft_setup_binsize<T>(gpu, 1, 7, 3, mstu, &o);
      assert(o.gpu_binsizex == 8 && o.gpu_binsizey >= 1 && o.gpu_binsizez >= 1);
      assert(o.gpu_np == np);
      assert(shared_memory_required<T>(3, 7, o.gpu_binsizex, o.gpu_binsizey,
                                       o.gpu_binsizez, o.gpu_np) <= limit);
    }
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
  { // integer_root at INT_MAX and its neighbour must not overflow int (dim 1 root)
    for (const int msub : {INT_MAX - 1, INT_MAX}) {
      auto o               = base_opts(2);
      o.gpu_maxsubprobsize = msub;
      cufinufft_setup_binsize<T>(gpu, 1, 4, 1, mstu, &o);
      assert(o.gpu_binsizex >= 1);
    }
  }
}

// ---- regression pins: GPU runs at large nf and M ----

using cuc = cuFloatComplex;

// One 1D float type-2 run; returns the M nonuniform values. binsizex 0 leaves
// the bin width to the heuristic.
std::vector<cuc> run_type2_float(int method, int binsizex, int64_t nmodes1,
                                 const std::vector<float> &x,
                                 const std::vector<cuc> &fk) {
  cufinufft_opts o;
  cufinufft_default_opts(&o);
  o.gpu_method            = method;
  o.gpu_binsizex          = binsizex;
  const int64_t nmodes[3] = {nmodes1, 1, 1};
  cufinufftf_plan p{};
  assert(cufinufftf_makeplan(2, 1, nmodes, 1, 1, 1e-4f, &p, &o) == 0);
  const int64_t M = (int64_t)x.size();
  float *dx       = nullptr;
  cuc *dc = nullptr, *dfk = nullptr;
  assert(cudaMalloc(reinterpret_cast<void **>(&dx), x.size() * sizeof(float)) ==
         cudaSuccess);
  assert(
      cudaMalloc(reinterpret_cast<void **>(&dc), x.size() * sizeof(cuc)) == cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void **>(&dfk), fk.size() * sizeof(cuc)) ==
         cudaSuccess);
  assert(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) ==
         cudaSuccess);
  assert(cudaMemcpy(dfk, fk.data(), fk.size() * sizeof(cuc), cudaMemcpyHostToDevice) ==
         cudaSuccess);
  assert(
      cufinufftf_setpts(p, M, dx, nullptr, nullptr, 0, nullptr, nullptr, nullptr) == 0);
  assert(cufinufftf_execute(p, dc, dfk) == 0);
  std::vector<cuc> c(x.size());
  assert(cudaMemcpy(c.data(), dc, x.size() * sizeof(cuc), cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  cudaFree(dx);
  cudaFree(dc);
  cudaFree(dfk);
  assert(cufinufftf_destroy(p) == 0);
  return c;
}

// Pins the integer bin index in compute_bin_index. At nf = 2^25 the old float
// reciprocal-multiply index puts each x below one bin right of its true bin
// (folded coordinate 6*b + 4, 1/6f rounds up), the subprob tile misses support
// cells, and method 2 disagrees with the bin-order-insensitive method-1 run.
void test_bin_index_boundary() {
  const int64_t nmodes       = 1 << 24; // nf = next235(2 * 2^24) = 2^25
  // float bit patterns of x near pi whose fold is rescaled = 6*b + 4
  const uint32_t flip_bits[] = {0x40490C96, 0x40490CFD, 0x40490D61, 0x40490DC8,
                                0x40490E2B, 0x40490E93, 0x40490EFA, 0x40490F5E,
                                0x40490FC5, 0x40490FD8};
  std::vector<float> x;
  for (uint32_t b : flip_bits) {
    float v;
    std::memcpy(&v, &b, sizeof v);
    x.push_back(v);
  }
  for (float v : {0.3f, -1.2f, 2.5f, -2.9f, 1.1f, -0.4f}) x.push_back(v);
  std::vector<cuc> fk(1 << 24);
  uint32_t s = 1;
  for (auto &z : fk) {
    s              = s * 1664525u + 1013904223u;
    const float re = float(s >> 8) / float(1 << 24) - 0.5f;
    s              = s * 1664525u + 1013904223u;
    const float im = float(s >> 8) / float(1 << 24) - 0.5f;
    z              = make_cuFloatComplex(2 * re, 2 * im);
  }
  const auto c1 = run_type2_float(1, 0, nmodes, x, fk); // auto bin: one bin
  const auto c2 = run_type2_float(2, 6, nmodes, x, fk); // bin 6: the flip case
  double num = 0, den = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double dr = (double)cuCrealf(c1[i]) - cuCrealf(c2[i]);
    const double di = (double)cuCimagf(c1[i]) - cuCimagf(c2[i]);
    num += dr * dr + di * di;
    den += (double)cuCrealf(c1[i]) * cuCrealf(c1[i]) +
           (double)cuCimagf(c1[i]) * cuCimagf(c1[i]);
  }
  const double rel_l2 = std::sqrt(num / den);
  fprintf(stderr, "bin-index pin: rel_l2=%.3g (n=%zu)\n", rel_l2, x.size());
  assert(rel_l2 < 1e-5);
}

// ---- invalid opts: rejected at makeplan, before any device work ----

// gpu_maxsubprobsize below 1 divides the subprob counts. It must return
// FINUFFT_ERR_INVALID_ARGUMENT.
void test_invalid_opts() {
  const int64_t nmodes[3] = {8, 1, 1};
  for (const int method : {1, 2, 3}) {
    for (const int msub : {0, -1}) {
      cufinufft_opts o     = base_opts(method);
      o.gpu_maxsubprobsize = msub;
      cufinufftf_plan p{};
      assert(cufinufftf_makeplan(2, 1, nmodes, 1, 1, 1e-4f, &p, &o) ==
             FINUFFT_ERR_INVALID_ARGUMENT);
    }
  }
}

} // namespace

int main() {
  const auto gpu = GpuCapabilities::query(0);
  test_invalid_opts();
  run<float>(gpu);
  run<double>(gpu);
  test_bin_index_boundary();
  printf("cufinufft_heuristics_test: all assertions passed\n");
  return 0;
}

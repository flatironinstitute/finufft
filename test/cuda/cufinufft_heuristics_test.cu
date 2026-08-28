// Host-side checks of cufinufft_setup_binsize: every auto pick stays inside the
// shared-memory budget, explicit user (bin, np) survive verbatim, a partial user
// bin gets its unset dims filled, and the msub / smem roots floor exactly.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>

#include <cufinufft.h>
#include <cufinufft/heuristics.hpp>

using cufinufft::common::cufinufft_setup_binsize;
using cufinufft::common::shared_memory_required;

namespace {

cufinufft_opts base_opts(int method) {
  cufinufft_opts o;
  cufinufft_default_opts(&o);
  o.gpu_method = method;
  o.upsampfac  = 2.0;
  return o;
}

template<typename T> void run(const GpuCapabilities &gpu) {
  const CUFINUFFT_BIGINT mstu[3] = {128, 128, 128};
  const auto limit               = std::size_t(gpu.max_smem_per_block_optin);

  // auto picks: bins set in active dims, 1 elsewhere, tile + np inside the budget
  for (int method = 1; method <= 3; ++method)
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 2; ++type)
        for (int ns : {4, 7, 10}) {
          auto o = base_opts(method);
          cufinufft_setup_binsize<T>(gpu, type, ns, dim, mstu, &o);
          assert(o.gpu_binsizex >= 1);
          assert(o.gpu_binsizey >= 1 && (dim >= 2 || o.gpu_binsizey == 1));
          assert(o.gpu_binsizez >= 1 && (dim >= 3 || o.gpu_binsizez == 1));
          if (method >= 2)
            assert(shared_memory_required<T>(dim, ns, o.gpu_binsizex, o.gpu_binsizey,
                                             o.gpu_binsizez, o.gpu_np) <= limit);
          if (method == 3) assert(o.gpu_np >= 16);
        }

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

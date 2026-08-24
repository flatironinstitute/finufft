/* Tester for the tiled spread and interp: the row stride, the subproblem cut, the two L2
   tile ceilings, when the sort runs, one axis of tiles, that a type-2 plan tiles as its
   type-1 twin, and translation covariance across the periodic boundary. Each is a
   property of the schedule or the wrap, so none of it needs a reference implementation.
   The zero-point spreadinterponly case lives in dumbinputs.cpp.
*/

#include <finufft/simd.hpp>
#include <finufft/spread.hpp>
#include <finufft/test_defs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace {

int fails      = 0;
int test_debug = 0; // FINUFFT_TEST_DEBUG passes through to opts.debug
// A sanitizer build, and a run under valgrind, are one to two orders of magnitude slower
// than the plain build. FINUFFT_TEST_SMALL cuts the grid edges and the point counts.
int test_small = 0; // FINUFFT_TEST_SMALL

void check(bool ok, const char *what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    ++fails;
  }
}

// ------------------------------------------------------------------ synthetic layouts

// Points a tile holds, from its index, the number of axes it sits at the end of, and
// the tile count. Occupancy decides which tiles the cut has to skip; a point set
// aligned to the grid can populate a last row on its own.
using Occupancy = UBIGINT (*)(BIGINT t, int on_last, BIGINT ntiles);

constexpr Occupancy occupancies[]{
    [](BIGINT, int, BIGINT) -> UBIGINT { return 300; },           // every tile
    [](BIGINT, int l, BIGINT) -> UBIGINT { return l ? 0 : 300; }, // interior only
    [](BIGINT, int l, BIGINT) -> UBIGINT { return l ? 300 : 0; }, // last rows only
    [](BIGINT t, int, BIGINT) -> UBIGINT { return t % 3 == 1 ? 0 : 300; }, // interleaved
    [](BIGINT t, int, BIGINT n) -> UBIGINT { return t == n / 2 ? 300 : 0; }, // one tile
};

// Fills the tile offsets from the occupancy, and returns how many populated tiles sit on
// the last row of an axis, so a caller can tell a vacuous case from a real one.
template<typename Count> BIGINT fill(SpreadTileData &tiles, int ndims, Count count) {
  BIGINT ntiles = 1;
  for (int d = 0; d < ndims; ++d) ntiles *= tiles.nt[d];
  BIGINT run = 0, roundoff = 0;
  for (BIGINT t = 0; t < ntiles; ++t) {
    int on_last = 0; // axes on which this tile is the last row
    BIGINT i    = t;
    for (int d = 0; d < ndims; ++d) {
      on_last += i % tiles.nt[d] == tiles.nt[d] - 1;
      i /= tiles.nt[d];
    }
    const UBIGINT n = count(t, on_last, ntiles);
    roundoff += n && on_last;
    run += BIGINT(n);
    tiles.starts[size_t(t) + 1] = run;
  }
  return roundoff;
}

void size_tiles(SpreadTileData &tiles, int ndims) {
  BIGINT ntiles = 1;
  for (int d = 0; d < ndims; ++d) ntiles *= tiles.nt[d];
  tiles.starts.assign(size_t(ntiles) + 1, 0);
}

// A layout of `nt` tiles per axis holding `per` points each. With guard_row the last tile
// of every axis stays empty, as the bin sort's round-off row does; `last` gives the grid
// points in the last populated row, 0 meaning a full row.
SpreadTileData layout(std::array<BIGINT, 3> nt, int edge, UBIGINT per, int ndims,
                      bool guard_row = false, int last = 0) {
  SpreadTileData tiles;
  tiles.edge = edge;
  tiles.nt   = nt;
  for (int d = 0; d < ndims; ++d) { // grid points the populated rows cover
    const BIGINT rows = nt[d] - (guard_row ? 1 : 0);
    tiles.ngrid[d]    = edge * (rows - 1) + (last ? last : edge);
  }
  size_tiles(tiles, ndims);
  fill(tiles, ndims, [&](BIGINT, int on_last, BIGINT) -> UBIGINT {
    return guard_row && on_last ? 0 : per;
  });
  return tiles;
}

// ------------------------------------------------------------------ the subproblem cut

// Subproblems a layout must produce: one per non-empty tile, and one more for every whole
// cap that tile's points run past.
UBIGINT expected_subs(const SpreadTileData &tiles, UBIGINT cap) {
  UBIGINT n = 0;
  for (size_t t = 1; t < tiles.starts.size(); ++t) {
    const UBIGINT occ = UBIGINT(tiles.starts[t]) - UBIGINT(tiles.starts[t - 1]);
    if (occ) n += 1 + (occ - 1) / cap;
  }
  return n;
}

void test_row_stride() {
  // Subgrid::set_size1 must clear the SIMD tail of the innermost store, and must never
  // leave the stride on a whole eight cache lines: such a stride reaches only eight of
  // the L1's 64 sets, and a subgrid walking many rows on it thrashes them.
  const auto walk = [](auto tag) {
    using T                = decltype(tag);
    constexpr BIGINT line  = 64 / BIGINT(2 * sizeof(T));
    constexpr BIGINT alias = 8 * line;
    BIGINT widened         = 0;
    for (uint8_t ns = finufft::common::MIN_NSPREAD; ns <= finufft::common::MAX_NSPREAD<T>;
         ++ns) {
      const BIGINT tail = BIGINT(finufft::spreadinterp::get_padding<T>(2 * ns) / 2);
      for (BIGINT n1 = ns; n1 <= 4096; ++n1) {
        Subgrid sub;
        sub.set_size1<T>(n1, tail);
        check(sub.size1 == n1, "set_size1 keeps the unpadded extent");
        check(sub.padded_size1 >= n1 + tail, "the row stride clears the SIMD tail");
        check(sub.padded_size1 % alias != 0, "the row stride avoids an eight-line alias");
        check(sub.padded_size1 <= n1 + tail + line,
              "the row stride adds at most one line");
        widened += (sub.padded_size1 != n1 + tail);
      }
    }
    // Without this the checks above still pass on a rule that widens nothing.
    check(widened > 0, "the alias rule really widens some strides");
  };
  walk(double{});
  walk(float{});
}

// One subproblem per non-empty tile, and the breakpoints walk the sorted list from its
// start to its end without a gap or an overlap, so every point is spread exactly once.
void test_cut() {
  const int ndims   = 3;
  const UBIGINT per = 200;
  for (const std::array<BIGINT, 3> nt :
       {std::array<BIGINT, 3>{12, 12, 12}, std::array<BIGINT, 3>{13, 13, 13},
        std::array<BIGINT, 3>{3, 3, 3}})
    for (const bool guard_row : {false, true}) {
      const auto tiles = layout(nt, 32, per, ndims, guard_row);
      const auto sched =
          spread_schedule(tiles, UBIGINT(tiles.starts.back()), 1 << 20, 16, 1);
      check(sched.bounds.size() == expected_subs(tiles, sched.points_per_subproblem) + 1,
            "every non-empty tile becomes as many subproblems as the cap needs");
      check(sched.bounds.front() == 0 &&
                sched.bounds.back() == UBIGINT(tiles.starts.back()),
            "the subproblems cover the whole sorted list");
      bool rising = true;
      for (size_t p = 1; p < sched.bounds.size(); ++p)
        rising = rising && sched.bounds[p] > sched.bounds[p - 1];
      check(rising, "the breakpoints rise, so no point falls in two subproblems");
    }

  // Occupancy the layout helper cannot express: a hole in the middle of the ring, and a
  // single populated tile, both of which the cut must skip over rather than split on.
  for (const auto occ : occupancies) {
    auto tiles = layout({8, 8, 8}, 32, per, ndims);
    fill(tiles, ndims, occ);
    const auto sched =
        spread_schedule(tiles, UBIGINT(tiles.starts.back()), 1 << 20, 16, 1);
    check(sched.bounds.size() == expected_subs(tiles, sched.points_per_subproblem) + 1,
          "an empty tile takes no subproblem");
  }
}
// ------------------------------------------------------------------ the tiling rule

// True where even a one-cell tile's padded subgrid leaves L2. The tile then stays one
// cell, so the L2-fit invariants below hold only where this is false.
bool no_tile_fits(int ndims, int ns) {
  return spread_pow_ndims(4.0 + ns, ndims) >
         double(finufft::utils::getL2CacheSize()) / 16;
}

// The rule that sizes a tile, checked against its own invariants so it does not depend
// on the machine's cache size. Every dimension is tiled; a cache too small for even a
// one-cell padded subgrid keeps the tile at one cell.
void test_ceilings() {
  for (const int ndims : {1, 2, 3}) {
    // whatever the density, the chosen tile respects both L2 ceilings
    const int ns = 13; // the width sigma 1.15 picks at eps=1e-6, not the sigma 2 width
    if (no_tile_fits(ndims, ns)) {
      check(spread_tile_doublings(4, ndims, ns, 1.0) == 0,
            "an oversize cell stays a one-cell tile");
      continue;
    }
    for (double dens = 1e-6; dens < 1e3; dens *= 4) {
      const int shift = spread_tile_doublings(4, ndims, ns, dens);
      const double e2 = double(4 << shift);
      check(spread_pow_ndims(e2 + ns, ndims) <=
                double(finufft::utils::getL2CacheSize()) / 16,
            "the tile's padded subgrid fits L2");
      check(shift == 0 ||
                dens * spread_pow_ndims(e2, ndims) <= double(spread_point_budget()),
            "the tile's strengths fit a quarter of L2");
    }
    // density only ever shrinks the tile, and a sparse grid is tiled like a dense one
    int prev_dens = -1;
    for (double dens = 1e-6; dens < 1e3; dens *= 4) {
      const int shift = spread_tile_doublings(4, ndims, ns, dens);
      check(prev_dens < 0 || shift <= prev_dens, "a denser grid never grows the tile");
      prev_dens = shift;
    }
  }
  // Every kernel width the library can pick. ns follows eps and the upsampling factor, so
  // it reaches 16 and beyond at tight tolerances, and the ceiling reads it to bound the
  // padded tile edge. It may not break anywhere on the ladder.
  for (int ndims = 1; ndims <= 3; ++ndims) {
    const double l2_cells = double(finufft::utils::getL2CacheSize()) / 16;
    int prev              = 1 << 30;
    for (int ns = 2; ns <= 24; ++ns) {
      if (no_tile_fits(ndims, ns)) {
        check(spread_tile_doublings(4, ndims, ns, 1e-4) == 0,
              "a tile too big for L2 stays one cell");
        continue;
      }
      const int shift = spread_tile_doublings(4, ndims, ns, 1e-4);
      const double e  = double(4 << shift);
      check(spread_pow_ndims(e + ns, ndims) <= l2_cells,
            "the padded tile fits L2 at every ns");
      check(shift <= prev, "a wider kernel never grows the tile");
      prev = shift;
    }
  }
  // The first width whose padded one-cell subgrid leaves L2. It follows the cache size,
  // so derive it instead of fixing a number: the tile stays one cell from there on.
  for (int ndims = 2; ndims <= 3; ++ndims) {
    int ns = 2;
    while (ns < 1 << 16 && !no_tile_fits(ndims, ns)) ++ns;
    check(spread_tile_doublings(4, ndims, ns, 1.0) == 0,
          "past the L2 width the tile stays one cell");
  }
}

// ------------------------------------------------------------------ real plans

// Uniform points on [-pi, pi) with random strengths, enough to drive the plan-level
// property tests below.
void uniform_points(BIGINT M, int ndims, std::vector<FLT> &x, std::vector<FLT> &y,
                    std::vector<FLT> &z, std::vector<CPX> &c) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<FLT> U(-PI, PI);
  x.assign(size_t(M), 0);
  y.assign(size_t(M), 0);
  z.assign(size_t(M), 0);
  c.assign(size_t(M), CPX(0, 0));
  for (BIGINT j = 0; j < M; ++j) {
    x[size_t(j)] = U(rng);
    if (ndims > 1) y[size_t(j)] = U(rng);
    if (ndims > 2) z[size_t(j)] = U(rng);
    c[size_t(j)] = CPX(U(rng), U(rng));
  }
}

// The sort pays wherever a later pass returns to a fine cell the cache has dropped; one
// thread writing a fine grid inside L2 never does, and that case is the only route left
// to the untiled chunk cut. Pin both directions of the rule.
void test_sort_rule() {
  const BIGINT M     = 20000;
  const auto plan_of = [&](BIGINT N, int nthr, bool past_l2, const char *what) {
    std::vector<FLT> x, y, z;
    std::vector<CPX> c;
    uniform_points(M, 3, x, y, z, c);
    finufft_opts opts;
    FINUFFT_DEFAULT_OPTS(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac        = 2.0;
    opts.nthreads         = nthr;
    std::array<BIGINT, 3> modes{N, N, N};
    const double tol = sizeof(FLT) == 4 ? 1e-3 : 1e-6;
    try {
      FINUFFT_PLAN_T<FLT> plan(1, 3, modes.data(), +1, 1, FLT(tol), &opts);
      if (plan.setpts(M, x.data(), y.data(), z.data(), 0, nullptr, nullptr, nullptr) > 1)
        return;
      // the fine grid this plan really got, which is what the rule is stated over
      const UBIGINT bytes = 2 * sizeof(FLT) * UBIGINT(plan.grid_size());
      const bool fits     = bytes <= UBIGINT(finufft::utils::getL2CacheSize());
      const bool want     = !(nthr == 1 && fits);
      check(fits != past_l2, "the arm reaches the cache regime it is named for");
      char msg[160];
      snprintf(msg, sizeof(msg), "%s: fine grid %lld bytes on T%d %s sorted", what,
               (long long)bytes, nthr, want ? "is" : "is not");
      check(plan.sorted() == want, msg);
      // sorting always tiles: a one-cell tile is the smallest and is always taken
      snprintf(msg, sizeof(msg), "%s: T%d %s tiled", what, nthr, want ? "is" : "is not");
      check(plan.tile_data().empty() == !want, msg);
      printf("\tsort rule: %s on T%d, fine grid %lld bytes: %s, %s\n", what, nthr,
             (long long)bytes, plan.sorted() ? "sorted" : "not sorted",
             plan.tile_data().empty() ? "chunk cut" : "tiled");
    } catch (const std::exception &e) {
      printf("\t3D N=%lld T%d: no plan (%s)\n", (long long)N, nthr, e.what());
    }
  };
  // an edge whose fine grid is twice one core's L2, so the arm stays past L2 on any
  // machine; a fixed edge lands exactly on the L2 boundary in one precision or the other
  const auto past_l2_edge = [] {
    const double cells =
        2.0 * double(finufft::utils::getL2CacheSize()) / double(2 * sizeof(FLT));
    return BIGINT(std::ceil(std::cbrt(cells)));
  };
  plan_of(16, 1, false, "a grid inside L2"); // fine grid 16^3, well under one core's L2
  plan_of(16, 8, false, "a grid inside L2"); // writers revisit what the cache drops
  plan_of(past_l2_edge(), 1, true, "a grid past L2"); // one thread, nothing stays in
                                                      // cache
}

// One dimension tiles like any other: the tile is an interval of the fine grid, which is
// the shape a chunk of the sorted list has there anyway, so the tile arithmetic has to
// hold in 1D too.
void test_one_dim_tiles() {
  const BIGINT M = 200000, N = 1024;
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  uniform_points(M, 1, x, y, z, c);
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spreadinterponly = 1;
  opts.upsampfac        = 2.0;
  opts.nthreads         = 8;
  std::array<BIGINT, 3> modes{N, 1, 1};
  // fp32 cannot reach 1e-6 at upsampfac 2: the rounding floor dominates
  const double tol = sizeof(FLT) == 4 ? 1e-3 : 1e-6;
  try {
    FINUFFT_PLAN_T<FLT> plan(1, 1, modes.data(), +1, 1, FLT(tol), &opts);
    check(plan.setpts(M, x.data(), nullptr, nullptr, 0, nullptr, nullptr, nullptr) <= 1,
          "a 1D plan takes its points");
    check(!plan.tile_data().empty(), "one dimension is tiled");
  } catch (const std::exception &e) {
    printf("\t1D N=%lld: no plan (%s)\n", (long long)N, e.what());
  }
}

// One geometry for both directions: a type-2 plan tiles exactly as the type-1 plan of
// the same problem does, since interpolation reads the padded subgrid that spreading
// writes. Nothing else pins that.
void test_interp_tiles() {
  const BIGINT M = 300000, N = 512;
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  uniform_points(M, 2, x, y, z, c);
  std::array<BIGINT, 3> modes{N, N, 1};
  const double tol    = sizeof(FLT) == 4 ? 1e-3 : 1e-6;
  const auto tiles_of = [&](int type) {
    finufft_opts opts;
    FINUFFT_DEFAULT_OPTS(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac        = 2.0;
    opts.nthreads         = 8;
    FINUFFT_PLAN_T<FLT> plan(type, 2, modes.data(), +1, 1, FLT(tol), &opts);
    check(plan.setpts(M, x.data(), y.data(), nullptr, 0, nullptr, nullptr, nullptr) <= 1,
          "the plan takes its points");
    check(!plan.tile_data().empty(), "both directions are tiled");
    return plan.tile_data();
  };
  try {
    const auto spread = tiles_of(1), interp = tiles_of(2);
    check(spread.edge == interp.edge && spread.nt == interp.nt &&
              spread.starts == interp.starts,
          "a type-2 sort builds the tiles a type-1 sort does");
  } catch (const std::exception &e) {
    printf("\ttype 2 N=%lld: no plan (%s)\n", (long long)N, e.what());
  }
}

// Translation covariance across the periodic boundary: shifting every point by k whole
// cells circularly shifts the spread grid, and interpolation off the shifted grid returns
// the unshifted values. The wrap sits in copy/add_wrapped_subgrid, so one 3D arm covers
// it.
void test_wrap_translation(BIGINT N, BIGINT M, double sigma, double tol) {
  const BIGINT k = N / 2 + 3; // whole cells to shift; the odd 3 breaks symmetry
  const double h = 2 * PI / double(N);
  printf("\twrap translation 3D N=%lld sigma=%.3g tol=%.3g shift %lld cells\n",
         (long long)N, sigma, tol, (long long)k);
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  uniform_points(M, 3, x, y, z, c);
  // squeezed 10 cells clear of both ends, the cluster never wraps, so it is the wrap-free
  // reference for its shifted copy, whose kernel support crosses the boundary
  const auto shift = [&](std::vector<FLT> &p) {
    std::vector<FLT> s(p.size());
    for (size_t j = 0; j < p.size(); ++j) {
      p[j] = FLT(double(p[j]) * (1.0 - 20.0 / double(N)));
      s[j] = FLT(double(p[j]) + double(k) * h); // the library folds it back
    }
    return s;
  };
  const std::vector<FLT> xs = shift(x), ys = shift(y), zs = shift(z);
  const auto ng = size_t(N) * size_t(N) * size_t(N);
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spreadinterponly = 1;
  opts.upsampfac        = sigma;
  opts.nthreads         = 4;
  opts.debug            = test_debug;
  // shifting the coordinates costs O(N*eps) of a cell, amplified by the kernel slope
  const FLT bound       = FLT(300) * FLT(N) * std::numeric_limits<FLT>::epsilon();
  // where the shifted grid reads its unshifted twin
  const auto from       = [&](BIGINT n1, BIGINT n2, BIGINT n3) {
    const auto w = [&](BIGINT n) {
      return size_t(((n - k) % N + N) % N);
    };
    return w(n1) + size_t(N) * (w(n2) + size_t(N) * w(n3));
  };
  const auto m = size_t(M);
  std::vector<CPX> G(ng), Gs(ng), F(ng), Fs(ng), o(m), os(m);
  for (size_t i = 0; i < ng; ++i)
    F[i] = CPX(FLT(std::sin(0.7 * double(i))), FLT(std::cos(0.3 * double(i))));
  for (BIGINT n3 = 0; n3 < N; ++n3)
    for (BIGINT n2 = 0; n2 < N; ++n2)
      for (BIGINT n1 = 0; n1 < N; ++n1)
        Fs[size_t(n1) + size_t(N) * (size_t(n2) + size_t(N) * size_t(n3))] =
            F[from(n1, n2, n3)];
  const auto spread = [&](const std::vector<FLT> &px, const std::vector<FLT> &py,
                          const std::vector<FLT> &pz, std::vector<CPX> &out) {
    return FINUFFT3D1(M, const_cast<FLT *>(px.data()), const_cast<FLT *>(py.data()),
                      const_cast<FLT *>(pz.data()), c.data(), +1, FLT(tol), N, N, N,
                      out.data(), &opts);
  };
  const auto interp = [&](const std::vector<FLT> &px, const std::vector<FLT> &py,
                          const std::vector<FLT> &pz, std::vector<CPX> &grid,
                          std::vector<CPX> &out) {
    return FINUFFT3D2(M, const_cast<FLT *>(px.data()), const_cast<FLT *>(py.data()),
                      const_cast<FLT *>(pz.data()), out.data(), +1, FLT(tol), N, N, N,
                      grid.data(), &opts);
  };
  check(spread(x, y, z, G) <= 1 && spread(xs, ys, zs, Gs) <= 1, "both spreads ran");
  check(interp(x, y, z, F, o) <= 1 && interp(xs, ys, zs, Fs, os) <= 1,
        "both interps ran");
  FLT gworst = 0, gscale = 0;
  for (BIGINT n3 = 0; n3 < N; ++n3)
    for (BIGINT n2 = 0; n2 < N; ++n2)
      for (BIGINT n1 = 0; n1 < N; ++n1) {
        const CPX ref   = G[from(n1, n2, n3)];
        const size_t to = size_t(n1) + size_t(N) * (size_t(n2) + size_t(N) * size_t(n3));
        gworst          = std::max(gworst, std::abs(Gs[to] - ref));
        gscale          = std::max(gscale, std::abs(ref));
      }
  check(gworst / gscale < bound, "spreading shifted points shifts the grid");
  FLT oworst = 0, oscale = 0;
  for (size_t j = 0; j < m; ++j) {
    oworst = std::max(oworst, std::abs(o[j] - os[j]));
    oscale = std::max(oscale, std::abs(o[j]));
  }
  check(oworst / oscale < bound, "interpolating off the shifted grid crosses the wrap");
}

} // namespace

int main() {
  // A crash must not eat the progress made so far: CI captures a pipe, where stdout is
  // fully buffered and a segfault discards everything printed before it.
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (const char *d = std::getenv("FINUFFT_TEST_DEBUG")) test_debug = std::atoi(d);
  if (const char *s = std::getenv("FINUFFT_TEST_SMALL")) test_small = std::atoi(s);
  test_row_stride();
  test_cut();
  test_ceilings();
  test_sort_rule();
  test_one_dim_tiles();
  test_interp_tiles();
  constexpr bool single = std::is_same_v<FLT, float>;
  test_wrap_translation(32, 3000, 2.0, single ? 1e-4 : 1e-9);
  // the widest kernel the precision reaches: the subgrid pad and the wrap span are
  // largest
  if constexpr (single)
    test_wrap_translation(48, 3000, 1.25, 5e-5);
  else
    test_wrap_translation(48, 3000, 1.75, 1e-13);
  if (fails) {
    printf("%d spread schedule check(s) failed\n", fails);
    return 1;
  }
  return 0;
}

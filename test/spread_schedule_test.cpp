/* Tester for the tiled spread and interp: the row stride, the subproblem cut, the two L2
   tile ceilings, when the sort runs, the state one plan carries from setpts to setpts,
   and translation covariance across the periodic boundary. Each is a property of the
   schedule or the wrap that no reference implementation can show, and none of it repeats
   what finufft{1,2,3}d_test, tolsweep and adjointness check against a direct sum. The
   zero-point spreadinterponly case lives in dumbinputs.cpp.
*/

#include <finufft/simd.hpp>
#include <finufft/spread.hpp>
#include <finufft/test_defs.hpp>

#include "utils/norms.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace {

int fails          = 0;
int test_debug     = 0; // FINUFFT_TEST_DEBUG passes through to opts.debug
// A sanitizer build, and a run under valgrind, are one to two orders of magnitude slower
// than the plain build. FINUFFT_TEST_SMALL cuts the grid edges and the point counts.
int test_small     = 0; // FINUFFT_TEST_SMALL
// The churn below draws its point counts, its point sets and the order of its spectral
// scales from this. Fixed, so a failure replays; FINUFFT_TEST_SEED sweeps it by hand.
uint64_t test_seed = 2718281828;

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

// Fills the tile offsets from the occupancy.
template<typename Count> void fill(SpreadTileData &tiles, int ndims, Count count) {
  BIGINT ntiles = 1;
  for (int d = 0; d < ndims; ++d) ntiles *= tiles.nt[d];
  BIGINT run = 0;
  for (BIGINT t = 0; t < ntiles; ++t) {
    int on_last = 0; // axes on which this tile is the last row
    BIGINT i    = t;
    for (int d = 0; d < ndims; ++d) {
      on_last += i % tiles.nt[d] == tiles.nt[d] - 1;
      i /= tiles.nt[d];
    }
    run += BIGINT(count(t, on_last, ntiles));
    tiles.starts[size_t(t) + 1] = run;
  }
}

// A layout of `nt` tiles per axis holding `per` points each. With guard_row the last tile
// of every axis stays empty, as the bin sort's round-off row does.
SpreadTileData layout(std::array<BIGINT, 3> nt, int edge, UBIGINT per, int ndims,
                      bool guard_row = false) {
  SpreadTileData tiles;
  tiles.edge    = edge;
  tiles.nt      = nt;
  BIGINT ntiles = 1;
  for (int d = 0; d < ndims; ++d) { // grid points the populated rows cover
    tiles.ngrid[d] = edge * (nt[d] - (guard_row ? 1 : 0));
    ntiles *= nt[d];
  }
  tiles.starts.assign(size_t(ntiles) + 1, 0);
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
  // Subgrid::cells must hold the widest SIMD access of the last row, and set_row_layout
  // must never leave the stride on a whole eight cache lines: such a stride reaches only
  // eight of the L1's 64 sets, and a subgrid walking many rows on it thrashes them.
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
        sub.size2 = 3;
        sub.size3 = 5;
        sub.set_row_layout<T>(n1, tail);
        check(sub.size1 == n1, "set_row_layout keeps the unpadded extent");
        check(sub.padded_size1 >= n1, "the row stride holds a whole row");
        check(sub.padded_size1 % alias != 0, "the row stride avoids an eight-line alias");
        check(sub.padded_size1 <= n1 + line, "the row stride adds at most one line");
        // The last row starts at padded_size1*(size2*size3-1) and its innermost store
        // reaches size1+tail cells past that start.
        const UBIGINT rows = UBIGINT(sub.size2 * sub.size3);
        check(sub.cells() >= UBIGINT(sub.padded_size1) * (rows - 1) + UBIGINT(n1 + tail),
              "the buffer holds the widest access of the last row");
        // and no more: one tail for the whole buffer, not one per row. Fails on a rule
        // that folds the tail into the stride.
        check(sub.cells() <= UBIGINT(sub.padded_size1) * rows + UBIGINT(tail),
              "the buffer carries one tail, not one per row");
        widened += (sub.padded_size1 != n1);
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
  const int ndims      = 3;
  const UBIGINT per    = 200;
  const auto check_cut = [](const SpreadTileData &tiles, const SpreadSchedule &sched) {
    const auto &b = sched.bounds;
    const auto &s = tiles.starts;
    check(b.size() == expected_subs(tiles, sched.points_per_subproblem) + 1,
          "every non-empty tile becomes as many subproblems as the cap needs");
    check(b.front() == 0 && b.back() == UBIGINT(s.back()),
          "the subproblems cover the whole sorted list");
    check(sched.points_per_subproblem <= spread_point_budget(),
          "the cap stays inside the strengths' quarter-L2 budget");
    bool rising = true, capped = true, aligned = true;
    for (size_t p = 1; p < b.size(); ++p) {
      rising = rising && b[p] > b[p - 1];
      capped = capped && b[p] - b[p - 1] <= sched.points_per_subproblem;
    }
    // Independent of the count formula: shifting a breakpoint across a tile boundary
    // keeps the count but breaks the alignment.
    for (size_t t = 1; t < s.size(); ++t)
      if (s[t] > s[t - 1])
        aligned = aligned && std::binary_search(b.begin(), b.end(), UBIGINT(s[t - 1])) &&
                  std::binary_search(b.begin(), b.end(), UBIGINT(s[t]));
    check(rising, "the breakpoints rise, so no point falls in two subproblems");
    check(capped, "no subproblem exceeds the cap the schedule reports");
    check(aligned, "every non-empty tile begins and ends on a breakpoint");
  };
  for (const std::array<BIGINT, 3> nt :
       {std::array<BIGINT, 3>{12, 12, 12}, std::array<BIGINT, 3>{13, 13, 13},
        std::array<BIGINT, 3>{3, 3, 3}})
    for (const bool guard_row : {false, true}) {
      const auto tiles = layout(nt, 32, per, ndims, guard_row);
      check_cut(tiles,
                spread_schedule(tiles, UBIGINT(tiles.starts.back()), 1 << 20, 16, 1));
    }

  // Occupancy the layout helper cannot express: a hole in the middle of the ring, and a
  // single populated tile, both of which the cut must skip over rather than split on.
  for (const auto occ : occupancies) {
    auto tiles = layout({8, 8, 8}, 32, per, ndims);
    fill(tiles, ndims, occ);
    check_cut(tiles,
              spread_schedule(tiles, UBIGINT(tiles.starts.back()), 1 << 20, 16, 1));
  }
}
// ------------------------------------------------------------------ the tiling rule

// True where even a one-cell tile's padded subgrid leaves L2. The tile then stays one
// cell, so the L2-fit invariants below hold only where this is false.
// Cells the subproblem allocates for a tile of this edge: the stride carries one
// anti-alias line and the buffer one tail, both of which the ceiling must pay for.
double padded_tile_cells(double edge, int ndims, int ns) {
  const double line = 64.0 / double(2 * sizeof(FLT));
  const double tail = double(finufft::spreadinterp::get_padding<FLT>(2 * ns) / 2);
  return (edge + ns + line) * spread_pow_ndims(edge + ns, ndims - 1) + tail;
}

// L2 in the fixed complex-cell unit the ceiling is stated in
double l2_cells() {
  return double(finufft::utils::getL2CacheSize() / spread_bytes_per_point);
}

bool no_tile_fits(int ndims, int ns) {
  return padded_tile_cells(4.0, ndims, ns) > l2_cells();
}

// The rule that sizes a tile, checked against its own invariants so it does not depend
// on the machine's cache size. Every dimension is tiled; a cache too small for even a
// one-cell padded subgrid keeps the tile at one cell.
void test_ceilings() {
  for (const int ndims : {1, 2, 3}) {
    // whatever the density, the chosen tile respects both L2 ceilings
    const int ns = 13; // the width sigma 1.15 picks at eps=1e-6, not the sigma 2 width
    if (no_tile_fits(ndims, ns)) {
      check(spread_tile_doublings<FLT>(4, ndims, ns, 1.0) == 0,
            "an oversize cell stays a one-cell tile");
      continue;
    }
    // density only ever shrinks the tile, and a sparse grid is tiled like a dense one
    int prev = 1 << 30;
    for (double dens = 1e-6; dens < 1e3; dens *= 4) {
      const int shift = spread_tile_doublings<FLT>(4, ndims, ns, dens);
      const double e2 = double(4 << shift);
      check(padded_tile_cells(e2, ndims, ns) <= l2_cells(),
            "the tile's padded subgrid fits L2");
      check(shift == 0 ||
                dens * spread_pow_ndims(e2, ndims) <= double(spread_point_budget()),
            "the tile's strengths fit a quarter of L2");
      check(shift <= prev, "a denser grid never grows the tile");
      prev = shift;
    }
  }
  // Every kernel width the library can pick. ns follows eps and the upsampling factor, so
  // it reaches 16 and beyond at tight tolerances, and the ceiling reads it to bound the
  // padded tile edge. It may not break anywhere on the ladder.
  for (int ndims = 1; ndims <= 3; ++ndims) {
    int prev = 1 << 30;
    for (int ns = 2; ns <= 24; ++ns) {
      if (no_tile_fits(ndims, ns)) {
        check(spread_tile_doublings<FLT>(4, ndims, ns, 1e-4) == 0,
              "a tile too big for L2 stays one cell");
        continue;
      }
      const int shift = spread_tile_doublings<FLT>(4, ndims, ns, 1e-4);
      const double e  = double(4 << shift);
      check(padded_tile_cells(e, ndims, ns) <= l2_cells(),
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
    check(spread_tile_doublings<FLT>(4, ndims, ns, 1.0) == 0,
          "past the L2 width the tile stays one cell");
  }
}

// ------------------------------------------------------------------ real plans

// Uniform 3D points on [-pi, pi) with random strengths, enough to drive the plan-level
// property tests below.
void uniform_points(BIGINT M, std::vector<FLT> &x, std::vector<FLT> &y,
                    std::vector<FLT> &z, std::vector<CPX> &c, uint64_t seed = 12345) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<FLT> U(-PI, PI);
  x.resize(size_t(M));
  y.resize(size_t(M));
  z.resize(size_t(M));
  c.resize(size_t(M));
  for (BIGINT j = 0; j < M; ++j) {
    x[size_t(j)] = U(rng);
    y[size_t(j)] = U(rng);
    z[size_t(j)] = U(rng);
    c[size_t(j)] = CPX(U(rng), U(rng));
  }
}

// What the spread reads out of a plan after setpts, checked against the points and the
// fine grid that plan holds now. A permutation or a layout left over from an earlier
// setpts fails one of these: the coverage checks read the point count of this setpts, and
// the box check reads its fine grid.
void check_setpts_state(const FINUFFT_PLAN_T<FLT> &plan, BIGINT M, const char *what) {
  using finufft::spreadinterp::fold_rescale;
  char msg[192];
  const auto say = [&](const char *s) -> const char * {
    snprintf(msg, sizeof(msg), "%s: %s", what, s);
    return msg;
  };
  const auto &perm  = plan.sort_indices();
  const auto &tiles = plan.tile_data();
  check(BIGINT(perm.size()) >= M, say("the permutation holds every point"));
  // the permutation is a permutation of this setpts' points, so every point is spread
  // once and none of the indices reaches past the arrays this setpts was given
  std::vector<char> hit(size_t(M), 0);
  BIGINT twice = 0, out_of_range = 0, moved = 0;
  for (BIGINT j = 0; j < M; ++j) {
    const BIGINT p = perm[size_t(j)];
    moved += p != j;
    if (p < 0 || p >= M) {
      ++out_of_range;
      continue;
    }
    twice += hit[size_t(p)]++;
  }
  check(out_of_range == 0,
        say("every index of the permutation is a point of this setpts"));
  check(twice == 0, say("no point is spread twice"));
  check(tiles.empty() == !plan.sorted(),
        say("the layout follows the verdict of this setpts"));
  if (!plan.sorted()) {
    check(moved == 0, say("an unsorted setpts leaves the identity permutation"));
    return;
  }
  BIGINT ntiles = 1;
  UBIGINT cells = 1;
  for (int d = 0; d < 3; ++d) {
    ntiles *= tiles.nt[d];
    cells *= UBIGINT(tiles.ngrid[d]);
    check(tiles.nt[d] * tiles.edge >= tiles.ngrid[d],
          say("the tiles cover the fine grid"));
  }
  check(cells == plan.grid_size(), say("the layout tiles the plan's own fine grid"));
  check(BIGINT(tiles.starts.size()) == ntiles + 1,
        say("one offset per tile, and one end"));
  check(tiles.starts.front() == 0 && tiles.starts.back() == M,
        say("the tiles hold the points of this setpts, and no others"));
  bool rising = true;
  for (size_t t = 1; t < tiles.starts.size(); ++t)
    rising = rising && tiles.starts[t] >= tiles.starts[t - 1];
  check(rising, say("the tile offsets never step back"));
  // Every point sits in the box of the tile it was binned into. The sort bins on a
  // reciprocal cell size, so a point within a rounding of a face may bin either side; a
  // point in the wrong tile is a whole tile edge out.
  const auto &XYZ = plan.getXYZ();
  const FLT slack = FLT(tiles.edge) / 100;
  BIGINT outside  = 0;
  for (size_t t = 1; t < tiles.starts.size(); ++t)
    for (BIGINT j = tiles.starts[t - 1]; j < tiles.starts[t]; ++j) {
      BIGINT rest = BIGINT(t) - 1;
      for (int d = 0; d < plan.dim; ++d) {
        const BIGINT row = rest % tiles.nt[d];
        rest /= tiles.nt[d];
        const FLT f = fold_rescale<FLT>(XYZ[d][perm[size_t(j)]], UBIGINT(tiles.ngrid[d]));
        outside +=
            f < FLT(row * tiles.edge) - slack || f > FLT((row + 1) * tiles.edge) + slack;
      }
    }
  check(outside == 0, say("every point sits in the box of the tile it was binned into"));
}

// The sort pays wherever a later pass returns to a fine cell the cache has dropped; one
// thread writing a fine grid inside L2 never does, and that case is the only route left
// to the untiled chunk cut. Pin both directions of the rule.
void test_sort_rule() {
  const BIGINT M     = 20000;
  const auto plan_of = [&](int dim, BIGINT N, int nthr, bool past_l2, const char *what) {
    std::vector<FLT> x, y, z;
    std::vector<CPX> c;
    uniform_points(M, x, y, z, c);
    finufft_opts opts;
    FINUFFT_DEFAULT_OPTS(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac        = 2.0;
    opts.nthreads         = nthr;
    std::array<BIGINT, 3> modes{N, dim > 1 ? N : 1, dim > 2 ? N : 1};
    const double tol = sizeof(FLT) == 4 ? 1e-3 : 1e-6;
    FINUFFT_PLAN_T<FLT> plan(1, dim, modes.data(), +1, 1, FLT(tol), &opts);
    check(plan.setpts(M, x.data(), y.data(), z.data(), 0, nullptr, nullptr, nullptr) <= 1,
          "the sort-rule setpts ran");
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
  };
  // an edge whose fine grid is twice one core's L2, so the arm stays past L2 on any
  // machine; a fixed edge lands exactly on the L2 boundary in one precision or the other
  const auto past_l2_edge = [] {
    const double cells =
        2.0 * double(finufft::utils::getL2CacheSize()) / double(2 * sizeof(FLT));
    return BIGINT(std::ceil(std::cbrt(cells)));
  };
  plan_of(3, 16, 1, false, "a grid inside L2"); // fine grid 16^3, under one core's L2
  plan_of(3, 16, 8, false, "a grid inside L2"); // any multithread run sorts, even in L2
  plan_of(3, past_l2_edge(), 1, true, "a grid past L2"); // one thread, nothing cached
  // sorting must tile in every dimension, not only the 3D grids the arms above use
  plan_of(1, 1024, 8, false, "a 1D grid");
  plan_of(2, 32, 8, false, "a 2D grid");
}

// The layout a plan spreads on is a fact of its last setpts: indexSort clears the layout
// before it decides, and nothing else writes it, so every spread and interp until the
// next setpts reads that one layout. With the sort off it stays empty, and the schedule
// then cuts the point list into equal chunks of one tile spanning the whole fine grid.
void test_unsorted_cut() {
  const BIGINT N   = 64;
  const double tol = sizeof(FLT) == 4 ? 1e-3 : 1e-6;
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spreadinterponly = 1;
  opts.upsampfac        = 2.0;
  opts.spread_sort      = 0;
  opts.debug            = test_debug;
  std::array<BIGINT, 3> modes{N, N, N};
  UBIGINT most_chunks = 0;
  for (const int nthr : {1, 4}) {
    opts.nthreads = nthr;
    FINUFFT_PLAN_T<FLT> plan(1, 3, modes.data(), +1, 1, FLT(tol), &opts);
    // two point sets on one plan: the second setpts must not read the first's layout
    for (const BIGINT M : {BIGINT(20000), BIGINT(3000)}) {
      uniform_points(M, x, y, z, c);
      check(
          plan.setpts(M, x.data(), y.data(), z.data(), 0, nullptr, nullptr, nullptr) <= 1,
          "the unsorted setpts ran");
      check(!plan.sorted(), "spread_sort=0 does not sort");
      check(plan.tile_data().empty(), "an unsorted plan carries no tile layout");
      const auto sched =
          spread_schedule(plan.tile_data(), UBIGINT(M), plan.grid_size(), nthr, 1);
      const UBIGINT cap = sched.points_per_subproblem;
      check(sched.bounds.front() == 0 && sched.bounds.back() == UBIGINT(M),
            "the unsorted cut covers the whole point list");
      check(sched.bounds.size() == 1 + (UBIGINT(M) + cap - 1) / cap,
            "the unsorted cut is one tile, split by the cap alone");
      UBIGINT widest = 0, narrowest = UBIGINT(M);
      for (size_t p = 1; p < sched.bounds.size(); ++p) {
        const UBIGINT gap = sched.bounds[p] - sched.bounds[p - 1];
        widest            = std::max(widest, gap);
        narrowest         = std::min(narrowest, gap);
      }
      check(widest <= cap, "no chunk of the unsorted cut runs past the cap");
      check(widest - narrowest <= 1, "the chunks of one tile are equal to a point");
      most_chunks = std::max(most_chunks, UBIGINT(sched.bounds.size() - 1));
      printf("\tunsorted cut: T%d M=%lld -> %lld chunk(s) of %lld\n", nthr, (long long)M,
             (long long)(sched.bounds.size() - 1), (long long)cap);
    }
  }
  // Without this the checks above hold on a cut that never splits, where equal chunks and
  // a cap are vacuous.
  check(most_chunks > 1, "the unsorted cut really splits on more than one thread");

  // A later setpts can re-size the fine grid the verdict is stated over: type 3 sizes it
  // from the points, and types 1 and 2 on auto upsampfac re-plan when the density moves.
  // A layout left over from a sorted setpts would have the next, unsorted one cut on a
  // tiling of a grid it no longer has. Type 3 reaches the flip on any L2, so walk its
  // spectral scale up to the first sorted verdict and then come back down.
  finufft_opts o3;
  FINUFFT_DEFAULT_OPTS(&o3);
  o3.nthreads = 1; // the only route to the unsorted verdict under the default sort=2
  o3.debug    = test_debug;
  std::array<BIGINT, 3> nk_modes{1, 1, 1}; // unread: type 3 sizes itself from nk
  const BIGINT M = 3000, NK = 3000;
  std::vector<FLT> s, t, u;
  std::vector<CPX> d;
  uniform_points(M, x, y, z, c);
  uniform_points(NK, s, t, u, d);
  FINUFFT_PLAN_T<FLT> p3(3, 3, nk_modes.data(), +1, 1, FLT(tol), &o3);
  const auto setpts_at = [&](double scale) {
    std::vector<FLT> ss(s.size()), tt(t.size()), uu(u.size());
    for (size_t j = 0; j < s.size(); ++j) {
      ss[j] = FLT(double(s[j]) * scale);
      tt[j] = FLT(double(t[j]) * scale);
      uu[j] = FLT(double(u[j]) * scale);
    }
    const int ier =
        p3.setpts(M, x.data(), y.data(), z.data(), NK, ss.data(), tt.data(), uu.data());
    if (ier <= 1) {
      // type 3 bins on its own rescaled copy of the points, which every setpts rewrites
      char what[64];
      snprintf(what, sizeof(what), "type 3 at spectral scale %.3g", scale);
      check_setpts_state(p3, M, what);
      printf("\ttype-3 setpts at spectral scale %.3g: %s, fine grid %lld cells\n", scale,
             p3.sorted() ? "sorted" : "not sorted", (long long)p3.grid_size());
    }
    return ier;
  };
  // The scale the flip needs depends on this core's L2, so grow until the verdict turns
  // rather than fixing a scale, which keeps the widest fine grid as small as it can be.
  bool saw_sorted = false;
  for (double scale = 1.0; scale <= 1e3 && !saw_sorted; scale *= 2.0) {
    if (setpts_at(scale) > 1) break;
    check(p3.tile_data().empty() == !p3.sorted(),
          "the type-3 layout follows the verdict of its setpts");
    saw_sorted = p3.sorted();
  }
  check(saw_sorted, "the walk up the spectral scale reaches the sorted verdict");
  // and back to a fine grid inside L2: the layout the sorted setpts built must be gone,
  // or this plan would spread its next transform on a tiling of the wrong grid
  check(setpts_at(1e-2) <= 1, "the narrow type-3 setpts ran");
  check(!p3.sorted() && p3.tile_data().empty(),
        "an unsorted setpts drops the layout a sorted one left");
}

// ------------------------------------------------------- one plan, many setpts

// A plan is not a fresh plan: every setpts runs on the state the last one left. With the
// upsampfac on auto the point count alone re-plans the fine grid, so one plan crosses the
// sort rule in both directions. After each setpts check the state the plan now holds,
// then the transform it computes against a plan that has seen nothing else: a layout left
// over from an earlier setpts misplaces whole strengths, which no rounding hides. The
// fresh plan itself is what finufft3d_test, tolsweep and adjointness check against a
// direct sum, so a direct sum here would only repeat them one configuration at a time.
void test_setpts_churn() {
  const double tol          = sizeof(FLT) == 4 ? 1e-4 : 1e-9;
  // What thread arrival order may move a type-1 result by: the add back reassociates and
  // the deconvolution amplifies it to a few tol; a stale layout gives O(1).
  // Interp writes each point once whatever the thread count, so type 2 may not move at
  // all beyond a reassociating library's floor.
  const double drift_bound  = 100 * tol;
  const double drift2_bound = sizeof(FLT) == 4 ? 1e-6 : 1e-13;
  const int rounds          = test_small ? 2 : 4;
  std::mt19937_64 rng(test_seed);
  std::vector<FLT> x, y, z;
  std::vector<CPX> c, cj, cjfresh;

  // An edge whose fine grid at sigma 1.25 is half of one core's L2: the sigma the
  // heuristic picks at low density stays inside it, the one it picks at high density does
  // not, and the probe below finds the point count between them.
  const double cells =
      0.5 * double(finufft::utils::getL2CacheSize()) / double(2 * sizeof(FLT));
  const BIGINT N  = BIGINT(std::ceil(std::cbrt(cells) / 1.25));
  const size_t ng = size_t(N) * size_t(N) * size_t(N);
  std::array<BIGINT, 3> modes{N, N, N};
  std::vector<CPX> F(ng), Ffresh(ng), Fin(ng);
  for (size_t i = 0; i < ng; ++i)
    Fin[i] = CPX(FLT(std::sin(0.7 * double(i))), FLT(std::cos(0.3 * double(i))));

  // The point count at which the density re-plans the fine grid past this core's L2 and
  // the rule turns the sort on. It follows the cache size, so probe for it rather than
  // fix a number; setpts alone settles the verdict, so the probe runs no transform.
  const BIGINT Mlo = 300;
  BIGINT Mhi       = 0;
  {
    finufft_opts o;
    FINUFFT_DEFAULT_OPTS(&o);
    o.nthreads = 1; // the only route to the unsorted verdict under the default sort=2
    o.debug    = test_debug;
    FINUFFT_PLAN_T<FLT> probe(1, 3, modes.data(), +1, 1, FLT(tol), &o);
    for (BIGINT M = Mlo; M <= 3000000 && Mhi == 0; M *= 10) {
      uniform_points(M, x, y, z, c, rng());
      check(probe.setpts(M, x.data(), y.data(), z.data(), 0, nullptr, nullptr, nullptr) <=
                1,
            "the probe setpts ran");
      if (probe.sorted()) Mhi = M;
    }
    check(Mhi > 0, "the point count alone reaches the sorted verdict");
  }

  // The thread count and the sort pick the route: one thread under the default sort is
  // the only way to the unsorted chunk cut, spread_sort=0 forces that cut onto four
  // threads, and spread_nthr_atomic=0 forces the atomic add back.
  struct Arm {
    int sort, nthr, atomic;
  };
  static constexpr Arm arms[]{{2, 1, -1}, {2, 4, 0}, {0, 4, -1}};
  int sorted_rounds = 0, unsorted_rounds = 0;
  for (const auto arm : arms) {
    finufft_opts opts;
    FINUFFT_DEFAULT_OPTS(&opts);
    opts.spread_sort        = arm.sort;
    opts.nthreads           = arm.nthr;
    opts.spread_nthr_atomic = arm.atomic;
    opts.debug              = test_debug;
    FINUFFT_PLAN_T<FLT> plan(1, 3, modes.data(), +1, 1, FLT(tol), &opts);
    // interp reads the layout that this sort built, so a type-2 twin takes every point
    // set the type-1 plan takes
    FINUFFT_PLAN_T<FLT> plan2(2, 3, modes.data(), +1, 1, FLT(tol), &opts);
    // either side of the flip, shuffled, so no round can rely on the one before it
    std::vector<BIGINT> counts;
    for (int r = 0; r < rounds; ++r) {
      counts.push_back(Mlo);
      counts.push_back(Mhi);
    }
    std::shuffle(counts.begin(), counts.end(), rng);
    for (size_t r = 0; r < counts.size(); ++r) {
      const BIGINT M = counts[r];
      char what[112], msg[224];
      snprintf(what, sizeof(what), "spread_sort=%d T%d atomic=%d round %d M=%lld",
               arm.sort, arm.nthr, arm.atomic, int(r), (long long)M);
      uniform_points(M, x, y, z, c, rng());
      const auto setpts = [&](FINUFFT_PLAN_T<FLT> &p) {
        return p.setpts(M, x.data(), y.data(), z.data(), 0, nullptr, nullptr, nullptr);
      };
      check(setpts(plan) <= 1, "the churned type-1 setpts ran");
      check_setpts_state(plan, M, what);
      check(plan.execute(c.data(), F.data()) <= 1, "the churned type-1 execute ran");
      FINUFFT_PLAN_T<FLT> fresh(1, 3, modes.data(), +1, 1, FLT(tol), &opts);
      check(setpts(fresh) <= 1, "the fresh type-1 setpts ran");
      check(fresh.execute(c.data(), Ffresh.data()) <= 1, "the fresh type-1 execute ran");
      const FLT drift = relerrtwonorm(BIGINT(ng), Ffresh.data(), F.data());
      snprintf(msg, sizeof(msg), "type 1 %s: drift from a fresh plan is %.3g", what,
               double(drift));
      check(double(drift) <= drift_bound, msg);

      cj.assign(size_t(M), CPX(0, 0));
      cjfresh.assign(size_t(M), CPX(0, 0));
      check(setpts(plan2) <= 1, "the churned type-2 setpts ran");
      check_setpts_state(plan2, M, what);
      check(plan2.execute(cj.data(), Fin.data()) <= 1, "the churned type-2 execute ran");
      FINUFFT_PLAN_T<FLT> fresh2(2, 3, modes.data(), +1, 1, FLT(tol), &opts);
      check(setpts(fresh2) <= 1, "the fresh type-2 setpts ran");
      check(fresh2.execute(cjfresh.data(), Fin.data()) <= 1,
            "the fresh type-2 execute ran");
      const FLT drift2 = relerrtwonorm(M, cjfresh.data(), cj.data());
      snprintf(msg, sizeof(msg), "type 2 %s: drift from a fresh plan is %.3g", what,
               double(drift2));
      check(double(drift2) <= drift2_bound, msg);

      sorted_rounds += plan.sorted();
      unsorted_rounds += !plan.sorted();
      printf("\tchurn %s: %s, sigma %.3g, %lld fine cells, drift %.3g / %.3g\n", what,
             plan.sorted() ? "sorted" : "not sorted", plan.opts.upsampfac,
             (long long)plan.grid_size(), double(drift), double(drift2));
    }
  }
  // Without both, the sequence never crosses the rule and the churn proves nothing about
  // a layout outliving the grid it was built on.
  check(sorted_rounds > 0 && unsorted_rounds > 0,
        "the point count alone takes one plan across the sort rule both ways");
}
// Translation covariance across the periodic boundary: shifting every point by k whole
// cells circularly shifts the spread grid, and interpolation off the shifted grid returns
// the unshifted values. The wrap sits in copy/drain_wrapped_subgrid, so one 3D arm covers
// it.
void test_wrap_translation(BIGINT N, BIGINT M, double sigma, double tol) {
  const BIGINT k = N / 2 + 3; // whole cells to shift; the odd 3 breaks symmetry
  const double h = 2 * PI / double(N);
  printf("\twrap translation 3D N=%lld sigma=%.3g tol=%.3g shift %lld cells\n",
         (long long)N, sigma, tol, (long long)k);
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  uniform_points(M, x, y, z, c);
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
  if (const char *s = std::getenv("FINUFFT_TEST_SEED"))
    test_seed = std::strtoull(s, nullptr, 10);
  test_row_stride();
  test_cut();
  test_ceilings();
  test_sort_rule();
  test_unsorted_cut();
  test_setpts_churn();
  constexpr bool single = std::is_same_v<FLT, float>;
  test_wrap_translation(32, 3000, 2.0, single ? 1e-4 : 1e-9);
  // the widest kernel the precision reaches: the subgrid pad and the wrap span are
  // largest
  test_wrap_translation(48, 3000, single ? 1.25 : 1.75, single ? 5e-5 : 1e-13);
  if (fails) {
    printf("%d spread schedule check(s) failed\n", fails);
    return 1;
  }
  return 0;
}

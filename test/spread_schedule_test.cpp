/* Tester for the tiled spread: how the sorted points split into subproblems, how a tile
   is sized, and whether the two directions agree on the geometry and on the values.

   Spreading adds a padded subgrid back into the fine grid, so two subproblems whose halos
   overlap would lose an update if they added at once. One lock per fine grid is what
   keeps them apart, and a lost update shows up as a spread grid that stops matching the
   single-threaded one. Interpolation only reads the grid, so it needs no lock; what it
   can get wrong is the block it reads, and a wrong block shows up as strengths that stop
   matching the untiled route.

   The tests are:
   1) the subproblems cover every point once, so nothing is spread twice or dropped, and a
   tile denser than the cap splits into subproblems that each stay under it;
   2) the two L2 ceilings that size a tile, and the kernel width at which no tile fits, in
   every dimension;
   3) the sort runs unless one thread already holds the fine grid in L2, which is the only
   route left to the untiled chunk cut;
   4) one axis of tiles behaves like the general case, where the ring is the whole grid;
   5) a type-2 plan tiles exactly as the type-1 plan of the same problem does, since one
   geometry now serves both directions;
   6) the spread grid of an adversarial point set is the same on many threads, and on many
   vectors of one batch, as on one, which is what a missed collision would break;
   7) the interpolated strengths of the same point sets are the same through the gathered
   block, through a tile too sparse for the gather, and through the untiled grid read
   straight.
*/

#include <finufft/simd.hpp>
#include <finufft/spread_schedule.hpp>
#include <finufft/test_defs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace {

int fails             = 0;
int test_debug        = 0; // FINUFFT_TEST_DEBUG passes through to opts.debug
// A sanitizer build, and a run under valgrind, are one to two orders of magnitude slower
// than the plain build, and the full sweep runs past ctest's timeout there.
// FINUFFT_TEST_SMALL cuts the grid edges, the point counts and the thread sweep. The tile
// decision reads the fine grid against L2, and the small grids still clear L2, so every
// route the sweep covers stays covered.
int test_small          = 0; // FINUFFT_TEST_SMALL
BIGINT grid_checks    = 0; // grid comparisons made, so a silent early return cannot pass
BIGINT interp_checks  = 0; // strength comparisons made, for the same reason
// Worst comparison seen, as a fraction of the tolerance it was judged against: a test
// that passes with no margin left is a test about to fail on another machine.
double worst_grid_rel = 0;
double worst_interp_rel = 0;
void check(bool ok, const char *what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    ++fails;
  }
}

// ------------------------------------------------------------------ synthetic layouts

// Points a tile holds, from its index, the number of axes it sits at the end of, and the
// tile count. Occupancy decides which tiles the cut has to skip, and a distribution can
// populate a last row on its own: a set aligned to the grid puts a whole plane at the top
// of an axis.
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

// ------------------------------------------------------------------ real point sets

// Coordinate sets that put points where the periodic argument is sharpest. `Boundary` and
// `Corners` aim at the round-off cell, `Outside` folds in from several periods away,
// `Lattice` puts a plane of points on every grid line, and `Edges` lands points exactly
// on tile boundaries.
enum class Pts { Uniform, Boundary, Corners, Outside, Lattice, Edges, Clustered, One };
constexpr const char *pts_name[]{"uniform", "boundary", "corners",   "outside",
                                 "lattice", "edges",    "clustered", "one point"};

void make_points(Pts kind, BIGINT M, BIGINT N, int ndims, std::vector<FLT> &x,
                 std::vector<FLT> &y, std::vector<FLT> &z, std::vector<CPX> &c) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<FLT> U(-PI, PI);
  std::uniform_int_distribution<int> P(-3, 3);
  x.assign(size_t(M), 0);
  y.assign(size_t(M), 0);
  z.assign(size_t(M), 0);
  c.assign(size_t(M), CPX(0, 0));
  const FLT top = PI, bot = -PI; // the two ends of the range
  const auto lat = [&](BIGINT k) {
    return FLT(-PI + 2 * PI * FLT(k % N) / FLT(N));
  };
  for (BIGINT j = 0; j < M; ++j) {
    std::array<FLT, 3> p{U(rng), U(rng), U(rng)};
    switch (kind) {
    case Pts::Uniform:
      break;
    case Pts::Boundary: // a third of the points sit exactly on the top of every axis
      if (j % 3 == 0) p = {top, top, top};
      break;
    case Pts::Corners: // every combination of the two ends of the range
      for (int d = 0; d < 3; ++d) p[d] = (j >> d) & 1 ? top : bot;
      break;
    case Pts::Outside: // the same points, moved whole periods away before folding
      for (int d = 0; d < 3; ++d) p[d] += FLT(2 * PI * P(rng));
      if (j % 101 == 0) p[0] = FLT(1e7);
      break;
    case Pts::Lattice:
      p = {lat(j), lat(j / N), lat(j / (N * N))};
      break;
    case Pts::Edges: // exactly on a grid line every 8 fine cells, tile edges included
      for (int d = 0; d < 3; ++d)
        p[d] = FLT(-PI + 2 * PI * FLT(((j >> (3 * d)) * 8) % (2 * N)) / FLT(2 * N));
      break;
    case Pts::Clustered: // one dense blob plus the top of the range
      for (int d = 0; d < 3; ++d) p[d] = j % 7 ? p[d] / FLT(64) : top;
      break;
    case Pts::One:
      p = {top, top, top};
      break;
    }
    x[size_t(j)] = p[0];
    if (ndims > 1) y[size_t(j)] = p[1];
    if (ndims > 2) z[size_t(j)] = p[2];
    c[size_t(j)] = CPX(U(rng), U(rng));
  }
}

// One grid edge per dimension, so the total grid stays comparable across dimensions. The
// 1D edge is the largest each precision accepts: a 1D line puts the whole grid on one
// axis, and check_sigma judges an fp32 tolerance from that fine grid and throws out of
// setpts.
BIGINT grid_edge(int ndims) {
  if (ndims == 1) return sizeof(FLT) == 4 ? 2048 : 65536;
  if (ndims == 2) return test_small ? 512 : 2048;
  return test_small ? 64 : 192;
}

// A small run keeps the ends of a thread sweep: T1 isolates the decomposition from the
// threading, and the widest thread count is where a collision has the most room.
bool skip_threads(int nthr) { return test_small && nthr != 1 && nthr != 16; }

BIGINT npts(BIGINT M) { return test_small ? M / 8 : M; }

// A missed collision loses an update, so the spread grid stops matching. The reference is
// the UNSORTED route: it cuts the point list into one chunk per thread instead of tiling,
// so its subgrids sit at different offsets and a systematic fault in the wrapped add back
// lands in different cells there than under tiling. A tiled-versus-tiled comparison
// cannot see such a fault, since both arms carry it identically. batchSize is
// min(maxbatchsize, ntrans), so several vectors only land inside one pass when ntrans
// exceeds one; every vector carries the same strengths, so each of the ntrans grids must
// equal the reference.
void check_grid_matches(Pts kind, BIGINT M, BIGINT N, int ndims, double tol,
                        double sigma = 2.0) {
  printf("\tspread %dD N=%lld M=%lld %s\n", ndims, (long long)N, (long long)M,
         pts_name[int(kind)]);
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  make_points(kind, M, N, ndims, x, y, z, c);
  BIGINT ng = 1;
  for (int d = 0; d < ndims; ++d) ng *= N;
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spreadinterponly = 1;
  opts.debug = opts.spread_debug = test_debug;
  opts.upsampfac                 = sigma;
  std::vector<CPX> cm; // the strengths repeated once per vector
  const auto run = [&](int nthr, int ntrans, int sort, std::vector<CPX> &F) {
    printf("\t  T%d ntrans%d sort%d\n", nthr, ntrans, sort);
    opts.nthreads     = nthr;
    opts.maxbatchsize = ntrans;
    opts.spread_sort  = sort;
    cm.resize(size_t(M) * ntrans);
    for (int t = 0; t < ntrans; ++t)
      std::copy(c.begin(), c.end(), cm.begin() + size_t(t) * M);
    F.assign(size_t(ng) * ntrans, CPX(0, 0));
    if (ndims == 1)
      return FINUFFT1D1MANY(ntrans, M, x.data(), cm.data(), +1, FLT(tol), N, F.data(),
                            &opts);
    if (ndims == 2)
      return FINUFFT2D1MANY(ntrans, M, x.data(), y.data(), cm.data(), +1, FLT(tol), N, N,
                            F.data(), &opts);
    return FINUFFT3D1MANY(ntrans, M, x.data(), y.data(), z.data(), cm.data(), +1,
                          FLT(tol), N, N, N, F.data(), &opts);
  };
  std::vector<CPX> F1, Fn;
  // A refused plan fails here rather than skipping, so a case that stops reaching the
  // route cannot pass in silence.
  const int ref_ier = run(1, 1, 0, F1);
  check(ref_ier <= 1, "the untiled reference spread ran");
  if (ref_ier > 1) return;
  FLT scale = 0;
  for (const auto v : F1) scale = std::max(scale, std::abs(v));
  check(scale > 0, "the untiled route spread something to compare against");
  if (scale == 0) return;
  const auto compare = [&](int nthr, int ntrans) {
    const int ier = run(nthr, ntrans, 2, Fn);
    check(ier <= 1, "the tiled spread ran");
    if (ier > 1) return;
    for (int t = 0; t < ntrans; ++t) {
      FLT worst = 0;
      for (BIGINT g = 0; g < ng; ++g)
        worst = std::max(worst, std::abs(F1[g] - Fn[size_t(t) * ng + g]));
      const FLT rel = worst / scale;
      char what[160];
      snprintf(what, sizeof(what),
               "%dD N=%lld %s T%d ntrans %d vector %d: the tiled spread grid moved "
               "away from the untiled route",
               ndims, (long long)N, pts_name[int(kind)], nthr, ntrans, t);
      ++grid_checks;
      // Both routes sum the same terms into a cell, in different groups: a cache tile
      // splits at the point budget, the untiled whole-grid tile at one piece per thread.
      // Reordering M terms
      // costs about eps*sqrt(M) relative, which a fixed tolerance cannot cover across the
      // precisions. A subproblem the lock let slip would be a percent of the cell, since
      // a subproblem holds thousands of the points, so nothing real hides under this.
      const FLT bound = FLT(8) * std::numeric_limits<FLT>::epsilon() * std::sqrt(FLT(M));
      worst_grid_rel  = std::max(worst_grid_rel, double(rel / bound));
      const bool near = rel < bound;
      if (!near)
        printf("\trelative difference %.3g against %.3g\n", (double)rel, (double)bound);
      check(near, what);
    }
  };
  // T1 first: it is tiled against an untiled reference, so it isolates the decomposition
  // from the threading. Anything it shows is not a collision.
  for (const int nthr : {1, 2, 3, 4, 8, 16})
    if (!skip_threads(nthr)) compare(nthr, 1);
  // ntrans>1 is what puts several vectors inside one pass, so the (vector, subproblem)
  // collapse and the per-vector guard get exercised. It spreads ntrans times per call, so
  // it runs on fewer thread counts.
  for (const int nthr : {2, 8, 16})
    if (!skip_threads(nthr)) compare(nthr, 3);
}

// Interpolation reads the fine grid, so no thread count can lose an update; what a wrong
// block would lose is the values themselves. Three routes meet here - a tile whose block
// the gather copies out, a tile the gather cannot pay for, and the unsorted grid read
// straight - and sort=0 pins the last one as the reference. A point whose support crosses
// the grid edge sums the wrap in a different order from the block, so the routes agree to
// rounding rather than to the bit; anything the block gets wrong is O(1) against that.
void check_interp_matches(Pts kind, BIGINT M, BIGINT N, int ndims, double tol) {
  printf("\tinterp %dD N=%lld M=%lld %s\n", ndims, (long long)N, (long long)M,
         pts_name[int(kind)]);
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  make_points(kind, M, N, ndims, x, y, z, c);
  BIGINT ng = 1;
  for (int d = 0; d < ndims; ++d) ng *= N;
  std::vector<CPX> F(size_t(ng), CPX(0, 0));
  for (BIGINT g = 0; g < ng; ++g) // a grid no symmetry can cancel
    F[size_t(g)] = CPX(FLT(std::sin(0.7 * double(g))), FLT(std::cos(0.3 * double(g))));
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spreadinterponly = 1;
  opts.debug = opts.spread_debug = test_debug;
  opts.upsampfac                 = 2.0;
  std::vector<CPX> ref(size_t(M), CPX(0, 0)), got(size_t(M), CPX(0, 0));
  const auto run = [&](int nthr, int sort, std::vector<CPX> &out) {
    printf("\t  T%d sort%d\n", nthr, sort);
    opts.nthreads    = nthr;
    opts.spread_sort = sort;
    out.assign(size_t(M), CPX(0, 0));
    if (ndims == 1)
      return FINUFFT1D2(M, x.data(), out.data(), +1, FLT(tol), N, F.data(), &opts);
    if (ndims == 2)
      return FINUFFT2D2(M, x.data(), y.data(), out.data(), +1, FLT(tol), N, N, F.data(),
                        &opts);
    return FINUFFT3D2(M, x.data(), y.data(), z.data(), out.data(), +1, FLT(tol), N, N, N,
                      F.data(), &opts);
  };
  // The untiled route, read straight off the fine grid. A refused plan fails here rather
  // than skipping: a silent return once hid every 1D fp32 case behind check_sigma's
  // fp32 tolerance floor, and the case count did not move.
  const int ref_ier = run(1, 0, ref);
  check(ref_ier <= 1, "the untiled reference transform ran");
  if (ref_ier > 1) return;
  FLT scale = 0;
  for (const auto v : ref) scale = std::max(scale, std::abs(v));
  check(scale > 0, "the untiled route interpolated something to compare against");
  if (scale == 0) return;
  for (const int nthr : {1, 2, 8, 16}) {
    if (skip_threads(nthr)) continue;
    const int ier = run(nthr, 2, got);
    check(ier <= 1, "the tiled transform ran");
    if (ier > 1) return;
    FLT worst = 0;
    for (BIGINT j = 0; j < M; ++j)
      worst = std::max(worst, std::abs(got[size_t(j)] - ref[size_t(j)]));
    const FLT rel = worst / scale;
    char what[160];
    snprintf(what, sizeof(what),
             "%dD N=%lld %s T%d: the interpolated strengths moved away from the untiled "
             "route",
             ndims, (long long)N, pts_name[int(kind)], nthr);
    ++interp_checks;
    const FLT bound  = sizeof(FLT) == 4 ? FLT(2e-5) : FLT(1e-12);
    worst_interp_rel = std::max(worst_interp_rel, double(rel / bound));
    const bool near  = rel < bound;
    if (!near)
      printf("\trelative difference %.3g against %.3g\n", (double)rel, (double)bound);
    check(near, what);
  }
}

// The sort pays wherever a later pass returns to a fine cell the cache has dropped, and
// one thread writing a fine grid inside L2 is the one case it never does. That case is
// the only route left to the untiled chunk cut, so pin both directions of the rule: the
// small single-thread grid skips the sort, and the same grid on many threads, or a grid
// too big for L2 on one thread, takes it and tiles.
void test_sort_rule() {
  const BIGINT M     = 20000;
  const auto plan_of = [&](BIGINT N, int nthr, const char *what) {
    std::vector<FLT> x, y, z;
    std::vector<CPX> c;
    make_points(Pts::Uniform, M, N, 3, x, y, z, c);
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
  plan_of(16, 1, "a grid inside L2"); // fine grid 32^3, well under one core's L2
  plan_of(16, 8, "a grid inside L2"); // several writers revisit what the cache drops
  plan_of(64, 1, "a grid past L2");   // one thread, but nothing stays in cache
}

// One dimension tiles like any other: the tile is an interval of the fine grid, which is
// the shape a chunk of the sorted list has there anyway, so the tile arithmetic has to
// hold in 1D too.
void test_one_dim_tiles() {
  const BIGINT M = 200000, N = 1024;
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  make_points(Pts::Boundary, M, N, 1, x, y, z, c);
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

// One geometry for both directions: a type-2 plan tiles exactly as the type-1 plan of the
// same problem does, since interpolation reads the same padded subgrid that spreading
// writes. Nothing else pins that, and an untiled interp would read the fine grid
// straight.
void test_interp_tiles() {
  const BIGINT M = 300000, N = 512;
  std::vector<FLT> x, y, z;
  std::vector<CPX> c;
  make_points(Pts::Uniform, M, N, 2, x, y, z, c);
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

// Spreading no points must leave a zero grid, not the caller's buffer. The driver returns
// before the subproblem loop when M is zero, so the zeroing has to happen ahead of it.
void test_no_points() {
  const BIGINT N = 64;
  std::vector<CPX> F(size_t(N) * N * N, CPX(1, 1)); // a marker the transform must erase
  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.spreadinterponly = 1;
  opts.upsampfac        = 2.0;
  opts.nthreads         = 4;
  const double tol      = sizeof(FLT) == 4 ? 1e-3 : 1e-6;
  const int ier = FINUFFT3D1(0, nullptr, nullptr, nullptr, nullptr, +1, FLT(tol), N, N, N,
                             F.data(), &opts);
  check(ier <= 1, "a transform of no points runs");
  if (ier > 1) return;
  FLT worst = 0;
  for (const auto v : F) worst = std::max(worst, std::abs(v));
  check(worst == FLT(0), "a transform of no points leaves a zero grid");
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
  test_no_points();

  // The spread grid, on point sets that put points where the periodic arithmetic is
  // sharpest. One grid size and one tolerance per dimension keeps this in a few seconds.
  // The last case is dense enough to split its tiles on almost any cache.
  // 1D spreads a line, where the kernel box is one row and the plane weight is 1, so it
  // takes its own branch through the subproblem kernel. Its grid edge is the largest the
  // precision accepts, as in the interpolation loop below.
  for (const int ndims : {1, 2, 3}) {
    for (const auto kind : {Pts::Uniform, Pts::Boundary, Pts::Corners, Pts::Outside,
                            Pts::Lattice, Pts::Edges, Pts::Clustered, Pts::One})
      check_grid_matches(kind, npts(ndims == 3 ? 300000 : 400000), grid_edge(ndims),
                         ndims, sizeof(FLT) == 4 ? 1e-4 : 1e-9);
  }
  check_grid_matches(Pts::Uniform, npts(4000000), test_small ? 256 : 1024, 2,
                     sizeof(FLT) == 4 ? 1e-4 : 1e-6);

  // The widest kernel the precision reaches. A wide kernel needs a tight tolerance at a
  // low upsampling factor, and fp32's rounding floor refuses most of those. The halo the
  // lock has to cover is largest there, so the collision has the most room to happen.
  {
    const double sigma = sizeof(FLT) == 4 ? 1.25 : 1.75;
    const double tol   = sizeof(FLT) == 4 ? 5e-5 : 1e-13;
    check_grid_matches(Pts::Boundary, npts(300000), grid_edge(3), 3, tol, sigma);
  }

  // The other direction, on the same point sets. The dense cases put every route through
  // the gather, and the sparse ones are below the size a block has to earn, so the same
  // grids also cover the route that reads the fine grid inside a tiled plan.
  for (const int ndims : {1, 2, 3}) {
    const BIGINT dense = npts(ndims == 3 ? 300000 : 400000);
    for (const auto kind : {Pts::Uniform, Pts::Boundary, Pts::Corners, Pts::Outside,
                            Pts::Lattice, Pts::Edges, Pts::Clustered, Pts::One})
      for (const BIGINT M : {dense, BIGINT(2000)})
        check_interp_matches(kind, M, grid_edge(ndims), ndims,
                             sizeof(FLT) == 4 ? 1e-4 : 1e-9);
  }

  printf("\tgrid comparisons across thread counts and batch sizes: %lld\n",
         (long long)grid_checks);
  printf("worst grid %.2f and strength %.2f of the tolerance allowed\n", worst_grid_rel,
         worst_interp_rel);
  check(grid_checks > 50, "the spread grid was really compared across thread counts");
  printf("\tinterpolated strength comparisons across routes: %lld\n",
         (long long)interp_checks);
  check(interp_checks > 50, "the strengths were really compared across routes");
  if (fails) {
    printf("%d spread schedule check(s) failed\n", fails);
    return 1;
  }
  return 0;
}

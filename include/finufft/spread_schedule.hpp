// Tiled spreading policy: how big a cache tile is and how the sorted points split into
// subproblems. All of it is a pure function of the tile layout, the point count and the
// thread geometry, so it pulls in no SIMD or plan machinery and a test can check it on a
// synthetic layout.

#pragma once

#include <finufft/plan.hpp>
#include <finufft/utils.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

// Subproblems one thread draws, so a thread that takes a long one still ends within 1/K
// of the ideal makespan.
constexpr UBIGINT spread_subproblems_per_thread = 4;

// Points whose strengths fill a quarter of one core's L2: the budget the tile sizer aims
// at and the cap a single subproblem may hold. The 16 bytes an element takes here is a
// fixed count, not sizeof: fp32 does better on the smaller tile the same count gives it
// than on the wider one its own element size would allow.
inline UBIGINT spread_point_budget() noexcept {
  return UBIGINT(finufft::utils::getL2CacheSize()) / (4 * 16);
}

// x to the power of the dimension.
inline double spread_pow_ndims(double x, int ndims) noexcept {
  double v = 1;
  for (int d = 0; d < ndims; ++d) v *= x;
  return v;
}

// How the spread splits the points into subproblems. Pure function of the tile layout and
// the thread geometry, so a test can check the schedule without running a transform.
struct SpreadSchedule {
  std::vector<UBIGINT> bounds;       // NU index breakpoints, one subproblem per gap
  UBIGINT points_per_subproblem = 0; // points one subproblem may hold
};

inline SpreadSchedule spread_schedule(const SpreadTileData &tiles, UBIGINT M,
                                      UBIGINT grid_cells, int nthr, int batchSize) {
  // Equal pieces of the point list, one piece at a time out of each tile: the pieces of a
  // tile share that tile's box, so a tile over the cap becomes that many subproblems of
  // the same box rather than one that holds a thread for several tiles' worth of work.
  const auto cut = [](const std::vector<BIGINT> &starts, UBIGINT cap) {
    SpreadSchedule sched;
    sched.points_per_subproblem = cap;
    sched.bounds.push_back(0);
    for (size_t t = 1; t < starts.size(); ++t) {
      const UBIGINT lo = sched.bounds.back(), hi = UBIGINT(starts[t]);
      if (hi <= lo) continue; // empty tile
      const UBIGINT pieces = 1 + (hi - lo - 1) / cap;
      for (UBIGINT c = 1; c < pieces; ++c)
        sched.bounds.push_back(lo + (hi - lo) * c / pieces);
      sched.bounds.push_back(hi);
    }
    return sched;
  };
  // The threads want K subproblems each; one vector of the batch per thread already fills
  // the machine, so the count divides by the threads sharing a vector.
  const UBIGINT threads_per_vector = (UBIGINT(nthr) + batchSize - 1) / batchSize;
  // An unsorted point list is one tile, and that tile is the whole fine grid: its box
  // spans the grid whatever the point count. A subproblem there pays its whole box to
  // zero and add back and saves the gather of its own points staying in L2, so it holds
  // the larger of the point budget and its own cell count, and never more than a thread's
  // share of the points. A grid smaller than the budget is cut by the budget, a grid far
  // larger is cut once per thread.
  if (tiles.starts.size() < 2) {
    const UBIGINT cap = std::min((M + threads_per_vector - 1) / threads_per_vector,
                                 std::max(spread_point_budget(), grid_cells));
    return cut({0, BIGINT(M)}, std::max(cap, UBIGINT(1)));
  }

  // One subproblem per non-empty cache tile, read straight off the tile offsets. A cache
  // tile's padded subgrid fits L2 by construction, so two cache ceilings cap it: twice
  // what an average filled tile holds, and the strengths one core's L2 holds.
  const auto &starts   = tiles.starts;
  UBIGINT filled_tiles = 0;                          // tiles holding at least one point
  for (size_t t = 1; t < starts.size(); ++t) filled_tiles += starts[t] > starts[t - 1];
  filled_tiles = std::max(filled_tiles, UBIGINT(1)); // an empty layout must not divide by
                                                     // 0
  UBIGINT cap  = std::min(2 * M / filled_tiles, spread_point_budget());
  // Lower the cap only when the filled tiles cannot supply the K subproblems a thread
  // wants, since a cap near the occupancy a tile typically has would split half the tiles
  // for nothing.
  const UBIGINT wanted = spread_subproblems_per_thread * threads_per_vector;
  if (filled_tiles < wanted) cap = std::min(cap, 1 + (M - 1) / wanted);
  return cut(starts, std::max(cap, UBIGINT(1)));
}

// Doublings of the fine cell that make up one spread tile edge; zero keeps the tile at
// one cell.
// One core's L2 sets the size: the tile's strengths want a quarter of it, measured over
// 2D and 3D, and its padded subgrid wants no more than all of it. The subgrid a tile
// writes is the tile grown by the kernel width, so the ceiling is on the padded edge,
// not the bare one.
// TODO: the tile pays for its halo whatever the density, and a sparse grid writes the
// halo of every tile it populates. Empty tiles cost nothing, so what is left to win is a
// tile edge that grows with the halo it has to pay for.
inline int spread_tile_doublings(int cell, int ndims, int nspread,
                                 double density) noexcept {
  const auto padded_fits_l2 = [=](double edge) {
    return spread_pow_ndims(edge + nspread, ndims) <=
           double(finufft::utils::getL2CacheSize()) / 16; // all of L2, same fixed 16
  };
  // Grow while the next doubling keeps the padded subgrid in L2 and the tile's strengths
  // in a quarter of it. The point budget only caps growth: a tile over it is split into
  // subproblems of the cap instead.
  int doublings = 0;
  for (double edge = 2.0 * cell;
       padded_fits_l2(edge) &&
       density * spread_pow_ndims(edge, ndims) <= double(spread_point_budget());
       edge *= 2)
    ++doublings;
  return doublings;
}

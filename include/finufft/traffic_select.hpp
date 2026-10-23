#ifndef FINUFFT_TRAFFIC_SELECT_HPP
#define FINUFFT_TRAFFIC_SELECT_HPP

/* Automatic choice of the sort bin dimensions and the maximum subproblem size
   for type-1 spreading, from an estimate of the DRAM lines (64 B) the bin sort
   and the spreading stage move per nonuniform point. Port of
   perf/binsweep/traffic_model.py (thesis Chapter 6, "Selection algorithm").

   Candidates: power-of-two bins bx in [16, 2048], by, bz in [4, 64] with
   bx >= by >= bz whose kernel-padded region fits 0.9 of L1d (the default
   16x4x4 if none does), and 40 log-spaced max_sp values up to M/nthr, kept
   if >= 2500. The pair with the fewest estimated lines is returned.

   Cache sizes are read from sysfs (Linux only). Where they cannot be read,
   cache_info() reports failure and the caller keeps the library defaults.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#ifdef __linux__
#include <fstream>
#include <string>
#endif

namespace finufft {
namespace traffic_select {

inline constexpr double LINE = 64;

// fitted ramp ends, in units of the per-thread share of the last-level cache
inline constexpr double SUB_LO = 0.5, SUB_HI = 2.0, BUF_LO = 0.1, BUF_HI = 1.5;
inline constexpr double G      = 0.94; // fraction of fine-grid lines the add step fetches

// cache sizes in bytes; ok is false if they could not be determined
struct Caches {
  double l1d = 0; // L1 data, per core
  double llc = 0; // last-level cache, shared by the spreading threads
  bool ok    = false;
};

#ifdef __linux__
// parse a sysfs cache size such as "32K", "1024K" or "50M"; 0 on failure
inline double parse_cache_size(const std::string &s) {
  std::size_t pos = 0;
  double v        = 0;
  try {
    v = std::stod(s, &pos);
  } catch (...) {
    return 0;
  }
  const char unit = pos < s.size() ? s[pos] : '\0';
  if (unit == 'K' || unit == 'k') return v * 1024;
  if (unit == 'M' || unit == 'm') return v * 1024 * 1024;
  if (unit == 'G' || unit == 'g') return v * 1024 * 1024 * 1024;
  return v;
}

// read the L1 data and last-level caches of cpu0 from
// /sys/devices/system/cpu/cpu0/cache/index*/
inline Caches read_sysfs_caches() {
  Caches c;
  int llc_level = 0;
  const std::string base = "/sys/devices/system/cpu/cpu0/cache/index";
  for (int i = 0; i < 16; ++i) {
    std::ifstream flevel(base + std::to_string(i) + "/level"),
        ftype(base + std::to_string(i) + "/type"), fsize(base + std::to_string(i) + "/size");
    if (!flevel || !ftype || !fsize) continue;
    int level = 0;
    std::string type, size;
    if (!(flevel >> level) || !(ftype >> type) || !(fsize >> size)) continue;
    const double bytes = parse_cache_size(size);
    if (type == "Instruction") continue;
    if (level == 1 && type == "Data") c.l1d = bytes;
    if (level > llc_level) {
      llc_level = level;
      c.llc     = bytes;
    }
  }
  c.ok = c.l1d > 0 && c.llc > 0 && llc_level > 1;
  return c;
}
#endif

// cache sizes of this machine, determined once; ok == false off Linux or on failure
inline const Caches &cache_info() {
#ifdef __linux__
  static const Caches c = read_sysfs_caches();
#else
  static const Caches c{};
#endif
  return c;
}

struct Choice {
  std::array<int, 3> bins{16, 4, 4};
  int max_sp   = 0;
  double lines = 0; // estimated DRAM lines per point, sort + spread
};

// kernel padding of a subgrid, in cells per axis
inline int halo(int ns) { return (ns % 2 == 0) ? ns - 1 : ns; }

inline double ramp(double x, double lo, double hi) {
  if (hi <= lo) return x > lo ? 1.0 : 0.0;
  return std::min(1.0, std::max(0.0, std::log(std::max(x, 1.0) / lo) / std::log(hi / lo)));
}

// average subgrid cells of a subproblem of s points (thesis Eqs. subgrid_1d/2d/3d)
inline double subgrid_cells(int d, const std::array<double, 3> &N, double rho, double w,
                            double s, const std::array<int, 3> &b) {
  if (d == 1) return s / rho + b[0] + w;
  if (d == 2) {
    const double x = s / (rho * b[1]), r = x / N[0], f = std::min(1.0, r);
    return (1 - f) * (x + b[0] + w) * (b[1] + w) +
           f * (N[0] + w) * ((std::max(1.0, r) + 1) * b[1] + w);
  }
  const double x = s / (rho * b[1] * b[2]), r = x / N[0], f = std::min(1.0, r);
  const double q = r * b[1] / N[1], g = std::min(1.0, q);
  return (1 - f) * (x + b[0] + w) * (b[1] + w) * (b[2] + w) +
         (f - g) * (N[0] + w) * ((std::max(1.0, r) + 1) * b[1] + w) * (b[2] + w) +
         g * (N[0] + w) * (N[1] + w) * ((std::max(1.0, q) + 1) * b[2] + w);
}

// lines per point of the spreading stage, reads + writes (thesis Eqs. spread_reads/writes)
inline double spread_lines(int d, int sz, const std::array<double, 3> &N, double M,
                           double max_sp, const std::array<int, 3> &b, int ns, int nthr,
                           const Caches &cache) {
  const double L3T = cache.llc / nthr;
  const double w = halo(ns), L = LINE / (2.0 * sz), rho = M / (N[0] * N[1] * N[2]);
  const double s   = M / std::max<double>(nthr, std::ceil(M / max_sp));
  const double cpp = subgrid_cells(d, N, rho, w, s, b) / s;
  const double U   = (1 - std::exp(-rho * std::pow(ns, d - 1) * (ns + L - 1))) / (rho * L);
  const double S = cpp / L, Bf = (d + 2) * sz / LINE;
  const double hs = ramp(cpp * s * 2 * sz, SUB_LO * L3T, SUB_HI * L3T);
  const double hb = ramp(s * (d + 2) * sz, BUF_LO * L3T, BUF_HI * L3T);
  const double reads  = (d + 1) + 2 * Bf * hb + 2 * S * hs + U * hs + G * S;
  const double writes = Bf * hb + 2 * S * hs + U * hs + G * S;
  return reads + writes;
}

// solve D(n*) = K for increasing D, return P(gap > n*) = exp(-n*/m)
template<typename F> inline double p_miss(F D, double m, double K) {
  double lo = 0, hi = 1;
  while (D(hi) < K) {
    hi *= 2;
    if (hi > 1e15) return 0;
  }
  for (int i = 0; i < 100; ++i) {
    const double mid = (lo + hi) / 2;
    (D(mid) < K ? lo : hi) = mid;
  }
  return std::exp(-lo / m);
}

// lines per point of the bin sort, reads + writes (thesis Eqs. sort_reads/writes plus
// the constant 2c + 1/8 + 1/8, kept so the totals match the Python model)
inline double sort_lines(int d, int sz, const std::array<double, 3> &N, double M,
                         const std::array<int, 3> &b, int sort_nthr, int nthr,
                         const Caches &cache) {
  double nbins = 1;
  for (int i = 0; i < d; ++i) nbins *= std::floor(N[i] / b[i]) + 1;
  const double Lc = std::ceil(4 * nbins / LINE);
  const int nt    = sort_nthr;
  const double K = (nt == 1 ? cache.llc : cache.llc / nthr) / LINE, c = d * sz / LINE;
  auto cl  = [&](double n) { return Lc * (1 - std::exp(-n / Lc)); };
  auto bl  = [&](double n) { return nbins * (1 - std::exp(-n / nbins)); };
  const double pc1 = p_miss([&](double n) { return cl(n) + c * n; }, Lc, K);
  const double pc2 = p_miss([&](double n) { return cl(n) + c * n + bl(n); }, Lc, K);
  const double open = std::min(nt * nbins, M * 8 / LINE) + nt * Lc;
  const double pret = open <= cache.llc / LINE ? 0 : 1 - (cache.llc / LINE) / open;
  const double copies = nt > 1 ? (nt * Lc + (nt + 1) * Lc) / M : Lc / M;
  const double reads  = 2 * c + 1.0 / 8 + pc1 + pc2 + 7.0 / 8 * pret + copies;
  const double writes = 1.0 / 8 + pc1 + pc2 + 7.0 / 8 * pret + copies;
  return reads + writes;
}

// candidate bins: power-of-two, bx >= by >= bz, kernel-padded bin within 0.9 L1d
inline std::vector<std::array<int, 3>> candidate_bins(int d, int sz, int ns,
                                                      const Caches &cache) {
  const int W = halo(ns);
  std::vector<std::array<int, 3>> out;
  const int nx[] = {16, 32, 64, 128, 256, 512, 1024, 2048}, nyz[] = {4, 8, 16, 32, 64};
  for (int x : nx)
    for (int y : (d > 1 ? std::vector<int>(nyz, nyz + 5) : std::vector<int>{1}))
      for (int z : (d > 2 ? std::vector<int>(nyz, nyz + 5) : std::vector<int>{1})) {
        if (!(x >= y && y >= z)) continue;
        const double bytes = double(x + W) * (d > 1 ? y + W : 1) * (d > 2 ? z + W : 1) * 2 * sz;
        if (bytes <= 0.9 * cache.l1d) out.push_back({x, y, z});
      }
  if (out.empty()) out.push_back({16, d > 1 ? 4 : 1, d > 2 ? 4 : 1});
  return out;
}

// candidate max_sp: 40 log-spaced values from 200 to ceil(M/nthr), kept if >= 2500
inline std::vector<int> candidate_max_sp(double M, int nthr) {
  const double one = std::ceil(M / nthr), a = std::log10(200.0), z = std::log10(one);
  std::vector<int> out;
  for (int i = 0; i < 40; ++i) {
    const double v = std::nearbyint(std::pow(10.0, a + (z - a) * i / 39));
    if (v >= 2500) out.push_back(int(v));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  if (out.empty()) out.push_back(int(one));
  return out;
}

/* Choose bins and max_sp for spreading M points of size sz (bytes per real) onto
   the fine grid N (unused dims 1) with kernel width ns, nthr spreading threads and
   sort_nthr sorting threads, given the cache sizes. A nonzero fixed_bins[0] or
   fixed_max_sp pins that parameter to the user's value and only the other is chosen. */
inline Choice select(int d, int sz, const std::array<double, 3> &N, double M, int ns,
                     int nthr, int sort_nthr, const std::array<int, 3> &fixed_bins,
                     int fixed_max_sp, const Caches &cache) {
  nthr      = std::max(nthr, 1);
  sort_nthr = std::max(sort_nthr, 1);
  const auto bins = fixed_bins[0] > 0 ? std::vector<std::array<int, 3>>{fixed_bins}
                                      : candidate_bins(d, sz, ns, cache);
  const auto sps  = fixed_max_sp > 0 ? std::vector<int>{fixed_max_sp}
                                     : candidate_max_sp(M, nthr);
  Choice best;
  best.lines = -1;
  for (const auto &b : bins) {
    const double so = sort_lines(d, sz, N, M, b, sort_nthr, nthr, cache);
    for (int s : sps) {
      const double t = so + spread_lines(d, sz, N, M, s, b, ns, nthr, cache);
      if (best.lines < 0 || t < best.lines) best = {b, s, t};
    }
  }
  return best;
}

} // namespace traffic_select
} // namespace finufft

#endif // FINUFFT_TRAFFIC_SELECT_HPP

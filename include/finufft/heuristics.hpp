#pragma once

// Complexity-based upsampfac (sigma) selection for the type-1/2 and type-3 setpts
// paths: cost-model primitives + a generic minimizer. Home for the tuning constants.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <finufft/spreadinterp.hpp>     // finufft::spreadinterp::get_padding<TF>
#include <finufft_common/constants.h>   // PI, MAX_CHECK_SIGMA, MIN/MAX_AUTO_UPSAMPFAC
#include <finufft_common/kernel.h>      // ns formulas, feasibility, fine_grid_len
#include <finufft_common/spread_opts.h> // finufft_spread_opts

namespace finufft::heuristics {

// The cost-minimizing feasible upsampfac and its predicted cost.
struct sigma_info {
  double sigma; // cost-minimizing feasible upsampfac
  double cost;  // its predicted cost (same flop units across all transform types)
};

// --- cost primitives (all in the same arbitrary-but-consistent flop units) ---

// Inner spread work per NU point: the 2*ns-wide complex row is vectorized, so it
// runs as ceil(2*ns / simd_width) SIMD ops, not 2*ns scalars. Precision/ISA-adaptive
// (wider lanes in f32, fewer ops); measured ns-per-point scales as this x ns^(dim-1),
// so the inner row must be counted in vectors — counting padded scalars (~2*ns)
// over-weights wide kernels and biases the pick toward too-large sigma.
template<typename TF> inline double spread_row(int ns) {
  const double padded_2ns = 2.0 * ns + finufft::spreadinterp::get_padding<TF>(2 * ns);
  const double simd = (double)finufft::spreadinterp::get_padded_simd_width<TF>(2 * ns);
  return std::max(padded_2ns / simd, 1.0);
}

// spread_row x ns^(dim-1) per-point transverse factor.
template<typename TF> inline double spread_cost(double npts, int ns, int dim) {
  double outer = 1.0; // ns^(dim-1)
  for (int idim = 1; idim < dim; ++idim) outer *= (double)ns;
  return npts * spread_row<TF>(ns) * outer;
}

// FFT cost weight per fine-grid point per log2 unit, relative to one spread flop.
// Rises with threads: spreading parallelizes ~linearly, the FFT sublinearly.
inline double c_fft(int nthreads) {
  // calibrated on ccmlin075 (AVX-512, FFTW); see devel/calibrate_upsampfac.cpp.
  // Centred in a wide flat plateau (C in ~[1.2,3] all pick within 3% of optimum on
  // 1D/2D/3D, f32/f64, single/multi-thread); one shared set covers FFTW and DUCC0.
  constexpr double C_FFT_BASE = 2.0;
  constexpr double K_FFT_THREAD = 0.50;
  return C_FFT_BASE * std::pow((double)std::max(1, nthreads), K_FFT_THREAD);
}

// FFT cost c*G*log2(G), G = fine-grid volume.
inline double fft_cost(double c, const double *nmodes, double sigma, int ns, int dim) {
  double G = 1.0;
  for (int idim = 0; idim < dim; ++idim)
    G *= (double)finufft::common::fine_grid_len(sigma, nmodes[idim], ns);
  return c * G * std::log2(G);
}

// --- candidate enumeration ---

// Kernel width ns the plan would actually use at this (tol, sigma): the theoretical
// width clamped to the compiled/feasible range, exactly as setup_spreadinterp picks it.
// Lets the cost model score each candidate sigma at its real ns.
template<typename TF>
inline int kernel_width_at(double tol, int dim, int type, double sigma) {
  finufft_spread_opts so{};
  so.kerformula = 0;
  so.upsampfac = sigma;
  return finufft::kernel::clamp_kernel_ns<TF>(
      finufft::kernel::theoretical_kernel_ns(tol, dim, type, 0, so), sigma);
}

// Returns the feasible upsampfac (sigma) that minimizes cost, and its predicted cost.
//   cost(sigma, ns): caller-supplied score of a candidate in consistent flop units.
//   maxN:            largest mode count over dims (1 for type 3).
// If tol is unachievable, returns the largest sigma and the plan pipeline reports it.
template<typename TF, class Cost>
sigma_info minimize(double tol, int dim, int type, double maxN, Cost &&cost) {
  using namespace finufft::common;
  constexpr double eps_mach = std::numeric_limits<TF>::epsilon();
  constexpr bool is_float = std::is_same_v<TF, float>;
  const auto feasible = [&](double sigma) {
    return upsampfac_feasible(sigma, tol, dim, type, eps_mach, MAX_NSPREAD<TF>, is_float,
                              maxN);
  };
  const double sigma_min =
      analytic_upsampfac(tol, dim, type, eps_mach, MAX_NSPREAD<TF>, is_float, maxN);
  const int ns_min = kernel_width_at<TF>(tol, dim, type, sigma_min);
  sigma_info best{sigma_min, cost(sigma_min, ns_min)};
  if (!feasible(sigma_min)) return best; // tol unachievable; pipeline reports
  for (int ns_t = ns_min - 1;
       ns_t >= kernel_width_at<TF>(tol, dim, type, MAX_AUTO_UPSAMPFAC); --ns_t) {
    const double s = std::clamp(smallest_sigma_for_ns(tol, dim, type, ns_t), sigma_min,
                                MAX_AUTO_UPSAMPFAC);
    if (!feasible(s)) continue;
    const double c = cost(s, kernel_width_at<TF>(tol, dim, type, s));
    if (c < best.cost) best = {s, c};
  }
  return best;
}

// --- transform-specific selectors ---

// Type 1/2: spread of npts points + FFT. Also the inner type-2 cost for best_type3.
template<typename TF>
sigma_info best_type12(double tol, int dim, int type, int nthreads, const double *nmodes,
                       double npts) {
  const double c = c_fft(nthreads);
  const double maxN = *std::max_element(nmodes, nmodes + dim);
  const auto cost = [&](double sigma, int ns) {
    return spread_cost<TF>(npts, ns, dim) + fft_cost(c, nmodes, sigma, ns, dim);
  };
  return minimize<TF>(tol, dim, type, maxN, cost);
}

// Fine-grid length set_nhg_type3 builds for one dim at this sigma3, from the
// source/target interval half-widths X,S. Thin wrapper over the shared finufft::common
// helper; the cost model only needs the length (no plan to mutate) and ignores the
// MAX_NF allocation guard, so pass max_nf=BIGINT max to always next235-round.
inline double type3_fine_grid_len(double sigma3, double X, double S, int ns3) {
  return (double)std::get<0>(
      finufft::common::nhg_type3(sigma3, X, S, ns3, std::numeric_limits<BIGINT>::max()));
}

// Returns the cost-minimizing upsampfac (sigma3) for a type-3 transform.
// Type 3: outer spread of nj sources (width ns3) onto a fine grid of size
// nfdim(sigma3) ∝ sigma3, then a full inner type-2 NUFFT evaluating nk targets on
// that grid (its own sigma2 re-optimized via best_type12). Minimizing over feasible
// sigma3 trades the outer spread (cheaper at large sigma3, narrow kernel) against the
// inner t2 (cheaper at small sigma3, smaller grid). X,S are the per-dim source/target
// interval half-widths (from arraywidcen).
template<typename TF>
double best_type3(double tol, int dim, int nthreads, double nj, const double *X,
                  const double *S, double nk) {
  const auto cost = [&](double sigma3, int ns3) {
    std::array<double, 3> nmodes{1.0, 1.0, 1.0};
    for (int idim = 0; idim < dim; ++idim)
      nmodes[idim] = type3_fine_grid_len(sigma3, X[idim], S[idim], ns3);
    const double inner = best_type12<TF>(tol, dim, 2, nthreads, nmodes.data(), nk).cost;
    return spread_cost<TF>(nj, ns3, dim) + inner;
  };
  return minimize<TF>(tol, dim, /*type=*/3, /*maxN=*/1.0, cost).sigma;
}

} // namespace finufft::heuristics

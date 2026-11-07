// Upsampfac (sigma) selection for plans made with upsampfac=0: a flop model and the
// search over kernel widths that minimizes it.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <finufft/plan.hpp>
#include <finufft/simd.hpp> // get_padding, get_padded_simd_width
#include <finufft_common/constants.h>
#include <finufft_common/kernel.h>

using namespace finufft::common;
using namespace finufft::kernel;

// Spread work per NU point: the 2*ns-wide complex row runs as ceil(2*ns/simd_width)
// SIMD ops, times ns^(dim-1) rows.
static double spread_flops(bool is_float, double npts, int ns, int dim) {
  using namespace finufft::spreadinterp;
  const double padded_2ns =
      2.0 * ns + (is_float ? get_padding<float>(2 * ns) : get_padding<double>(2 * ns));
  const double simd = (double)(is_float ? get_padded_simd_width<float>(2 * ns)
                                        : get_padded_simd_width<double>(2 * ns));
  double rows       = 1.0;
  for (int idim = 1; idim < dim; ++idim) rows *= (double)ns;
  return npts * std::max(padded_2ns / simd, 1.0) * rows;
}

// Type 1/2: spread the NU points, then the FFT c*G*log2(G) on the fine grid of volume
// G. In a type-3 plan this is the inner type 2, with nk points. c weighs one FFT point
// against one spread op; it rises with threads, as spreading scales ~linearly and the
// FFT sublinearly. Calibrated on ccmlin075 (AVX-512, FFTW).
template<typename TF>
double FINUFFT_PLAN_T<TF>::flops_type12(int ns, double sigma,
                                        const std::array<double, 3> &nmodes) const {
  const double npts = type == 3 ? m.nk : m.nj;
  const double c    = 2.0 * std::sqrt((double)std::max(1, opts.nthreads));
  double G          = 1.0;
  for (int idim = 0; idim < dim; ++idim)
    G *= (double)fine_grid_len(sigma, nmodes[idim], ns);
  return spread_flops(std::is_same_v<TF, float>, npts, ns, dim) + c * G * std::log2(G);
}

// Type 3: spread nj sources onto the nhg_type3 grid, then the inner type 2 to nk
// targets at its own best sigma. S holds the target half-widths.
template<typename TF>
double FINUFFT_PLAN_T<TF>::flops_type3(int ns, double sigma,
                                       const std::array<double, 3> &S) const {
  const bool is_float = std::is_same_v<TF, float>;
  std::array<double, 3> nf{1.0, 1.0, 1.0};
  for (int idim = 0; idim < dim; ++idim)
    nf[idim] = (double)std::get<0>(nhg_type3(sigma, m.t3P.X[idim], S[idim], ns, MAX_NF));
  const double sigma2 = best_sigma_for_tol(2, nf);
  if (!std::isfinite(sigma2)) return std::numeric_limits<double>::infinity();
  const int ns2 = clamp_kernel_ns(theoretical_kernel_ns(m.tol, dim, 2, sigma2), sigma2,
                                  dim, is_float);
  return spread_flops(is_float, (double)m.nj, ns, dim) + flops_type12(ns2, sigma2, nf);
}

// Fewest-flops sigma in [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] that meets tol, or
// infinity if none does. nmodes holds the mode counts for type 1/2 and the target
// half-widths S for type 3.
template<typename TF>
double FINUFFT_PLAN_T<TF>::best_sigma_for_tol(int type,
                                              const std::array<double, 3> &nmodes) const {
  constexpr auto is_float = std::is_same_v<TF, float>;
  const auto eps_mach     = is_float ? std::numeric_limits<float>::epsilon()
                                     : std::numeric_limits<double>::epsilon();
  const auto tol          = m.tol;
  auto sigma              = std::numeric_limits<double>::infinity();
  auto flops              = std::numeric_limits<double>::infinity();
  const int max_ns        = is_float ? MAX_NSPREAD<float> : MAX_NSPREAD<double>;
  auto compute_flops      = [&](double sigma_ns) {
    // price the ns the plan will use at sigma_ns, as setup_spreadinterp sets it
    const int ns_plan = clamp_kernel_ns(theoretical_kernel_ns(tol, dim, type, sigma_ns),
                                        sigma_ns, dim, is_float);
    return type == 3 ? flops_type3(ns_plan, sigma_ns, nmodes)
                     : flops_type12(ns_plan, sigma_ns, nmodes);
  };
  for (int ns = 2; ns <= max_ns; ++ns) {
    const auto sigma_alias = min_sigma_for_aliasing_error(tol, dim, type, ns);
    const auto sigma_deconv = min_sigma_for_deconv_error(tol, dim, ns, eps_mach);
    const auto sigma_ns     = std::max({sigma_alias, sigma_deconv, MIN_AUTO_UPSAMPFAC});
    if (opts.debug > 1)
      printf("[%s] ns=%d: sigma_alias=%.3g sigma_deconv=%.3g sigma_ns=%.3g\n", __func__,
             ns, sigma_alias, sigma_deconv, sigma_ns);
    // also skips infinity; continue because sigma is not really monotone in ns.
    if (sigma_ns > MAX_AUTO_UPSAMPFAC) continue;
    constexpr double edge_pad = 1e-3;
    const auto sigma_pad      = sigma_ns + edge_pad;
    const auto flops_ns       = compute_flops(sigma_pad);
    if (opts.debug > 1) printf("[%s] ns=%d: flops_ns=%.3g\n", __func__, ns, flops_ns);
    if (flops_ns < flops) {
      flops = flops_ns;
      sigma = sigma_pad;
    }
  }
  return sigma;
}

template double FINUFFT_PLAN_T<float>::best_sigma_for_tol(
    int type, const std::array<double, 3> &nmodes) const;
template double FINUFFT_PLAN_T<double>::best_sigma_for_tol(
    int type, const std::array<double, 3> &nmodes) const;

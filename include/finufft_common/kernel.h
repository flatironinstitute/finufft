#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <type_traits>
#include <vector>

#include <finufft_common/constants.h>
#include <finufft_common/pswf.h>
#include <finufft_common/spread_opts.h>
#include <finufft_common/utils.h>
#include <finufft_errors.h>

namespace finufft::kernel {

template<class T, class F> std::vector<T> poly_fit(F &&f, int n) {
  static_assert(std::is_floating_point_v<T>, "T must be floating-point");
  /* Expects f, a function handle for arguments on [-1,1], both I/O type T.
     Returns vector of n coefficients a_{n-1}, ... a_1, a_0 of degree-(n-1)
     polynomial that interpolates f at a set of hard-wired Chebyshev nodes.

     Barbone, Fall 2025.
     Barnett 12/29/25-1/13/26 removed a,b to simplify; no poly defn confusion.
  */

  // 1) Type-1 Chebyshev nodes t_k, data samples y_k = f(t_k)
  std::vector<T> t(n), y(n);
  for (int k = 0; k < n; ++k) {
    t[k] = std::cos((T(2 * k + 1) * common::PI) / (T(2) * T(n))); // in (-1,1)
    // t[k]       = std::cos((T(k) * common::PI) / T(n-1)); // type-2 in [-1,1] also ok
    y[k] = static_cast<T>(f(t[k])); // evaluate this sample
  }

  // 2) Newton divided differences on t: coef[j] = f[t0,..,tj]
  std::vector<T> coef = y;
  for (int j = 1; j < n; ++j)
    for (int i = n - 1; i >= j; --i)
      coef[i] = (coef[i] - coef[i - 1]) / (t[i] - t[i - j]);

  // 3) Convert Newton form to monomial coeffs in t (low→high)
  // Multiply a polynomial p(t) by (t - c).
  // Input p holds coefficients low->high (p[0] + p[1] t + p[2] t^2 + ...).
  // Returns r of length p.size()+1 so that r represents (t - c)*p(t).
  // Concretely: r[i] = -c*p[i] + (i>0 ? p[i-1] : 0) for i=0..p.size()-1,
  // and r[p.size()] = p[p.size()-1].
  auto mul_by_linear = [](const std::vector<T> &p, T c) {
    std::vector<T> r(p.size() + 1, T(0));
    for (std::size_t i = 0; i < p.size(); ++i) {
      r[i] += -c * p[i]; // (t - c)*p
      r[i + 1] += p[i];
    }
    return r;
  };

  std::vector<T> c(n, T(0));  // output coefficients in t
  std::vector<T> basis{T(1)}; // Π_{m<j} (t - t_m), low→high
  c[0] += coef[0];
  for (int j = 1; j < n; ++j) {
    basis = mul_by_linear(basis, t[j - 1]);
    for (std::size_t m = 0; m < basis.size(); ++m) c[m] += coef[j] * basis[m];
  }
  std::reverse(c.begin(), c.end()); // convert to Horner ordering (high -> low)
  return c;
}

// The spread/interp kernel phi_beta(z) on z in [-1,1]. Not performance-critical;
// used only for polynomial interpolation (precompute_horner_coeffs). Always double.
// Defined in src/common/kernel.cpp.
std::function<double(double)> kernel_definition_lambda(const finufft_spread_opts &spopts);

// Tolerance prefactor in the kernel aliasing law tol = tolfac*exp(-(ns-1)*pi*u).
double kernel_tolfac(int dim, int type);

int theoretical_kernel_ns(double tol, int dim, int type,
                          const finufft_spread_opts &spopts);

// Clamp a theoretical kernel width to the width the plan will actually use: clip to
// [MIN_NSPREAD, max_nspread] (widest spreadinterp template compiled), then the float
// catastrophic-cancellation guard (constants.h: ns<=FLOAT_MAX_NS_CC for
// upsampfac<FLOAT_CC_UPSAMPFAC_LIMIT). Shared by setup_spreadinterp() and the upsampfac
// heuristic so they agree on the width.
inline int clamp_kernel_ns(int ns, double upsampfac, int max_nspread, bool is_float) {
  ns = std::max(common::MIN_NSPREAD, ns);
  ns = std::min(ns, max_nspread);
  if (is_float && upsampfac < common::FLOAT_CC_UPSAMPFAC_LIMIT)
    ns = std::min(ns, common::FLOAT_MAX_NS_CC);
  return ns;
}
template<class TF> inline int clamp_kernel_ns(int ns, double upsampfac) {
  return clamp_kernel_ns(ns, upsampfac, common::MAX_NSPREAD<TF>,
                         std::is_same_v<TF, float>);
}

void set_kernel_shape_given_ns(finufft_spread_opts &opts, int debug);

// Analytic PSWF self-FT parameters. The order-0 PSWF is an eigenfunction of the
// finite Fourier transform, so the kernel's continuous FT is a rescaled copy of the
// kernel itself: phihat(xi) = prefac * phi(grid_scale*xi), with phi the Horner
// approximant (evaluate_kernel_runtime, grid-unit arg), grid_scale = J2^2/beta,
// J2 = nspread/2, and prefac = phihat(0) = int phi dx (as phi(0)=1). That integral
// is closed-form from the panel coeffs: on panel i, phi = sum_j coeffs[j*stride+i]
// z^(nc-1-j), z in [-1,1], dz = 2dx, so odd powers cancel and the whole setup is
// ~ns.nc/2 flops, no quadrature. Integrating the approximant rather than the exact
// PSWF makes phihat the FT of the function actually spread. Valid only inside the
// kernel support, |xi| <= 1/grid_scale = 2.beta/nspread; type-3 deconv evaluates at
// |xi| <= pi/upsampfac (= h.gam.S, setpts), and over every reachable (tol, dim,
// precision, upsampfac >= 1.1) the margin 2.beta/nspread - pi/upsampfac is >= 0.47
// for kerformula 7-9. Returns (grid_scale, prefac). Used by the type-3 per-target
// evaluator (Kernel_onedim_FT); type-1/2 fseries keeps quadrature. GPU-reusable.
// Barbone 7/23/26.
template<class TF>
inline std::tuple<double, double> pswf_selfft_params(
    int nspread, double beta, const TF *coeffs, int nc, int stride) {
  double prefac = 0;
  for (int i = 0; i < nspread; ++i)
    for (int j = nc - 1; j >= 0; j -= 2) // even powers of z only
      prefac += double(coeffs[j * stride + i]) / (nc - j);
  const double J2 = nspread / 2.0; // half-width of kernel z-support
  return {J2 * J2 / beta, prefac};
}

// min and max number of poly coeffs allowed (compiled) for a given spread width ns.
// Since for low upsampfacs, ns=16 can need only nc~12, allow such low nc here.
// Note: spreadinterp.cpp compilation time grows with the gap between these bounds...
inline constexpr int min_nc_given_ns(int ns) {
  return std::max(common::MIN_NC, ns - 4); // note must stay in bounds from constants.h
}
inline constexpr int max_nc_given_ns(int ns) {
  return std::min(common::MAX_NC, ns + 3); // "
}

template<int NS, int NC> inline constexpr bool ValidKernelParams() noexcept {
  // NS = nspread (kernel width), NC = # poly coeffs in Horner evaluator.
  // Defines the compiled range of NC for each NS, in spreadinterp.
  // Other instantiations can be
  // compiled away at call sites using if constexpr to reduce binary size.
  // Barbone Dec 2025.
  // AHB changed to use the above two expressions, but needs checking if compile-time ok
  return (NC >= min_nc_given_ns(NS)) && (NC <= max_nc_given_ns(NS));
}

} // namespace finufft::kernel

namespace finufft::common {

// Fine-grid length one dimension gets at this sigma, mirroring set_nf_type12
// (makeplan.hpp): ceil(sigma*n_modes), floored at 2*ns, rounded up to the next even
// 2,3,5-smooth number. set_nf_type12 owns the BIGINT/MAX_NF version; this is the
// read-only mirror for the upsampfac selector.
inline long fine_grid_len(double sigma, double n_modes, int ns) {
  return next235(std::max((long)std::ceil(sigma * n_modes), (long)(2 * ns)), 2);
}

// Type-3 fine grid for one dimension, given the source/freq interval half-widths X,S.
// Clamps so X*S>=1 (handling X and/or S == 0), sizes the upsampled grid nf (floored at
// 2*ns, then next235-rounded unless it already exceeds max_nf, the allocation guard),
// and returns spacing h=2pi/nf and x-rescale gam=nf/(2*sigma*S). All math in double;
// callers narrow h/gam to their precision. Single source of truth for set_nhg_type3 on
// CPU (setpts.hpp) and GPU (cuda/makeplan.cu) and the type-3 upsampfac cost model
// (heuristics.hpp). Barnett 6/12/17 logic; extracted Barbone 6/26.
// Returns the tuple (nf, h, gam); destructure with structured bindings.
inline std::tuple<BIGINT, double, double> nhg_type3(double sigma, double X, double S,
                                                    int ns, BIGINT max_nf) {
  const int nss = ns + 1; // since ns may be odd
  double Xs = X, Ss = S; // tweak so X*S>=1, handling X=0 and/or S=0
  if (Xs == 0.0) {
    if (Ss == 0.0) {
      Xs = Ss = 1.0;
    } else
      Xs = 1.0 / Ss;
  } else
    Ss = std::max(Ss, 1.0 / Xs);
  double nfd = 2.0 * sigma * Ss * Xs / PI + nss;
  if (!std::isfinite(nfd)) nfd = 0.0;
  BIGINT nf = std::max((BIGINT)nfd, (BIGINT)(2 * ns)); // catch too-small / nan / +-inf
  if (nf < max_nf) nf = next235(nf, 2); // else too big; spread will error
  return {nf, 2.0 * PI / (double)nf, (double)nf / (2.0 * sigma * Ss)};
}

// Smallest sigma whose theoretical_kernel_ns(tol,dim,type) is <= ns_target.
double smallest_sigma_for_ns(double tol, int dim, int type, int ns_target);

// Minimum sigma achieving requested tol, as used by check_sigma.
double lowest_sigma(double tol, int dim, int ns, double eps_mach, double gridlen);

// Whether the plan pipeline would accept this sigma at this tol.
bool upsampfac_feasible(double sigma, double tol, int dim, int type, double eps_mach,
                        int max_nspread, bool is_float, double maxN);

// Smallest feasible sigma in [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] by bisection.
double analytic_upsampfac(double tol, int dim, int type, double eps_mach, int max_nspread,
                          bool is_float, double maxN);

} // namespace finufft::common

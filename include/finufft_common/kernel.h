#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
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

// Plan error model: one function per error source, each compared with tol, and next to
// each its inverse in sigma. An inverse returns the smallest 3-digit sigma above the
// edge of its target, or infinity above MAX_AUTO_UPSAMPFAC. Both must be changed at the
// same time.
// [FIN] is the 2019 SISC paper; docs/devnotes.rst lists the known gaps.
// No CPU types, so cufinufft can use it once it uses the same kernel and width law.

// Rounding of the inputs, amplified by the mode index, [FIN] Remark 9. mode_diagonal
// is the length of the diagonal of the box of modes.
inline double rounding_error(double eps_mach, double mode_diagonal) {
  return common::ROUND_FAC * eps_mach * mode_diagonal;
}

// Aliasing prefactor c_d of the width law.
double kernel_tolfac(int dim);

// Decay rate of the kernel tail past the top mode a, in cycles per fine-grid point.
inline double alias_decay(double sigma, double a) {
  const double cutoff = 1.0 - 0.5 / sigma; // [FIN] (4.5)
  return std::sqrt(std::fma(cutoff, cutoff, -a * a));
}

// Aliasing, as the smallest width that meets tol, [FIN] (4.7). It is a width and not
// an error, so the feasibility check and the width law agree at the edge of a width.
// Zero or NaN decay gives the max int.
inline int aliasing_ns(double tol, double tolfac, double decay) {
  if (!(decay > 0.0)) return std::numeric_limits<int>::max();
  const double w = std::ceil(std::log(tolfac / tol) / (common::PI * decay) + 1.0);
  return (int)std::clamp(w, 1.0, (double)std::numeric_limits<int>::max());
}

// The width law: aliasing_ns on an infinite grid, before clamp_kernel_ns.
int ns_for_sigma(double tol, int dim, double sigma);

// Inverse of ns_for_sigma.
double sigma_for_ns(double tol, int dim, int ns);

// Rounding amplified by the deconvolution, [FIN] (3.21). The paper states it in 1D;
// the factor dim is fitted.
inline double deconv_error(double eps_mach, int dim, int ns, double sigma, double decay) {
  return dim * eps_mach * std::exp(ns * common::PI * (1.0 - 0.5 / sigma - decay));
}

// Inverse of deconv_error at the type-3 top mode. Types 1/2 have a lower top mode, so
// for them it is an upper bound.
double sigma_for_deconv(double eps_mach, int dim, int ns, double tol);

// Tolerance of one plan. Type 3 runs an outer and an inner plan whose independent
// relative errors add as a root sum of squares, so the two plans split tol equally.
inline double plan_tol(double tol, int type) {
  return type == 3 ? tol / std::sqrt(2.0) : tol;
}

// Whether sigma is feasible: at the width the plan uses at sigma, the aliasing and
// deconvolution terms meet tol. Rounding is checked by the caller, since sigma does not
// move it. maxN is the largest mode count; type 3 ignores it.
bool upsampfac_feasible(double sigma, double tol, int dim, int type, bool is_float,
                        double maxN);

// Widest float ns that does not lose accuracy to catastrophic cancellation at low
// sigma: 8 plus a sigma- and dim-dependent step of 1/0.075 per unit over the knee.
inline int float_max_ns_cc(double sigma, int dim) {
  constexpr double UPSAMPFAC_STEP = 1.0 / 0.075;
  return 8 + static_cast<int>(std::floor(
                 std::max(0.0, (sigma - 0.975 - 0.125 * dim) * UPSAMPFAC_STEP)));
}

// Clamp a theoretical kernel width to the width the plan will actually use: clip to
// [MIN_NSPREAD, MAX_NSPREAD] (widest spreadinterp template compiled),
// then the float catastrophic-cancellation guard. Shared by setup_spreadinterp() and
// the upsampfac heuristic so they agree on the width.
inline int clamp_kernel_ns(int ns, double upsampfac, int dim, bool is_float) {
  using common::MAX_NSPREAD;
  const int max_ns = is_float ? MAX_NSPREAD<float> : MAX_NSPREAD<double>;
  ns               = std::max(common::MIN_NSPREAD, ns);
  ns               = std::min(ns, max_ns);
  // Float catastrophic-cancellation guard. Both sides are int, so a huge upsampfac
  // cannot overflow.
  if (is_float) ns = std::min(ns, float_max_ns_cc(upsampfac, dim));
  return ns;
}
template<class TF> inline int clamp_kernel_ns(int ns, double upsampfac, int dim) {
  return clamp_kernel_ns(ns, upsampfac, dim, std::is_same_v<TF, float>);
}

// The width setup_spreadinterp uses at sigma.
inline int kernel_width_at(double tol, int dim, double sigma, bool is_float) {
  return clamp_kernel_ns(ns_for_sigma(tol, dim, sigma), sigma, dim, is_float);
}
template<class TF> inline int kernel_width_at(double tol, int dim, double sigma) {
  return kernel_width_at(tol, dim, sigma, std::is_same_v<TF, float>);
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
inline BIGINT fine_grid_len(double sigma, double n_modes, int ns) {
  return next235(std::max(BIGINT(std::ceil(sigma * n_modes)), BIGINT(2 * ns)), 2);
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

} // namespace finufft::common

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <finufft_common/common.h>
#include <finufft_common/constants.h>
#include <finufft_common/kernel.h>
#include <finufft_common/pswf.h>
#include <finufft_common/safe_call.h>
#include <finufft_common/spread_opts.h>
#include <finufft_errors.h>
#include <functional>
#include <limits>

// this module uses finufft_spread_opts but does not know about FINUFFT_PLAN class
// nor finufft_opts. This allows it to be used by CPU & GPU.

namespace finufft::kernel {

std::function<double(double)> kernel_definition_lambda(
    const finufft_spread_opts &spopts) {
  /* The spread/interp kernel phi_beta(z) function on standard interval z in [-1,1].
     This evaluation does not need to be fast; it is used *only* for polynomial
     interpolation via Horner coeffs (the interpolant is evaluated fast).
     It can thus always be double-precision. No analytic Fourier transform pair is
     needed, thanks to numerical quadrature in finufft_core:onedim*; playing with
     new kernels is thus very easy.
    Inputs:
    spopts - spread_opts struct containing fields:
      beta        - shape parameter for ES, KB, or other prolate kernels
                    (a.k.a. c parameter in PSWF).
      kerformula  - positive integer selecting among kernel function types; see
                    docs in the code below.
                    (More than one value may give the same type, to allow
                    kerformula also to select a parameter-choice method.)
                    Note: the default 0 (in opts.spread_kerformula) is invalid here;
                    selection of a >0 kernel type must already have happened.
    Output: lambda function that can be called with z to return phi(z),
                    as in the notation of original 2019 paper ([FIN] in the docs).

    Notes: 1) no normalization of max value or integral is needed, since any
            overall factor is cancelled out in the deconvolve step. However,
            values as large as exp(beta) have caused floating-pt overflow; don't
            use them.
    Barnett rewritten 1/13/26 for double on [-1,1]; based on Barbone Dec 2025.
  */
  double beta = spopts.beta; // get shape param
  int kf      = spopts.kerformula;

  if (kf >= 7 && kf <= 9) {
    finufft::common::PSWF0 pswf(beta);
    return [pswf](double z) {
      if (std::abs(z) > 1.0) return 0.0; // restrict support to [-1,1]
      return pswf(z);
    }; // prolate (PSWF) Psi_0, normalized to 1 at z=0
  } else {
    fprintf(stderr, "[%s] unknown spopts.kerformula=%d\n", __func__, spopts.kerformula);
    throw finufft::exception(FINUFFT_ERR_KERFORMULA_NOTVALID);
  }
}

double kernel_tolfac(int dim, int type) {
  // Tolerance prefactor in the kernel aliasing law tol = tolfac*exp(-(ns-1)*pi*u).
  // PER_DIM^(dim-1) covers the measured error growth per dimension; TYPE3 covers the
  // type-3 outer spread. The two are separate fits that share the value 1.4. The width
  // law and its inverse both read it, so they cannot disagree.
  constexpr double TOLFAC_1D = 0.18; // 1D type-1/2 base prefactor
  constexpr double TOLFAC_PER_DIM = 1.4; // per-extra-dim worsening, ^(dim-1)
  constexpr double TOLFAC_TYPE3 = 1.4; // type-3 outer-spread extra worsening
  constexpr auto ipow = [](double base, int n) {
    double r = 1.0;
    for (int i = 0; i < n; ++i) r *= base;
    return r;
  };
  return TOLFAC_1D * ipow(TOLFAC_PER_DIM, dim - 1) * (type == 3 ? TOLFAC_TYPE3 : 1.0);
}

int theoretical_kernel_ns(double tol, int dim, int type, double sigma) {
  // Spread width ns that meets tol at upsampfac sigma > 1 in exact arithmetic, by the
  // aliasing law. One law for all PSWF kerformulas; no clip to the compiled range.
  const double tolfac = kernel_tolfac(dim, type);
  const double nsoff = 1.0; // width offset (helps balance err over sigma range)
  int ns = (int)std::ceil(
      std::log(tolfac / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)) +
      nsoff);
  return ns;
}

void set_kernel_shape_given_ns(finufft_spread_opts &spopts, int debug) {
  // Writes kernel shape parameter(s) (beta,...), into spopts, given previously-set
  // kernel info fields in spopts, principally: nspread, upsampfac, kerformula.
  // debug >0 causes stdout reporting.
  int ns       = spopts.nspread;
  double sigma = spopts.upsampfac;
  int kf       = spopts.kerformula;
  // Std shape param formula using ES model for cutoff, eg (4.5) in [FIN] with gamma=1.
  // For PSWF, aligns cut-off (start of aliasing) with freq (c) param. Used below...
  const double beta_cutoff = common::PI * (double)ns * (1.0 - 1.0 / (2.0 * sigma));

  if (kf == 7) {
    /* Shape param formula (designed for K-B), from Beatty et al,
      IEEE Trans Med Imaging, 2005 24(6):799-808. doi:10.1109/TMI.2005.848376
      "Rapid gridding reconstruction with a minimal oversampling ratio".
      This widens in real space, narrowing in k-space a little to exploit continued
      drop just after cutoff. We tweak Beatty's value 0.8 for ns=2 case to lower error.
    */
    double c_Beatty = (ns == 2) ? 0.5 : 0.8; // ns=2 case gives error fac 2 better for KB
    double pis      = common::PI * common::PI;
    spopts.beta = std::sqrt(beta_cutoff * beta_cutoff - c_Beatty / pis);

  } else if (kf == 8) {
    // Std shape param with const shift to exploit a little more tail decay,
    // in the style of Beatty (above) but without the sqrt; a const is better at high ns.
    // This is best for PSWF, within 0.1 digit.
    spopts.beta = beta_cutoff - 0.05; // Libin Lu 1/23/26
                                      // spopts.beta = beta_cutoff; // std param

  } else if (kf == 9) {
    double t = beta_cutoff / common::PI;
    // Marco's LSQ fit using simple functions of t, 1/23/26.
    spopts.beta = ((-0.00149087 * t + 0.0218459) * t + 3.06269) * t - 0.0365245;
  }

  if (debug || spopts.debug) {
    const char *kname = [kf] {
      switch (kf) {
      case 7:
        return "PSWF (Beatty beta)";
      case 8:
        return "PSWF (beta shift)";
      case 9:
        return "PSWF (beta Marco)";
      default:
        return "unknown";
      }
    }();
    printf("[setup_spreadinterp]\tkerformula=%d: %s...\n", kf, kname);
  }
}

} // namespace finufft::kernel

namespace finufft::common {

double smallest_sigma_for_ns(double tol, int dim, int type, int ns_target) {
  // Inverse of theoretical_kernel_ns: ns = ceil(log(tolfac/tol)/(pi*sqrt(1-1/sigma))
  // + 1) equals ns_target first where sqrt(1-1/sigma) = log(tolfac/tol)/(pi*(ns-1)).
  // Result <= MAX_AUTO_UPSAMPFAC, edge_rel_step above that edge: at the edge itself,
  // rounding can give ns_target+1 or eps_alias one ulp above tol.
  constexpr double edge_rel_step = 1e-12;
  const double tolfac = kernel::kernel_tolfac(dim, type);
  if (tol <= 0) return MAX_AUTO_UPSAMPFAC;
  if (tol >= tolfac) return 1.0 + 0.01; // any sigma > 1 meets this tol
  const double u = std::log(tolfac / tol) / ((ns_target - 1.0) * PI);
  if (u >= 1.0) return MAX_AUTO_UPSAMPFAC;
  return std::min((1.0 + edge_rel_step) / (1.0 - u * u), MAX_AUTO_UPSAMPFAC);
}

bool upsampfac_feasible(double sigma, double tol, int dim, int type, bool is_float,
                        double maxN) {
  // Returns eps_alias <= tol && eps_deconv <= tol. [FIN] notation, w = ns, n = nf,
  // N = maxN, frequencies in cycles per fine-grid point:
  //   b = 1 - 1/(2 sigma)   kernel cutoff, (4.5)
  //   a = N/(2n)            top mode; type 3: a = 1/(2 sigma) (nhg_type3)
  //   q = sqrt(b^2 - a^2)
  // eps_alias  = c_d E exp(-pi (w-1) q): (4.7), with the w-1 of the width law (1.7).
  //   c_d = kernel_tolfac (fitted). E = sqrt(x/(1 - e^-x)), x = 2 pi (w-1) a/(n q),
  //   turns the integral in c_d into the sum over the modes near a; type 3: E = 1.
  // eps_deconv = eps_mach exp(pi w (b - q)): deconvolution divides by phihat, so
  //   rounding grows by r_dyn = phihat(0)/phihat(a), (3.21) with beta = pi w b.
  //   [FIN] states (3.21) for type 3; types 1 and 2 divide by the same phihat.
  const int ns_t = kernel::theoretical_kernel_ns(tol, dim, type, sigma);
  const int ns                    = kernel::clamp_kernel_ns(ns_t, sigma, dim, is_float);
  const double nf                 = (double)fine_grid_len(sigma, maxN, ns);
  const double kernel_cutoff_freq = 1.0 - 0.5 / sigma;                         // b
  const double top_mode_freq      = type == 3 ? 0.5 / sigma : 0.5 * maxN / nf; // a
  const double top_mode_decay     =                                            // q
      std::sqrt(kernel_cutoff_freq * kernel_cutoff_freq - top_mode_freq * top_mode_freq);
  const double edge_mode_step =                                                // x
      type == 3 ? 0.0 : 2 * (ns - 1) * PI * top_mode_freq / (top_mode_decay * nf);
  const double edge_factor =                                                   // E
      edge_mode_step > 0 ? std::sqrt(edge_mode_step / -std::expm1(-edge_mode_step)) : 1.0;
  const double eps_alias = kernel::kernel_tolfac(dim, type) * edge_factor *
                           std::exp(-(ns - 1) * PI * top_mode_decay);
  const double eps_mach = is_float ? std::numeric_limits<float>::epsilon()
                                   : std::numeric_limits<double>::epsilon();
  const double eps_deconv =
      eps_mach * std::exp(ns * PI * (kernel_cutoff_freq - top_mode_decay));
  return eps_alias <= tol && eps_deconv <= tol;
}

double analytic_upsampfac(double tol, int dim, int type, bool is_float, double maxN) {
  // Returns min{sigma in [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] : feasible(sigma)},
  // else MAX_AUTO_UPSAMPFAC. E makes feasible not monotone in sigma: scan
  // sigma_k = MIN_AUTO_UPSAMPFAC + k scan_step to the first feasible sigma_k, then
  // bisect (sigma_{k-1}, sigma_k].
  auto feasible = [&](double sigma) {
    return upsampfac_feasible(sigma, tol, dim, type, is_float, maxN);
  };
  // shortcut: misses a feasible interval narrower than scan_step; refine if one shows up.
  constexpr double scan_step = 0.05;
  double lo = MIN_AUTO_UPSAMPFAC, hi = MIN_AUTO_UPSAMPFAC;
  while (!feasible(hi)) {
    if (hi >= MAX_AUTO_UPSAMPFAC) return MAX_AUTO_UPSAMPFAC;
    lo = hi;
    hi = std::min(hi + scan_step, MAX_AUTO_UPSAMPFAC);
  }
  if (hi == MIN_AUTO_UPSAMPFAC) return hi;
  for (int i = 0; i < 40; ++i) { // invariant: feasible(hi) && !feasible(lo)
    const double mid = 0.5 * (lo + hi);
    (feasible(mid) ? hi : lo) = mid;
  }
  return hi;
}

} // namespace finufft::common

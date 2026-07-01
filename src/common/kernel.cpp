#include <cmath>
#include <cstdio>
#include <finufft_common/common.h>
#include <finufft_common/kernel.h>
#include <finufft_common/safe_call.h>
#include <finufft_common/spread_opts.h>
#include <finufft_errors.h>
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

  if (kf == 1 || kf == 2) {
    // ES ("exponential of semicircle" or "exp sqrt"), see [FIN] reference.
    // Used in FINUFFT 2017-2025 (up to v2.4.1). max is 1, as of v2.3.0.
    const double expbeta = std::exp(beta);
    return [beta, expbeta](double z) {
      if (std::abs(z) > 1.0) return 0.0; // restrict support to [-1,1]
      return std::exp(beta * std::sqrt(1.0 - z * z)) / expbeta;
    };
  } else if (kf == 3) {
    // forwards Kaiser--Bessel (KB), normalized to max of 1.
    // std::cyl_bessel_i is from <cmath>, expects double. See src/common/utils.cpp
    const double besselbeta = common::cyl_bessel_i(0, beta);
    return [beta, besselbeta](double z) {
      if (std::abs(z) > 1.0) return 0.0; // restrict support to [-1,1]
      return common::cyl_bessel_i(0, beta * std::sqrt(1.0 - z * z)) / besselbeta;
    };
  } else if (kf == 4) {
    // continuous (deplinthed) KB, as in Barnett SIREV 2022, normalized to max nearly 1
    const double besselbeta = common::cyl_bessel_i(0, beta);
    return [beta, besselbeta](double z) {
      if (std::abs(z) > 1.0) return 0.0; // restrict support to [-1,1]
      return (common::cyl_bessel_i(0, beta * std::sqrt(1.0 - z * z)) - 1.0) / besselbeta;
    };
  } else if (kf == 5) {
    const double coshbeta = std::cosh(beta);
    return [beta, coshbeta](double z) {
      if (std::abs(z) > 1.0) return 0.0; // restrict support to [-1,1]
      return std::cosh(beta * std::sqrt(1.0 - z * z)) / coshbeta;
    }; // normalized cosh-type of Rmk. 13 [FIN]
  } else if (kf == 6) {
    const double coshbeta = std::cosh(beta);
    return [beta, coshbeta](double z) {
      if (std::abs(z) > 1.0) return 0.0; // restrict support to [-1,1]
      return (std::cosh(beta * std::sqrt(1.0 - z * z)) - 1.0) / coshbeta;
    }; // Potts-Tasche cont cosh-type
  } else if (kf >= 7 && kf <= 9) {
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
  // PER_DIM^(dim-1) compensates empirical per-dimension error worsening; type 3 is TYPE3
  // worse (affects the outer spread, not the inner type-2). Shared by
  // theoretical_kernel_ns, smallest_sigma_for_ns and analytic_upsampfac so they cannot
  // drift apart (a past source of type-3 inconsistency). PER_DIM and TYPE3 are distinct
  // empirical fudge factors that happen to share the value 1.4. Integer-power multiply
  // rather than std::pow.
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

int theoretical_kernel_ns(double tol, int dim, int type, int debug,
                          const finufft_spread_opts &spopts) {
  // returns ideal preferred spread width (ns, a.k.a. w) using convergence rate,
  // in exact arithmetic, to achieve requested tolerance tol. Possibly uses
  // other parameters in spopts (upsampfac, kerformula,...). No clipping of ns
  // to valid range done here. Input upsampfac must be >1.0.
  int ns       = 0;
  double sigma = spopts.upsampfac;

  if (spopts.kerformula == 1) // ES legacy ns choice (v2.4.1, ie 2025, and before)
    if (sigma == 2.0)
      ns = (int)std::ceil(std::log10(10.0 / tol));
    else
      ns = (int)std::ceil(
          std::log(1.0 / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)));
  else { // generic formula for PSWF-like kernels. Currently for kf=8, PSWF (beta shift)
    // tweak tolfac and nsoff for user tol matching (& tolsweep passing) over sigma...
    const double tolfac = kernel_tolfac(dim, type);
    const double nsoff = 1.0; // width offset (helps balance err over sigma range)
    ns                 = (int)std::ceil(
        std::log(tolfac / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)) +
        nsoff);
  }
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

  // these strings must match: kernel_definition(), the above, and the below
  const char *kernames[] = {"default",
                            "ES (legacy beta)",        // 1
                            "ES (Beatty beta)",        // 2
                            "KB (Beatty beta)",        // 3
                            "cont-KB (Beatty beta)",   // 4
                            "cosh-type (Beatty beta)", // 5
                            "cont cosh (Beatty beta)", // 6
                            "PSWF (Beatty beta)",      // 7
                            "PSWF (beta shift)",       // 8
                            "PSWF (beta Marco)"};      // 9
  if (kf == 1) {
    // Exponential of Semicircle (ES), the legacy logic, from 2017, used to v2.4.1
    double betaoverns = 2.30;
    if (ns == 2)
      betaoverns = 2.20;
    else if (ns == 3)
      betaoverns = 2.26;
    else if (ns == 4)
      betaoverns = 2.38;         // in hindsight this value was too large
    spopts.beta = betaoverns * (double)ns;
    if (sigma != 2.0) {          // low-sigma option, introduced v1.0 (2018-2025)
      const double gamma = 0.97; // safety factor, from [FIN] paper
      spopts.beta        = gamma * beta_cutoff;
    }

  } else if (kf >= 2 && kf <= 7) {
    /* Shape param formula (designed for K-B), from Beatty et al,
      IEEE Trans Med Imaging, 2005 24(6):799-808. doi:10.1109/TMI.2005.848376
      "Rapid gridding reconstruction with a minimal oversampling ratio".
      This widens in real space, narrowing in k-space a little to exploit continued
      drop just after cutoff. We tweak Beatty's value 0.8 for ns=2 case to lower error.
    */
    double c_Beatty = (ns == 2) ? 0.5 : 0.8; // ns=2 case gives error fac 2 better for KB
    double pis      = common::PI * common::PI;
    spopts.beta     = std::sqrt(beta_cutoff * beta_cutoff - c_Beatty / pis);
    // Expts show beta_cutoff with KB is 1/3-digit worse than Beatty, similar to ES.
    // In fact, in wsweepkerrcomp.m on KB we find beta_cutoff-0.17 is indistinguishable.
    // This is analogous to a safety factor of >0.99 around ns=10 (0.97 was too small)

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
    const char *kname = (kf >= 0 && kf <= 9) ? kernames[kf] : "unknown";
    printf("[setup_spreadinterp]\tkerformula=%d: %s...\n", kf, kname);
  }
}

} // namespace finufft::kernel

namespace finufft::common {

double smallest_sigma_for_ns(double tol, int dim, int type, int ns_target) {
  // Invert theoretical_kernel_ns's generic branch (kerformula=0):
  //   ns = ceil( log(tolfac/tol) / (pi*sqrt(1-1/sigma)) + nsoff ),  nsoff = 1.0.
  // Smallest sigma reaching ns_target sets the ceil argument to ns_target, so
  // sqrt(1-1/sigma) = log(tolfac/tol) / (pi*(ns_target-1)). Stays in lockstep with
  // the forward formula via kernel_tolfac and the matching nsoff.
  // Inverts only the generic (kerformula=0) branch of theoretical_kernel_ns, which is
  // what the upsampfac heuristic uses; a plan with kerformula>0 (e.g. legacy ES) uses a
  // different width law, but the heuristic does not call this for those, so they cannot
  // disagree.
  const double tolfac = kernel::kernel_tolfac(dim, type);
  // Clamped to MAX_AUTO_UPSAMPFAC (the performance range this inverse spans;
  // lowest_sigma re-applies the tighter accuracy cap).
  if (tol <= 0) return MAX_AUTO_UPSAMPFAC;
  if (tol >= tolfac) return MIN_CHECK_SIGMA + 0.01;
  const double u = std::log(tolfac / tol) / ((ns_target - 1.0) * PI);
  if (u >= 1.0) return MAX_AUTO_UPSAMPFAC;
  return std::min(1.0 / (1.0 - u * u), MAX_AUTO_UPSAMPFAC);
}

double lowest_sigma(double tol, int dim, int ns, double eps_mach, double gridlen) {
  // Minimum sigma achieving requested tol. Two regimes:
  //
  //   r = tol / eps_round,  eps_round = 0.48 * eps_mach * N.
  //
  // Kernel regime (r >= 10): pure analytical inversion of the aliasing formula,
  //   exact to ~0.0001 sigma (validated in find_sigma_bound.py).
  //
  // Transition regime (r < 10): the rounding floor matters. A polynomial
  //   correction in 1/r is added to sigma_pure to account for the floor:
  //     sigma = sigma_pure + a2/r^2 + a1/r + a0
  //   Coefficients fit by least-squares on empirical sigma_min data across
  //   N=50..5000, types 1-3, dim 1 (see devel/find_sigma_bound.py).
  //   Separate coefficients for ns>8 (double) and ns<=8 (float).
  const double eps_round = 0.48 * eps_mach * gridlen;
  const double r = tol / eps_round;
  if (r <= 0.5) return MAX_CHECK_SIGMA;
  // type=1 here: the floor correction below is type-agnostic and check_sigma's
  // feasibility view uses the type-1 (type 1/2) kernel prefactor.
  const double sigma_pure = smallest_sigma_for_ns(tol, dim, 1, ns);
  if (r >= 10.0)
    return std::min(sigma_pure, MAX_CHECK_SIGMA); // accuracy cap (constants.h)
  // Poly(1/r) correction coefficients {a2, a1, a0}, fit across all types:
  const double a2 = ns > 8 ? 0.014 : 0.555;
  const double a1 = ns > 8 ? 0.291 : -0.290;
  const double a0 = ns > 8 ? -0.043 : 0.071;
  const double inv_r = 1.0 / r;
  const double correction = (a2 * inv_r + a1) * inv_r + a0;
  return std::min(sigma_pure + std::max(correction, 0.0), MAX_CHECK_SIGMA);
}

bool upsampfac_feasible(double sigma, double tol, int dim, int type, double eps_mach,
                        int max_nspread, bool is_float, double maxN) {
  // Purpose: returns whether the plan pipeline would ACCEPT this upsampfac at this tol,
  // i.e. "feasible" = makeplan/check_sigma would neither throw nor silently lose
  // accuracy. The upsampfac heuristic only ever proposes sigmas that pass this, so its
  // pick always survives the real plan. maxN = largest mode count over dims (binds the
  // fine grid); unused for type 3.
  //
  // Mirrors the plan pipeline's gates: clamp_kernel_ns covers the setup_spreadinterp
  // width cap and the float catastrophic-cancellation guard (a clamped width would
  // throw there or silently lose accuracy). Type 3 has no check_sigma, so that is its
  // only gate; types 1/2 must also pass check_sigma's lowest_sigma test on the fine
  // grid set_nf_type12 would build at this sigma.
  // NB this assumes the generic (kerformula=0) width formula; see the so.kerformula=0
  // below. A plan run with opts.spread_kerformula>0 may need a slightly different ns, but
  // the heuristic and check_sigma both use the default kernel, so they stay consistent.
  finufft_spread_opts so{};
  so.kerformula = 0; // generic (PSWF-like) ns formula in theoretical_kernel_ns
  so.upsampfac = sigma;
  const int ns_t = kernel::theoretical_kernel_ns(tol, dim, type, 0, so);
  const int ns = kernel::clamp_kernel_ns(ns_t, sigma, max_nspread, is_float);
  if (ns < ns_t) return false;
  if (type == 3) return true;
  // fine-grid length as set_nf_type12 builds it (largest dim binds).
  const long nf = fine_grid_len(sigma, maxN, ns);
  return lowest_sigma(tol, dim, ns, eps_mach, (double)nf) <= sigma;
}

double analytic_upsampfac(double tol, int dim, int type, double eps_mach, int max_nspread,
                          bool is_float, double maxN) {
  // Smallest sigma in [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] the plan pipeline accepts
  // (via upsampfac_feasible), found by bisection. This is the optimum directly when the
  // FFT dominates (always type 3; sparse types 1/2) and is the lower end of the
  // heuristic's candidate range otherwise. Returns MAX_AUTO_UPSAMPFAC when tol is
  // unachievable (the largest sigma, which minimizes both ns and the roundoff
  // amplification, getting closest to tol; the plan pipeline then reports the error).
  // maxN = largest mode count over dims (1 for type 3).
  auto feasible = [&](double sigma) {
    return upsampfac_feasible(sigma, tol, dim, type, eps_mach, max_nspread, is_float,
                              maxN);
  };

  // feasible() is not exactly monotone (integer ns and 235-smooth grid steps), but
  // any flicker only costs a negligibly larger feasible sigma, never correctness:
  // the returned value was itself accepted by feasible().
  if (feasible(MIN_AUTO_UPSAMPFAC)) return MIN_AUTO_UPSAMPFAC;
  if (!feasible(MAX_AUTO_UPSAMPFAC))
    return MAX_AUTO_UPSAMPFAC; // pipeline reports the error
  double lo = MIN_AUTO_UPSAMPFAC, hi = MAX_AUTO_UPSAMPFAC;
  for (int i = 0; i < 40; ++i) { // invariant: feasible(hi) && !feasible(lo)
    const double mid = 0.5 * (lo + hi);
    (feasible(mid) ? hi : lo) = mid;
  }
  return hi; // smallest feasible sigma to ~1e-12 resolution
}

} // namespace finufft::common

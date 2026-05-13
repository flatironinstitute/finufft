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

double kernel_definition(const finufft_spread_opts &spopts, const double z) {
  /* The spread/interp kernel phi_beta(z) function on standard interval z in [-1,1].
     This evaluation does not need to be fast; it is used *only* for polynomial
     interpolation via Horner coeffs (the interpolant is evaluated fast).
     It can thus always be double-precision. No analytic Fourier transform pair is
     needed, thanks to numerical quadrature in finufft_core:onedim*; playing with
     new kernels is thus very easy.
    Inputs:
    z      - real ordinate on standard interval [-1,1]. Handling of edge cases
            at or near +-1 is no longer crucial, because precompute_horner_coeffs
            (the only user of this function) has interpolation nodes in (-1,1).
    spopts - spread_opts struct containing fields:
      beta        - shape parameter for ES, KB, or other prolate kernels
                    (a.k.a. c parameter in PSWF).
      kerformula  - positive integer selecting among kernel function types; see
                    docs in the code below.
                    (More than one value may give the same type, to allow
                    kerformula also to select a parameter-choice method.)
                    Note: the default 0 (in opts.spread_kerformula) is invalid here;
                    selection of a >0 kernel type must already have happened.
    Output: phi(z), as in the notation of original 2019 paper ([FIN] in the docs).

    Notes: 1) no normalization of max value or integral is needed, since any
            overall factor is cancelled out in the deconvolve step. However,
            values as large as exp(beta) have caused floating-pt overflow; don't
            use them.
    Barnett rewritten 1/13/26 for double on [-1,1]; based on Barbone Dec 2025.
  */
  if (std::abs(z) > 1.0) return 0.0;           // restrict support to [-1,1]
  double beta = spopts.beta;                   // get shape param
  double arg  = beta * std::sqrt(1.0 - z * z); // common argument for exp, I0, etc
  int kf      = spopts.kerformula;

  if (kf == 1 || kf == 2)
    // ES ("exponential of semicircle" or "exp sqrt"), see [FIN] reference.
    // Used in FINUFFT 2017-2025 (up to v2.4.1). max is 1, as of v2.3.0.
    return std::exp(arg) / std::exp(beta);
  else if (kf == 3)
    // forwards Kaiser--Bessel (KB), normalized to max of 1.
    // std::cyl_bessel_i is from <cmath>, expects double. See src/common/utils.cpp
    return common::cyl_bessel_i(0, arg) / common::cyl_bessel_i(0, beta);
  else if (kf == 4)
    // continuous (deplinthed) KB, as in Barnett SIREV 2022, normalized to max nearly 1
    return (common::cyl_bessel_i(0, arg) - 1.0) / common::cyl_bessel_i(0, beta);
  else if (kf == 5)
    return std::cosh(arg) / std::cosh(beta); // normalized cosh-type of Rmk. 13 [FIN]
  else if (kf == 6)
    return (std::cosh(arg) - 1.0) / std::cosh(beta); // Potts-Tasche cont cosh-type
  else if (kf >= 7 && kf <= 9)
    return common::pswf(beta, z); // prolate (PSWF) Psi_0, normalized to 1 at z=0
  else {
    fprintf(stderr, "[%s] unknown spopts.kerformula=%d\n", __func__, spopts.kerformula);
    throw finufft::exception(FINUFFT_ERR_KERFORMULA_NOTVALID);
    return std::numeric_limits<double>::quiet_NaN(); // never gets here, non-signalling
  }
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
    double tolfac = 0.18 * std::pow(1.4, (double)(dim - 1));
    // (here the dim power compensated for empirical worsening by dim)
    if (type == 3) // compensate for type 3 worse (affects outer spread, not inner t2)
      tolfac *= 1.4;
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

// Invert the kernel aliasing formula from theoretical_kernel_ns (kf>=2):
//   tol = tolfac * exp(-(ns - 1) * pi * u),  u = sqrt(1 - 1/sigma).
// Returns sigma in [MINSIGMA, MAXSIGMA]. Barbone 4/3/26.
static double invert_kernel_sigma(double tol, int ns, int dim) {
  const double tolfac = 0.18 * std::pow(1.4, dim - 1);
  if (tol <= 0) return MAXSIGMA;
  if (tol >= tolfac) return MINSIGMA + 0.01;
  const double u = std::log(tolfac / tol) / ((ns - 1.0) * PI);
  if (u >= 1.0) return MAXSIGMA;
  return std::min(1.0 / (1.0 - u * u), MAXSIGMA);
}

double estimated_tol(double sigma, int ns, int dim, double eps_mach, double gridlen) {
  // Estimated relative ell-2 NUFFT error = eps_alias + eps_round.
  //   eps_alias = tolfac * exp(-(ns-1)*pi*u)     [BAR] Thm 1, [FIN] Thm 7
  //   eps_round = 0.48 * eps_mach * N            [FIN] Remark 9
  // Barbone 4/3/26.
  // References:
  // [FIN] Barnett, Magland & af Klinteberg, SISC 2019, arxiv:1808.06736
  // [BAR] Barnett, Appl. Comput. Harmon. Anal. 2021, arxiv:2001.09405
  const double tolfac    = 0.18 * std::pow(1.4, dim - 1);
  const double u         = std::sqrt(1.0 - 1.0 / sigma);
  const double eps_alias = tolfac * std::exp(-(ns - 1.0) * PI * u);
  return eps_alias + 0.48 * eps_mach * gridlen;
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
  //
  // Barbone 4/3/26.
  const double eps_round = 0.48 * eps_mach * gridlen;
  const double r         = tol / eps_round;
  if (r <= 0.5) return MAXSIGMA;
  const double sigma_pure = invert_kernel_sigma(tol, ns, dim);
  if (r >= 10.0) return sigma_pure;
  // Poly(1/r) correction coefficients {a2, a1, a0}, fit across all types:
  const double a2         = ns > 8 ? 0.014 : 0.555;
  const double a1         = ns > 8 ? 0.291 : -0.290;
  const double a0         = ns > 8 ? -0.043 : 0.071;
  const double inv_r      = 1.0 / r;
  const double correction = (a2 * inv_r + a1) * inv_r + a0;
  return std::min(sigma_pure + std::max(correction, 0.0), MAXSIGMA);
}

} // namespace finufft::common

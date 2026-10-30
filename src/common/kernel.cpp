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

double kernel_tolfac(int dim) {
  constexpr double TOLFAC_1D      = 0.18; // 1D base prefactor
  constexpr double TOLFAC_PER_DIM = 1.4;  // per-extra-dim worsening, ^(dim-1)
  constexpr auto ipow = [](double base, int n) {
    double r = 1.0;
    for (int i = 0; i < n; ++i) r *= base;
    return r;
  };
  return TOLFAC_1D * ipow(TOLFAC_PER_DIM, dim - 1);
}

int ns_for_sigma(double tol, int dim, double sigma) {
  return aliasing_ns(tol, kernel_tolfac(dim), alias_decay(sigma, 0.5 / sigma));
}

// Sigma needs 3 digits. sigma_edge returns the smallest 3-digit sigma above 1/t, or
// infinity above MAX_AUTO_UPSAMPFAC. Each inverse solves for t = 1/sigma: no pole.
static double sigma_edge(double t) {
  if (t * common::MAX_AUTO_UPSAMPFAC < 1.0)
    return std::numeric_limits<double>::infinity();
  return (std::floor(1000.0 / t) + 1.0) / 1000.0;
}

double sigma_for_ns(double tol, int dim, int ns) {
  // u: the alias_decay that the width ns needs at tol.
  const double u =
      std::max(0.0, std::log(kernel_tolfac(dim) / tol) / ((ns - 1) * common::PI));
  return sigma_edge(1.0 - u * u);
}

double sigma_for_deconv(double eps_mach, int dim, int ns, double tol) {
  // r: the largest cutoff - alias_decay that meets tol, at most its value 1/2 at sigma 1.
  const double r = std::min(0.5, std::log(tol / (dim * eps_mach)) / (ns * common::PI));
  // t solves cutoff - alias_decay = r.
  return sigma_edge(r < 0.0 ? 0.0 : 2.0 * (std::sqrt(2.0 * r) - r));
}

bool upsampfac_feasible(double sigma, double tol, int dim, int type, bool is_float,
                        double maxN) {
  const int ns          = kernel_width_at(tol, dim, sigma, is_float);
  const double nf       = (double)common::fine_grid_len(sigma, maxN, ns);
  // Type 3 has no N: its top mode is fixed by sigma (nhg_type3).
  const double a        = type == 3 ? 0.5 / sigma : 0.5 * maxN / nf;
  const double decay    = alias_decay(sigma, a);
  // E: on a short grid the modes near a are discrete, and kernel_tolfac fits the
  // integral over them.
  const double x        = type == 3 ? 0.0 : 2 * (ns - 1) * common::PI * a / (decay * nf);
  const double E        = x > 0 ? std::sqrt(x / -std::expm1(-x)) : 1.0;
  const double eps_mach = is_float ? std::numeric_limits<float>::epsilon()
                                   : std::numeric_limits<double>::epsilon();
  return ns >= aliasing_ns(tol, kernel_tolfac(dim) * E, decay) &&
         deconv_error(eps_mach, dim, ns, sigma, decay) <= tol;
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

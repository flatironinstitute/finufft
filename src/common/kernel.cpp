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

static double kernel_tolfac(int dim, int type) {
  // Tolerance prefactor in the below kernel aliasing law. (Fitted Barnett Jan 2026).
  // PER_DIM^(dim-1) covers the measured error growth per dimension; TYPE3 covers the
  // type-3 outer spread. The two are separate fits that share the value 1.4. The width
  // law and its inverse both read it, so they cannot disagree.
  constexpr double TOLFAC_1D      = 0.18; // 1D type-1/2 base prefactor
  constexpr double TOLFAC_PER_DIM = 1.4;  // per-extra-dim worsening, ^(dim-1)
  constexpr double TOLFAC_TYPE3   = 1.4;  // type-3 outer-spread extra worsening
  return TOLFAC_1D * std::pow(TOLFAC_PER_DIM, dim - 1) * (type == 3 ? TOLFAC_TYPE3 : 1.0);
}

int theoretical_kernel_ns(double tol, int dim, int type, double sigma) {
  /* Spread/interp kernel width ns (aka w) that meets tol at upsampfac sigma > 1 in exact
     arithmetic, via a kernel aliasing law. The law is given by keeping only the
     exponential term in equation (1.7) or (4.6) in the [FIN] publication, then inverting
     for w given tol (ie eps_infty) and sigma.
     A prefactor (the above kernel_tolfac), and w offset (NSOFF), were hand fitted so ns
     with the PSWF kernel with default shape parameter gives errors lying below dotted
     lines in tolsweeptest.m and hence passes tolsweeptest.cpp (by Barnett, Jan 2026).
     The aliasing kernel error model is thus (tol being relative tolerance),
             tol ~ eps_infty = tolfac.exp(-pi*(w-NSOFF)*sqrt(1-1/sigma)) .
     Notes:
     1) min_sigma_for_aliasing_error() below must be the inverse of this function
        (without the ceil operation).
     2) The use of ceil to get the integer output is crucial (see testutils.cpp).
     3) Any similarity of (w-NSOFF) to (w-1) in eqn (1.7) is coincidental, since hand-fit.
     Barnett Jan 2026; tolfac split out Fall 2026; updated docs 11/8/26.
  */
  const double tolfac = kernel_tolfac(dim, type);
  int ns              = (int)std::ceil(
      std::log(tolfac / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)) +
      common::NSOFF);
  return ns;
}

// Widest ns that does not lose accuracy to catastrophic cancellation in float, at low
// sigma. Used below.
static int float_max_ns_cc(double sigma, int dim) {
  constexpr double UPSAMPFAC_STEP = 1.0 / 0.075;
  return 8 + static_cast<int>(std::floor(
                 std::max(0.0, (sigma - 0.975 - 0.125 * dim) * UPSAMPFAC_STEP)));
}

// Clip kernel width ns to compiled kernel evaluator limits, and apply max limit
// due to catastrophic cancellation.
int clamp_kernel_ns(int ns, double upsampfac, int dim, bool is_float) {
  using common::MAX_NSPREAD;
  const int max_ns = is_float ? MAX_NSPREAD<float> : MAX_NSPREAD<double>;
  ns               = std::max(common::MIN_NSPREAD, ns);
  ns               = std::min(ns, max_ns);
  // Float catastrophic-cancellation guard. Both sides are int, so a huge upsampfac
  // cannot overflow.
  if (is_float) ns = std::min(ns, float_max_ns_cc(upsampfac, dim));
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

double min_sigma_for_aliasing_error(double tol, int dim, int type, int ns) {
  // An analytic inverse for the (ns,sigma) relationship in theoretical_kernel_ns()
  // above, without its final ceil operation. Returns +infinity if no sigma can
  // achieve tol (eps_infty). See testutils.cpp for test.
  const double u =
      std::log(kernel_tolfac(dim, type) / tol) / ((ns - common::NSOFF) * common::PI);
  // crucial that the following is +infinity (since needs to be > MAX_AUTO_UPSAMPFAC)
  if (!(u < 1.0)) return std::numeric_limits<double>::infinity(); // unachievable
  if (u <= 0.0) return 1.0;
  return 1.0 / (1.0 - u * u);
}

double min_sigma_for_deconv_error(double tol, int dim, int ns, double eps_mach) {
  /* Analytic solution of the sigma that matches tol in an error model for
     rounding amplification due to division by kernel FT at the largest freq N/2.
     Its main use is in stopping too small a sigma being auto-chosen by
     best_sigma_for_tol(), and in deciding if to throw FINUFFT_ERR_EPS_TOO_SMALL
     in check_sigma().
     The formula inverted for sigma is
       tol ~ c * eps_mach * r_dyn(sigma,w),
     where c seems to be set to dim, the dimension (1,2, or 3), and
     r_dyn(sigma), the 1D deconvolution dynamic range, is (3.21) of the [FIN]
     paper,
       r_dyn ~ exp(beta - sqrt(beta^2 - (pi.w/2sigma)^2) ).
     Here w is ns, the width, and beta (shape param) also depends on sigma.
     We simplify this using (4.5) for beta (with gamma=1), the beta_cutoff
     formula in the above set_kernel_shape_given_ns(). This gives
       r_dyn ~ exp(beta) * exp(-pi.w.sqrt(1 - 1/sigma))
             ~ exp(beta) * tol,
     noting the aliasing formulae (1.7) or (4.6) used to match tol.
     Taking the log gives the r_dyn interpretation sketched in Fig. 3.1(b).

     *** to-do: understand the below AI-generated sigma inversion formula!

     Notes: theory would predict (r_dyn)^dim instead of c=dim above (to-do:
     investigate this dim-dependence).
     However, Barnett believes that the exact result of this formula is not
     crucial, because the empirically-fitted float_max_ns_cc() above is what
     actually prevents deconv CC (large ns, small sigma, float prec case).
  */
  const double v = 2.0 * std::log(tol / (dim * eps_mach)) / (common::PI * ns);
  if (!(v > 0.0)) return std::numeric_limits<double>::infinity();
  const double q = std::min(std::sqrt(v), 1.0);
  return 1.0 / (q * (2.0 - q));
}

} // namespace finufft::kernel

namespace finufft::common {

// For heuristics only: a mirror of makeplan:set_nf_type12() except without a plan.
// To-do: consider having makeplan call this as SSOT, since a pain to maintain two
// functions!
BIGINT fine_grid_len(double sigma, double n_modes, int ns) {
  return next235(std::max(BIGINT(std::ceil(sigma * n_modes)), BIGINT(2 * ns)), 2);
}

std::tuple<BIGINT, double, double> nhg_type3(double sigma, double X, double S, int ns,
                                             BIGINT max_nf) {
  /* Sets nf (number fine grid pts), h (fine grid spacing), gamma (x_j rescaling factor)
     for type 3 transforms only. These are returned in a tuple (nf, h, gam).
     Satisfying the constraints for type 3 is hard-won and elaborate; see Sec 3.3 [FIN].
     This is now the single source of truth for the set_nhg_type3 task, used by:
     CPU (setpts.hpp), GPU (cuda/makeplan.cu), and the type-3 upsampfac CPU cost model
     (src/heuristics.cpp).

     Inputs:
       X and S are the xj and sk interval half-widths respectively.
       ns = kernel width
       max_nf is an upper limit, historically to protect next235, and used the same
          MAX_NF fine-grid allocation guard. (to-do: check if needed).
     Outputs:
       nf is the size of upsampled grid for a given single dimension.
       h is the grid spacing = 2pi/nf
       gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).

     History: Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
     New logic 6/12/17. Extracted Barbone 6/26 to unify logic (SSOT); recovered doc
     Barnett 11/8/26.
  */
  const int nss = ns + 1; // since ns may be odd
  double Xs = X, Ss = S;  // tweak so X*S>=1, handling X=0 and/or S=0
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
  if (nf < max_nf) nf = next235(nf, 2);                // else too big; spread will error
  return {nf, 2.0 * PI / (double)nf, (double)nf / (2.0 * sigma * Ss)};
}

double rounding_error(double eps_mach, double mode_diagonal) {
  // Error model for inevitable rounding error amplification due to the condition
  // number of the NUDFT problem (Remark 9 in [FIN] paper; also see "math" section of
  // docs/trouble.rst). The prefactor is fitted (Barbone, Fall 2026).
  // mode_diagonal is the Pythagorean sum of the Ni mode dimensions (i=1,..,dim).
  // eps_mach is machine epsilon as usual.
  return ROUND_FAC * eps_mach * mode_diagonal;
}

} // namespace finufft::common

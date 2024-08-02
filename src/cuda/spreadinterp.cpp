#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <cufinufft/defs.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

#include <finufft_errors.h>

namespace cufinufft {
namespace spreadinterp {

template<typename T>
int setup_spreader(finufft_spread_opts &opts, T eps, T upsampfac, int kerevalmeth)
// Initializes spreader kernel parameters given desired NUFFT tolerance eps,
// upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), and ker eval meth
// (etiher 0:exp(sqrt()), 1: Horner ppval).
// Also sets all default options in finufft_spread_opts. See cnufftspread.h for opts.
// Must call before any kernel evals done.
// Returns: 0 success, 1, warning, >1 failure (see error codes in utils.h)
{
  if (upsampfac != 2.0 && upsampfac != 1.25) { // nonstandard sigma
    if (kerevalmeth == 1) {
      fprintf(stderr,
              "[%s] nonstandard upsampfac=%.3g cannot be handled by kerevalmeth=1\n",
              __func__, upsampfac);
      return FINUFFT_ERR_HORNER_WRONG_BETA;
    }
    if (upsampfac <= 1.0) {
      fprintf(stderr, "[%s] error: upsampfac=%.3g is <=1.0\n", __func__, upsampfac);
      return FINUFFT_ERR_UPSAMPFAC_TOO_SMALL;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (upsampfac > 4.0)
      fprintf(stderr, "[%s] warning: upsampfac=%.3g is too large to be beneficial!\n",
              __func__, upsampfac);
  }

  // defaults... (user can change after this function called)
  opts.spread_direction = 1; // user should always set to 1 or 2 as desired
  opts.upsampfac        = upsampfac;

  // as in FINUFFT v2.0, allow too-small-eps by truncating to eps_mach...
  int ier = 0;

  constexpr T EPSILON = std::numeric_limits<T>::epsilon();
  if (eps < EPSILON) {
    fprintf(stderr, "setup_spreader: warning, increasing tol=%.3g to eps_mach=%.3g.\n",
            (double)eps, (double)EPSILON);
    eps = EPSILON;
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }

  // Set kernel width w (aka ns) and ES kernel beta parameter, in opts...
  int ns = std::ceil(-log10(eps / (T)10.0)); // 1 digit per power of ten
  if (upsampfac != 2.0)                      // override ns for custom sigma
    ns = std::ceil(-log(eps) / (T(M_PI) * sqrt(1 - 1 / upsampfac))); // formula,
                                                                     // gamma=1
  ns = std::max(2, ns);   // we don't have ns=1 version yet
  if (ns > MAX_NSPREAD) { // clip to match allocated arrays
    fprintf(stderr,
            "%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; "
            "clipping to max %d.\n",
            __func__, upsampfac, (double)eps, ns, MAX_NSPREAD);
    ns  = MAX_NSPREAD;
    ier = FINUFFT_WARN_EPS_TOO_SMALL;
  }
  opts.nspread      = ns;
  opts.ES_halfwidth = T(ns * .5); // constants to help ker eval (except Horner)
  opts.ES_c         = 4.0 / (T)(ns * ns);

  T betaoverns = 2.30;            // gives decent betas for default sigma=2.0
  if (ns == 2) betaoverns = 2.20; // some small-width tweaks...
  if (ns == 3) betaoverns = 2.26;
  if (ns == 4) betaoverns = 2.38;
  if (upsampfac != 2.0) { // again, override beta for custom sigma
    T gamma    = 0.97;    // must match devel/gen_all_horner_C_code.m
    betaoverns = gamma * T(M_PI) * (1 - 1 / (2 * upsampfac)); // formula based on
                                                              // cutoff
  }
  opts.ES_beta = betaoverns * (T)ns; // set the kernel beta parameter
  // fprintf(stderr,"setup_spreader: sigma=%.6f, chose ns=%d
  // beta=%.6f\n",(double)upsampfac,ns,(double)opts.ES_beta);
  // // user hasn't set debug yet
  return ier;
}

template int setup_spreader(finufft_spread_opts &opts, float eps, float upsampfac,
                            int kerevalmeth);
template int setup_spreader(finufft_spread_opts &opts, double eps, double upsampfac,
                            int kerevalmeth);
template float evaluate_kernel(float x, const finufft_spread_opts &opts);
template double evaluate_kernel(double x, const finufft_spread_opts &opts);

} // namespace spreadinterp
} // namespace cufinufft

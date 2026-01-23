#include <algorithm>
#include <cmath>
#include <cstdio>
#include <finufft_common/common.h>
#include <finufft_common/kernel.h>
#include <finufft_common/spread_opts.h>

// this module uses finufft_spread_opts but does not know about FINUFFT_PLAN class
// nor finufft_opts. This allows it to be used by CPU & GPU.

namespace finufft::kernel {

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
  else { // generic formula for PSWF-like kernels.
    // tweak tolfac and nsoff for user tol matching (& tolsweep passing)...
    const double tolfac = (type == 3) ? 0.5 : 0.3; // only applies to outer of type 3
    const double nsoff  = 0.8; // width offset (helps balance err over sigma range)
    ns                  = (int)std::ceil(
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

  // these strings must match: kernel_definition(), the above, and the below
  const char *kernames[] = {"default",
                            "ES (legacy beta)",        // 1
                            "ES (Beatty beta)",        // 2
                            "KB (Beatty beta)",        // 3
                            "cont-KB (Beatty beta)",   // 4
                            "cosh-type (Beatty beta)", // 5
                            "cont cosh (Beatty beta)", // 6
                            "PSWF (Beatty beta)",      // 7
                            "PSWF (tuned beta)"};      // 8
  if (kf == 1) {
    // Exponential of Semicircle (ES), the legacy logic, from 2017, used to v2.4.1
    double betaoverns = 2.30;
    if (ns == 2)
      betaoverns = 2.20;
    else if (ns == 3)
      betaoverns = 2.26;
    else if (ns == 4)
      betaoverns = 2.38;

    if (sigma != 2.0) { // low-sigma option, introduced v1.0 (2018-2025)
      const double gamma = 0.97;
      betaoverns         = gamma * common::PI * (1.0 - 1.0 / (2.0 * sigma));
    }
    spopts.beta = betaoverns * (double)ns;

  } else if (kf >= 2 || kf <= 7) {
    // Shape param formula (designed for K-B), from Beatty et al,
    // IEEE Trans Med Imaging, 2005 24(6):799-808. doi:10.1109/TMI.2005.848376
    // "Rapid gridding reconstruction with a minimal oversampling ratio".
    double t    = (double)ns * (1.0 - 1.0 / (2.0 * sigma));
    double c_beatty = (ns == 2) ? 0.5 : 0.8; // Beatty but tweak ns=2: KB err fac 2 better
    spopts.beta = common::PI * std::sqrt(t * t - c_beatty); // just below std cutoff PI*t
    // in fact, in wsweepkerrcomp.m on KB we find beta=pi*t-0.17 is indistinguishable.
    // This is analogous to a safety factor of >0.99 around ns=10 (0.97 was too small)
  } else if (kf == 8) {
    // Marco's LSQ fit of a functional form for beta(sigma,ns)...
    const double sigmasq = sigma * sigma;
    const double A       = -0.19638654 + 2.31685991 * sigma - 0.53110991 * sigmasq;
    const double B       = 2.29051829 - 2.82937718 * sigma + 0.91381927 * sigmasq;
    const double C       = -0.61525503;
    spopts.beta          = A * (double)ns + B + C / (double)ns;
  }

  // Plain shape param formula using std model for cutoff: (4.5) in [FIN], gamma=1:
  // spopts.beta = common::PI * (double)ns * (1.0 - 1.0 / (2.0 * sigma));
  // Expts show this formula with KB is 1/3-digit worse than Beatty, similar to ES.

  if (debug || spopts.debug)
    printf("[setup_spreadinterp]\tkerformula=%d: %s...\n", kf, kernames[kf]);
}

} // namespace finufft::kernel

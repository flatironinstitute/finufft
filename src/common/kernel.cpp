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
  // to valid range done here.
  int ns;
  double sigma = spopts.upsampfac;

  if (spopts.kerformula == 1) // ES legacy ns choice (v2.4.1, ie 2025, and before)
    if (sigma == 2.0)
      ns = (int)std::ceil(std::log10(10.0 / tol));
    else
      ns = (int)std::ceil(
          std::log(1.0 / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)));
  else {                         // generic formula for PSWF-like kernels
    const double fudgefac = 1.0; // *** todo: tweak it, per kerformula
    ns                    = (int)std::ceil(
        std::log(fudgefac / tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)));
  }
  return ns;
}

void set_kernel_shape_given_ns(finufft_spread_opts &spopts, int debug) {
  // Writes kernel shape parameter(s) (beta,...), into spopts, given previously-set
  // kernel info fields in spopts, principally: nspread, upsampfac, kerformula.
  // debug >0 causes stdout reporting.
  int ns       = spopts.nspread;
  double sigma = spopts.upsampfac;

  // these strings must match: kernel_definition, and the below
  const char *kernames[] = {"default", "ES (legacy params)", "KB"};

  if (spopts.kerformula == 1) {
    // Exponential of Semicircle (ES)
    double betaoverns = 2.30; // the legacy logic, used 2017-2025.
    if (ns == 2)
      betaoverns = 2.20;
    else if (ns == 3)
      betaoverns = 2.26;
    else if (ns == 4)
      betaoverns = 2.38;

    if (sigma != 2.0) { // low-sigma choice, introduced v1.0 (2018-2025)
      const double gamma = 0.97;
      betaoverns         = gamma * common::PI * (1.0 - 1.0 / (2.0 * sigma));
    }
    spopts.beta = betaoverns * (double)ns;

  } else if (spopts.kerformula == 2) {
    // Kaiser-Bessel (KB), with shape param formula from Beatty et al,
    // IEEE Trans Med Imaging, 2005 24(6):799-808. doi:10.1109/TMI.2005.848376
    // "Rapid gridding reconstruction with a minimal oversampling ratio".
    double t    = (double)ns * (1.0 - 1.0 / (2.0 * sigma));
    spopts.beta = common::PI * std::sqrt(t * t - 0.8); // just below PI*t
  }

  if (debug || spopts.debug)
    printf("[setup_spreadinterp]\tkerformula=%d: %s...\n", spopts.kerformula,
           kernames[spopts.kerformula]);
}

} // namespace finufft::kernel

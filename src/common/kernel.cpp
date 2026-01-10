#include <algorithm>
#include <cmath>
#include <cstdio>
#include <finufft_common/common.h>
#include <finufft_common/kernel.h>
#include <finufft_spread_opts.h>

// this module uses finufft_spread_opts but does not know about FINUFFT_PLAN class
// nor finufft_opts. This allows it to be used by CPU & GPU.

namespace finufft::kernel {

int theoretical_kernel_ns(double tol, int dim, int type, int debug,
                      const finufft_spread_opts &spopts) {
  // returns ideal preferred spread width (ns, a.k.a. w) using convergence rate,
  // in exact arithmetic, to achieve requested tolerance tol. Possibly uses
  // other parameters in spopts (upsampfac, kerformula,...). No clipping done.
  int ns;
  double sigma = spopts.upsampfac;

  if (spopts.kerformula==1 & sigma == 2.0) {   // legacy (2017-2025)
    ns = (int)std::ceil(-std::log10(tol / 10.0));
  
  } else {  // generic formula for PSWF-like kernels
    double fudgefac = 1.0;   // *** todo: tweak it, per kerformula
    ns = (int)std::ceil( std::log(fudgefac / tol) /
                         (finufft::common::PI * std::sqrt(1.0 - 1.0 / sigma)));
  }
  return ns;
}

void set_kernel_shape_given_ns(finufft_spread_opts &spopts) {
// Writes kernel shape parameter(s) (beta,...), into spopts, given previously-set
// kernel info fields in spopts, principally: nspread, upsampfac, kerformula.
  int ns = spopts.nspread;
  double sigma = spopts.upsampfac;
  spopts.ES_halfwidth = (double)ns / 2.0;
  opts.ES_c = 4.0 / (double)(ns * ns); // *** kill c param

  // these strings must match: kernel_definition, and the below
  const char* kernames[] = {"default", "ES (legacy params)", "KB"};

  if (spopts.kerformula == 1) {
    // Exponential of Semicircle (ES)
    double betaoverns = 2.30;    // legacy logic 2017-2025.
    if (ns == 2)
      betaoverns = 2.20;
    else if (ns == 3)
      betaoverns = 2.26;
    else if (ns == 4)
      betaoverns = 2.38;

    if (sigma != 2.0) {
      double gamma = 0.97;
      betaoverns   = gamma * common::PI * (1.0 - 1.0 / (2.0 * sigma));
    }
    spopts.ES_beta = betaoverns * (double)ns;

  } else if (spopts.kerformula == 2) {
    // Kaiser-Bessel (KB)
    // shape param formula from Beatty et al. 2005.
    double tmp   = (double)ns * (double)ns / (upsampfac * upsampfac);
    double term2 = (upsampfac - 0.5) * (upsampfac - 0.5);
    spopts.ES_beta = common::PI * std::sqrt(tmp * term2 - 0.8);
  }

  if (spopts.debug)
    printf("setup_spreadinterp:\tkerformula=%d: %s...\n", spopts.kerformula,
      kernames[spopts.kerformula]);
  }

} // namespace finufft::kernel

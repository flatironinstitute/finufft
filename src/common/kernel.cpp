#include <algorithm>
#include <cmath>
#include <cstdio>
#include <finufft_common/common.h>
#include <finufft_common/kernel.h>
#include <finufft_common/spread_opts.h>

namespace finufft::kernel {

int compute_kernel_ns(double upsampfac, double tol, int kerformula,
                      const finufft_spread_opts &opts) {
  // Note: opts is unused here but kept for API consistency if needed later
  int ns;
  if (upsampfac == 2.0) {
    ns = (int)std::ceil(-std::log10(tol / 10.0));
  } else {
    ns = (int)std::ceil(
        -std::log(tol) / (finufft::common::PI * std::sqrt(1.0 - 1.0 / upsampfac)));
  }
  ns = std::max(2, ns);
  return ns;
}

void initialize_kernel_params(finufft_spread_opts &opts, double upsampfac, double tol,
                              int kerformula) {
  int ns;
  // Respect any pre-set opts.nspread (e.g., clipped by caller). If it's <=0 compute it.
  if (opts.nspread > 0) {
    ns = opts.nspread;
  } else {
    ns = compute_kernel_ns(upsampfac, tol, kerformula, opts);
  }

  opts.kerformula   = kerformula;
  opts.nspread      = ns; // ensure opts is populated with the (possibly clipped) ns
  opts.ES_halfwidth = (double)ns / 2.0;
  opts.ES_c = 4.0 / (double)(ns * ns); // *** move this into kernel.h, kill c param

  if (kerformula == 0) { // always the default
    // Exponential of Semicircle (ES)
    double betaoverns = 2.30;
    if (ns == 2)
      betaoverns = 2.20;
    else if (ns == 3)
      betaoverns = 2.26;
    else if (ns == 4)
      betaoverns = 2.38;

    if (upsampfac != 2.0) {
      double gamma = 0.97;
      betaoverns   = gamma * common::PI * (1.0 - 1.0 / (2.0 * upsampfac));
    }
    opts.ES_beta = betaoverns * (double)ns;

  } else if (kerformula == 1) {
    // Kaiser-Bessel (KB)
    // Formula from Beatty et al. 2005.
    double tmp   = (double)ns * (double)ns / (upsampfac * upsampfac);
    double term2 = (upsampfac - 0.5) * (upsampfac - 0.5);
    opts.ES_beta = common::PI * std::sqrt(tmp * term2 - 0.8);
  }

  if (opts.debug) {
    const char *kname = (kerformula == 1) ? "KB" : "ES";
    printf("setup_spreader: using spread kernel type %d (%s)\n", kerformula, kname);
    printf("setup_spreader eps=%.3g sigma=%.3g (%s): chose ns=%d beta=%.3g\n", tol,
           upsampfac, kname, ns, opts.ES_beta);
  }
}

// AHB prefer cut; only used in accuracy_test which will be changed ***
double sigma_max_tol(double upsampfac, int kerformula, int max_ns) {
  if (upsampfac == 2.0) {
    return 10.0 * std::pow(10.0, -(double)max_ns);
  } else {
    return std::exp(
        -(double)max_ns * finufft::common::PI * std::sqrt(1.0 - 1.0 / upsampfac));
  }
}

} // namespace finufft::kernel

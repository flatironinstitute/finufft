/* Checks the analytic PSWF self-FT used by type-3 deconv (kernel.h:
   pswf_selfft_params) against brute-force integration of the same piecewise
   polynomial, over the reachable (kerformula, upsampfac, nspread) space.

   phihat(xi) = prefac * phi(grid_scale*xi), phi the Horner approximant of the
   kernel in grid units. Three things can break, and each gets its own check:

   1) prefac == int phi dx: pure arithmetic on the panel coeffs (parity, stride,
      degree indexing), so it must hold to machine precision.
   2) the identity at xi != 0: exact for the true PSWF, so the error here is the
      approximant's, not the formula's. Compared against the kernel's own error
      scale tolfac*exp(-(ns-1)*pi*sqrt(1-1/sigma)).
   3) support: the identity needs |xi| <= 2*beta/nspread, while deconv evaluates
      up to |xi| = pi/upsampfac. Outside the support phi is 0, which would make
      deconv infinite, so the margin must stay positive.

   Not in CI; run by hand after touching the kernel shape formulas or the Horner
   coefficient layout, from the repo root:

   g++ -O2 -std=c++17 -Iinclude devel/check_pswf_selfft.cpp src/common/kernel.cpp \
       src/common/pswf.cpp src/common/utils.cpp -o /tmp/check_pswf_selfft

   Currently PASSES with min margin 0.473, max prefac_err 3.3e-15, and ft_err at
   worst 0.37 of the kernel error scale. Barbone.
*/
#include <cmath>
#include <cstdio>
#include <finufft_common/kernel.h>
#include <finufft_common/utils.h>
#include <vector>

using namespace finufft::kernel;
using finufft::common::gaussquad;
using finufft::common::PI;

// Panel coeffs exactly as precompute_horner_coeffs lays them out, but with
// stride = ns (no SIMD padding, which pswf_selfft_params takes as an argument).
static std::vector<double> fit_panels(const finufft_spread_opts &so, int nc) {
  const int ns = so.nspread;
  auto ker = kernel_definition_lambda(so);
  std::vector<double> c(size_t(nc) * ns);
  for (int j = 0; j < ns; ++j) {
    const double shift = 2 * j + 1 - ns;
    auto panel = poly_fit<double>([&](double x) { return ker((x + shift) / ns); }, nc);
    for (int k = 0; k < nc; ++k) c[size_t(k) * ns + j] = panel[k];
  }
  return c;
}

// phi(x), x in grid units, from the panel coeffs (mirrors evaluate_kernel_runtime)
static double eval_phi(const std::vector<double> &c, int ns, int nc, double x) {
  const double ns2 = ns / 2.0;
  for (int i = 0; i < ns; ++i)
    if (x > -ns2 + i && x <= -ns2 + i + 1) {
      const double z = 2 * (x - i) + (ns - 1);
      double r = 0;
      for (int k = 0; k < nc; ++k) r = r * z + c[size_t(k) * ns + i];
      return r;
    }
  return 0;
}

// int_{-ns/2}^{ns/2} phi(x) cos(xi.x) dx by per-panel Gauss (exact at xi=0; the
// rule must be per-panel, a global one would straddle the panel kinks)
static double phihat_quad(const std::vector<double> &c, int ns, int nc, double xi) {
  constexpr int NQ = 32;
  double z[NQ], w[NQ];
  gaussquad(NQ, z, w);
  double sum = 0;
  for (int i = 0; i < ns; ++i) {
    const double xc = -ns / 2.0 + i + 0.5;
    for (int n = 0; n < NQ; ++n) {
      const double x = xc + 0.5 * z[n];
      sum += 0.5 * w[n] * eval_phi(c, ns, nc, x) * std::cos(xi * x);
    }
  }
  return sum;
}

int main() {
  int fails = 0;
  printf("%3s %3s %5s %8s %10s %10s %10s %10s\n", "kf", "ns", "sigma", "margin",
         "prefac_err", "ft_err", "ker_err", "ratio");
  for (int kf : {7, 8, 9})
    for (double sigma : {1.1, 1.25, 1.5, 2.0, 3.0})
      for (int ns = 2; ns <= 16; ++ns) {
        finufft_spread_opts so{};
        so.nspread = ns, so.upsampfac = sigma, so.kerformula = kf;
        set_kernel_shape_given_ns(so, 0);
        const int nc = max_nc_given_ns(ns);
        const auto c = fit_panels(so, nc);
        const auto [grid_scale, prefac] =
            pswf_selfft_params(ns, so.beta, c.data(), nc, ns);

        // (1) prefac == phihat(0)
        const double p_err = std::abs(prefac - phihat_quad(c, ns, nc, 0)) / prefac;
        // (3) support margin: phi(grid_scale*xi) is 0 beyond grid_scale*xi = ns/2,
        // ie beyond |xi| = 2*beta/nspread; deconv evaluates up to |xi| = pi/sigma
        const double ximax = PI / sigma, margin = (ns / 2.0) / grid_scale - ximax;
        // (2) identity away from 0, relative to phihat(0) (deconv divides by phihat,
        // so the absolute error at large xi is what pollutes the result)
        double ft_err = 0;
        for (int m = 1; m <= 32; ++m) {
          const double xi = ximax * m / 32.0;
          ft_err =
              std::max(ft_err, std::abs(prefac * eval_phi(c, ns, nc, xi * grid_scale) -
                                        phihat_quad(c, ns, nc, xi)) /
                                   prefac);
        }
        // kernel error scale, floored at double roundoff (ft_err cannot beat it)
        const double ker_err = std::max(
            kernel_tolfac(1, 3) * std::exp(-(ns - 1) * PI * std::sqrt(1 - 1 / sigma)),
            1e-14);
        printf("%3d %3d %5.2f %8.3f %10.2e %10.2e %10.2e %10.2f%s\n", kf, ns, sigma,
               margin, p_err, ft_err, ker_err, ft_err / ker_err,
               (p_err > 1e-13 || margin <= 0 || ft_err > 10 * ker_err) ? "  FAIL" : "");
        fails += (p_err > 1e-13) + (margin <= 0) + (ft_err > 10 * ker_err);
      }
  printf("%s (%d failure(s))\n", fails ? "FAILED" : "PASSED", fails);
  return fails != 0;
}

/* Routines for the evaluation of the prolate spheroidal wavefunction
   of order zero (Psi_0^c) inside [-1,1], for arbitrary frequency parameter c.
   They use a basis of Legendre polynomials.
   This is a collection of Fortran codes by Vladimir Rokhlin, specifically
   legeexps.f and prolcrea.f
   The originals may be found in src/common/specialfunctions/ of the DMK repo
   https://github.com/flatironinstitute/dmk
   They have been converted to C and repackaged by Libin Lu.
*/

#include <finufft_common/pswf.h>
#include <finufft_common/safe_call.h>

namespace finufft::common {

namespace { // file-local helpers

static void prolcoef(double rlam, int k, double c, double &alpha, double &beta,
                     double &gamma) {
  double kf     = k;
  double alpha0 = kf * (kf - 1.) / ((2. * kf + 1.) * (2. * kf - 1.));
  double beta0  = ((kf + 1.) * (kf + 1.) / (2. * kf + 3.) + kf * kf / (2. * kf - 1.)) /
                 (2. * kf + 1.);
  double gamma0 = (kf + 1.) * (kf + 2.) / ((2. * kf + 1.) * (2. * kf + 3.));

  alpha = -c * c * alpha0;
  beta  = rlam - kf * (kf + 1.) - c * c * beta0;
  gamma = -c * c * gamma0;
}

// fills as, bs, cs in [0; (n+2)/2]
static void prolmatr(std::vector<double> &as, std::vector<double> &bs,
                     std::vector<double> &cs, int n, double c, double rlam) {
  for (int k = 0; 2 * k <= n + 2; ++k) {
    prolcoef(rlam, 2 * k, c, as[k], bs[k], cs[k]);

    if (k != 0) as[k] *= std::sqrt((2 * k + .5) / (2 * k - 1.5));
    cs[k] *= std::sqrt((2 * k + .5) / (2 * k + 2.5));
  }
}

static void prolql1(int n, std::vector<double> &d, std::vector<double> &e) {
  if (n == 1) return;

  for (int i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0.0;

  for (int l = 0; l < n; ++l) {
    int j = 0;
    while (true) {
      int m;
      for (m = l; m < n - 1; ++m) {
        double tst1 = std::abs(d[m]) + std::abs(d[m + 1]);
        double tst2 = tst1 + std::abs(e[m]);
        if (tst2 == tst1) break;
      }

      if (m == l) break;
      if (j == 30) throw finufft::exception(FINUFFT_ERR_PSWF_SETUP);
      ++j;

      double g = (d[l + 1] - d[l]) / (2. * e[l]);
      double r = std::sqrt(g * g + 1.0);
      g        = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
      double s = 1.0;
      double c = 1.0;
      double p = 0.0;

      for (int i = m - 1; i >= l; --i) {
        double f = s * e[i];
        double b = c * e[i];
        r        = std::sqrt(f * f + g * g);
        e[i + 1] = r;
        if (r == 0.0) {
          d[i + 1] -= p;
          e[m] = 0.0;
          break;
        }
        s        = f / r;
        c        = g / r;
        g        = d[i + 1] - p;
        r        = (d[i] - g) * s + 2. * c * b;
        p        = s * r;
        d[i + 1] = g + p;
        g        = c * r - b;
      }

      if (r == 0.) break;
      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
    }

    for (int i = l; (i > 0) && (d[i] < d[i - 1]); --i) std::swap(d[i], d[i - 1]);
  }
}

static void prolfact(std::vector<double> &a, const std::vector<double> &b,
                     const std::vector<double> &c, int n, std::vector<double> &u,
                     std::vector<double> &v, std::vector<double> &w) {
  // Eliminate down and up, and scale
  for (int i = 0; i + 1 < n; ++i) {
    double d = c[i + 1] / a[i];
    a[i + 1] -= b[i] * d;
    u[i]     = d;
    v[i + 1] = b[i] / a[i + 1];
    w[i + 1] = 1. / a[i + 1];
  }
  w[0] = 1. / a[0];
}

static void prolsolv(const std::vector<double> &u, const std::vector<double> &v,
                     const std::vector<double> &w, int n, std::vector<double> &rhs) {
  // Eliminate down
  for (int i = 0; i + 1 < n; ++i) rhs[i + 1] -= u[i] * rhs[i];

  // Eliminate up and scale
  for (int i = n - 1; i > 0; --i) {
    rhs[i - 1] -= rhs[i] * v[i];
    rhs[i] *= w[i];
  }
  rhs[0] *= w[0];
}

static void prolfun0(int n, double c, std::vector<double> &xk, double eps) {
  double delta = 1.0e-8;

  xk.resize(n / 2 + 3);
  std::vector<double> as(n / 2 + 2), bs(n / 2 + 2), cs(n / 2 + 2), u(n / 2 + 2),
      v(n / 2 + 2), w(n / 2 + 2);
  prolmatr(as, bs, cs, n, c, 0.);

  prolql1(n / 2, bs, as);

  std::fill(xk.begin(), xk.end(), 1.);

  double rlam = -bs[n / 2 - 1] + delta;
  prolmatr(as, bs, cs, n, c, rlam);

  prolfact(bs, cs, as, n / 2, u, v, w);

  constexpr int numit = 4;
  for (int ijk = 0; ijk < numit; ++ijk) {
    prolsolv(u, v, w, n / 2, xk);

    double d = 0;
    for (int j = 0; j < n / 2; ++j) d += xk[j] * xk[j];

    d = std::sqrt(d);
    for (int j = 0; j < n / 2; ++j) {
      xk[j] /= d;
      as[j] = xk[j];
    }
  }

  int imax = 0;
  for (int i = 0; i < n / 2; ++i) {
    if (std::abs(xk[i]) > eps) imax = i;
    xk[i] *= std::sqrt(i * 2 + .5);
  }
  xk.resize(imax + 1);
}

static void prolps0i(double c, std::vector<double> &work) {
  static constexpr std::array<int, 20> ns = {48,  64,  80,  92,  106, 120, 130,
                                             144, 156, 168, 178, 190, 202, 214,
                                             224, 236, 248, 258, 268, 280};

  int i = static_cast<int>(c / 10);
  int n = (i < int(ns.size())) ? ns[i] : static_cast<int>(c * 3) / 2;

  prolfun0(n, c, work, 1e-16);
}

} // anonymous namespace

PSWF0::PSWF0(double c_) : c(c_) {
  prolps0i(c, workdata);
  coef.resize(workdata.size());
  for (size_t i = 1; i < coef.size(); ++i) {
    double l   = 2 * i - 1.;
    coef[i][0] = ((2. * l - 1.) * (2. * l + 1.)) / (l * (l + 1.));
    coef[i][1] = ((2. * l + 1.) * (l - 1.) * (l - 1.) + l * l * (2. * l - 3)) /
                 (l * (l + 1.) * (2. * l - 3.));
    coef[i][2] = ((2. * l + 1.) * (l - 1.) * (l - 2.)) / (l * (l + 1.) * (2. * l - 3.));
  }
  xv0 = 1. / eval_raw(0.);
}

} // namespace finufft::common

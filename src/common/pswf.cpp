#include <array>
#include <cmath>
#include <finufft_common/pswf.h>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace finufft::common {

// start of legendre functions
static inline void legepol(double x, int n, double &pol, double &der) {
  double pkm1 = 1.0;
  double pk   = x;
  double pkp1;

  if (n == 0) {
    pol = 1.0;
    der = 0.0;
    return;
  }

  if (n == 1) {
    pol = x;
    der = 1.0;
    return;
  }

  pk   = 1.0;
  pkp1 = x;

  for (int k = 1; k < n; ++k) {
    pkm1 = pk;
    pk   = pkp1;
    pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
  }

  pol = pkp1;
  der = n * (x * pkp1 - pk) / (x * x - 1);
}

static inline void legetayl(double pol, double der, double x, double h, int n, int k,
                            double &sum, double &sumder) {
  double done = 1.0;
  double q0   = pol;
  double q1   = der * h;
  double q2   = (2 * x * der - n * (n + done) * pol) / (1 - x * x);
  q2          = q2 * h * h / 2;

  sum    = q0 + q1 + q2;
  sumder = q1 / h + q2 * 2 / h;

  if (k <= 2) return;

  double qi   = q1;
  double qip1 = q2;

  for (int i = 1; i <= k - 2; ++i) {
    double d = 2 * x * (i + 1) * (i + 1) / h * qip1 - (n * (n + done) - i * (i + 1)) * qi;
    d        = d / (i + 1) / (i + 2) * h * h / (1 - x * x);
    double qip2 = d;

    sum += qip2;
    sumder += d * (i + 2) / h;

    qi   = qip1;
    qip1 = qip2;
  }
}

/* Constructs Gaussian quadrature of order n.
   itype=1 => both roots (ts) and weights (whts) are computed.
   itype=0 => only roots (ts) are computed. */
static inline void legerts(int itype, int n, double *ts, double *whts) {
  int k     = 30;
  double d  = 1.0;
  double d2 = d + 1.0e-24;
  if (d2 != d) {
    k = 54;
  }

  int half      = n / 2;
  int ifodd     = n - 2 * half;
  double pi_val = atan(1.0) * 4.0;
  double h      = pi_val / (2.0 * n);

  /* Initial approximations (for i >= n/2+1) */
  int ii = 0;
  for (int i = 1; i <= n; i++) {
    if (i < (n / 2 + 1)) {
      continue;
    }
    ii++;
    double t   = (2.0 * i - 1.0) * h;
    ts[ii - 1] = -cos(t);
  }

  /* Start from center: find roots one by one via Newton updates */
  double pol = 1.0, der = 0.0;
  double x0 = 0.0;
  legepol(x0, n, pol, der);
  double x1 = ts[0];

  int n2      = (n + 1) / 2;
  double pol3 = pol, der3 = der;

  for (int kk = 1; kk <= n2; kk++) {
    if ((ifodd == 1) && (kk == 1)) {
      ts[kk - 1] = x0;
      if (itype > 0) {
        whts[kk - 1] = der;
      }
      x0   = x1;
      x1   = ts[kk];
      pol3 = pol;
      der3 = der;
      continue;
    }

    /* Newton iteration */
    int ifstop = 0;
    for (int i = 1; i <= 10; i++) {
      double hh = x1 - x0;

      legetayl(pol3, der3, x0, hh, n, k, pol, der);
      x1 = x1 - pol / der;

      if (fabs(pol) < 1.0e-12) {
        ifstop++;
      }
      if (ifstop == 3) {
        break;
      }
    }

    ts[kk - 1] = x1;
    if (itype > 0) {
      whts[kk - 1] = der;
    }

    x0   = x1;
    x1   = ts[kk];
    pol3 = pol;
    der3 = der;
  }

  /* Mirror roots around 0: fill second half of ts[] */
  for (int i = n2; i >= 1; i--) {
    ts[i - 1 + half] = ts[i - 1];
  }
  for (int i = 1; i <= half; i++) {
    ts[i - 1] = -ts[n - i];
  }
  if (itype <= 0) {
    return;
  }

  /* Mirror weights similarly */
  for (int i = n2; i >= 1; i--) {
    whts[i - 1 + half] = whts[i - 1];
  }
  for (int i = 1; i <= half; i++) {
    whts[i - 1] = whts[n - i];
  }

  /* Compute final weights = 2 / (1 - ts[i]^2) / (der[i]^2) */
  for (int i = 0; i < n; i++) {
    double tmp = 1.0 - ts[i] * ts[i];
    whts[i]    = 2.0 / tmp / (whts[i] * whts[i]);
  }
}

static inline void legepols(double x, int n, double *pols) {
  double pkm1 = 1.0;
  double pk   = x;

  if (n == 0) {
    pols[0] = 1.0;
    return;
  }

  if (n == 1) {
    pols[0] = 1.0;
    pols[1] = x;
    return;
  }

  pols[0] = 1.0;
  pols[1] = x;

  for (int k = 1; k < n; ++k) {
    double pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
    pols[k + 1] = pkp1;
    pkm1        = pk;
    pk          = pkp1;
  }
}

// TODO: legepols() is not tested yet.
// only itype !=2 is tested.
static inline void legeexps(int itype, int n, double *x,
                            std::vector<std::vector<double>> &u,
                            std::vector<std::vector<double>> &v, double *whts) {
  int itype_rts = (itype > 0) ? 1 : 0;

  // Call legerts to construct the nodes and weights of the n-point Gaussian quadrature
  legerts(itype_rts, n, x, whts);

  // legerts() is buggy now, type 2(code below) is not tested yet.
  // legerts_(&itype_rts, &n, x, whts);

  // If itype is not 2, return early
  if (itype != 2) return;

  // Construct the matrix of values of the Legendre polynomials at these nodes
  for (int i = 0; i < n; ++i) {
    std::vector<double> pols(n);
    legepols(x[i], n - 1, pols.data());
    for (int j = 0; j < n; ++j) {
      u[j][i] = pols[j];
    }
  }

  // Transpose u to get v
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      v[i][j] = u[j][i];
    }
  }

  // Construct the inverse u, converting the values of a function at Gaussian nodes into
  // the coefficients of a Legendre expansion of that function
  for (int i = 0; i < n; ++i) {
    double d = 1.0 * (2 * (i + 1) - 1) / 2;
    for (int j = 0; j < n; ++j) {
      u[i][j] = v[j][i] * whts[j] * d;
    }
  }
}

static inline void legeexev(double x, double &val, const double *pexp, int n) {
  double pjm2 = 1.0;
  double pjm1 = x;

  val = pexp[0] * pjm2 + pexp[1] * pjm1;

  for (int j = 2; j <= n; ++j) {
    double pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
    val += pexp[j] * pj;
    pjm2 = pjm1;
    pjm1 = pj;
  }
}

static inline void legeFDER(double x, double &val, double &der, const double *pexp,
                            int n) {
  double pjm2   = 1.0;
  double pjm1   = x;
  double derjm2 = 0.0;
  double derjm1 = 1.0;

  val = pexp[0] * pjm2 + pexp[1] * pjm1;
  der = pexp[1];

  for (int j = 2; j <= n; ++j) {
    double pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
    val += pexp[j] * pj;

    double derj = (2 * j - 1) * (pjm1 + x * derjm1) - (j - 1) * derjm2;
    derj /= j;
    der += pexp[j] * derj;

    pjm2   = pjm1;
    pjm1   = pj;
    derjm2 = derjm1;
    derjm1 = derj;
  }
}

// Function to compute the Legendre polynomial p_n(x) and its derivative p'_n(x)
static inline void legendre(int n, double x, double &pn, double &pn_prime) {
  if (n == 0) {
    pn       = 1.0;
    pn_prime = 0.0;
    return;
  }

  if (n == 1) {
    pn       = x;
    pn_prime = 1.0;
    return;
  }

  double pn_minus1 = 1.0; // P_0(x)
  double pn_minus2 = 0.0; // P_-1(x)
  pn               = x;   // P_1(x)

  for (int k = 2; k <= n; ++k) {
    pn_minus2 = pn_minus1;
    pn_minus1 = pn;
    pn        = ((2.0 * k - 1.0) * x * pn_minus1 - (k - 1.0) * pn_minus2) / k;
  }

  pn_prime = n * (x * pn - pn_minus1) / (x * x - 1.0);
}
// end of legendre functions

// start of prolate functions
void prolc180_der3(double eps, double &der3) {

  static const std::array<double, 180> der3s = {
      0.0,
      3.703231117106314e-02,
      2.662796075011415e-01,
      5.941926542747007e-01,
      9.578230754832373e-01,
      1.363569812949696e+00,
      1.807582627232425e+00,
      2.287040787443315e+00,
      2.800489841625932e+00,
      3.346520825211228e+00,
      3.925129833524817e+00,
      4.535419879018613e+00,
      5.177538088593701e+00,
      5.851057619952580e+00,
      6.555804054828174e+00,
      7.292020430656394e+00,
      8.059158990368712e+00,
      8.857123665436557e+00,
      9.686327541088655e+00,
      1.054640955930842e+01,
      1.143707141760266e+01,
      1.235843748588681e+01,
      1.331074380051023e+01,
      1.429316645410095e+01,
      1.530644212280254e+01,
      1.634980229052360e+01,
      1.742371527350574e+01,
      1.852786959134436e+01,
      1.966241028647265e+01,
      2.082706565691709e+01,
      2.202156399063638e+01,
      2.324513764895337e+01,
      2.450154874132168e+01,
      2.578557879197864e+01,
      2.710219152702352e+01,
      2.844590643457592e+01,
      2.982194872189250e+01,
      3.122457703819981e+01,
      3.265927724478793e+01,
      3.412004831578323e+01,
      3.561263588011575e+01,
      3.713704155571862e+01,
      3.869326681306944e+01,
      4.027479309954505e+01,
      4.188123512956217e+01,
      4.352576155609020e+01,
      4.519494867085516e+01,
      4.689544558088133e+01,
      4.862725312364265e+01,
      5.038308226775531e+01,
      5.216996738943379e+01,
      5.398790913473069e+01,
      5.583690809846664e+01,
      5.771696483563476e+01,
      5.962015154378219e+01,
      6.155414190868980e+01,
      6.351893636867634e+01,
      6.551453532837742e+01,
      6.753250221415664e+01,
      6.958958399152446e+01,
      7.166878031319368e+01,
      7.377852843589304e+01,
      7.591882864408177e+01,
      7.808968120601168e+01,
      8.028188878922057e+01,
      8.251372002218120e+01,
      8.476665370485364e+01,
      8.704988763041084e+01,
      8.936342198942501e+01,
      9.170725696750367e+01,
      9.408139273542218e+01,
      9.647574810864796e+01,
      9.891035968318585e+01,
      1.013649391732796e+02,
      1.038495680706917e+02,
      1.063748312950304e+02,
      1.089196851143706e+02,
      1.114945887021435e+02,
      1.140885801820887e+02,
      1.167234579055465e+02,
      1.193883857149036e+02,
      1.220720250413416e+02,
      1.247969275635160e+02,
      1.275402908406401e+02,
      1.303251684172720e+02,
      1.331282559724378e+02,
      1.359611431651965e+02,
      1.388238300708471e+02,
      1.417163167603507e+02,
      1.446386032954563e+02,
      1.475782236729200e+02,
      1.505599848644160e+02,
      1.535715460806204e+02,
      1.566000662021011e+02,
      1.596711024010027e+02,
      1.627588477107742e+02,
      1.658893590967149e+02,
      1.690363297860595e+02,
      1.722128508870634e+02,
      1.754189224418921e+02,
      1.786545444888072e+02,
      1.819197170667437e+02,
      1.852144402114015e+02,
      1.885387139581953e+02,
      1.918925383387225e+02,
      1.952759133924808e+02,
      1.986743765316800e+02,
      2.021167283277167e+02,
      2.055886308747634e+02,
      2.090752480704547e+02,
      2.126061275354775e+02,
      2.161514728445827e+02,
      2.197413293189910e+02,
      2.233454028450144e+02,
      2.269787784335390e+02,
      2.306414561065548e+02,
      2.343491422119853e+02,
      2.380705482795840e+02,
      2.418212564954385e+02,
      2.456012668766033e+02,
      2.494105794440804e+02,
      2.532491942146669e+02,
      2.571171112093926e+02,
      2.610143304358681e+02,
      2.649241522962268e+02,
      2.688798518867251e+02,
      2.728648537691562e+02,
      2.768791579444873e+02,
      2.809055686954829e+02,
      2.849783533576689e+02,
      2.890804403554123e+02,
      2.931942620075851e+02,
      2.973548295757224e+02,
      3.015268840118008e+02,
      3.057279930461074e+02,
      3.099762194632530e+02,
      3.142355613580597e+02,
      3.185422684529161e+02,
      3.228598432462443e+02,
      3.272064726902580e+02,
      3.315821567930542e+02,
      3.360057012097200e+02,
      3.404396183022963e+02,
      3.449025900686188e+02,
      3.493946165380561e+02,
      3.539156977033438e+02,
      3.584658335865259e+02,
      3.630450242038053e+02,
      3.676532695407853e+02,
      3.722905695962574e+02,
      3.769569244097179e+02,
      3.816523339874616e+02,
      3.863767983079612e+02,
      3.911100281057601e+02,
      3.958924783306446e+02,
      4.007039833307977e+02,
      4.055445431251419e+02,
      4.103933743959938e+02,
      4.152919201395830e+02,
      4.202195206860208e+02,
      4.251550223528558e+02,
      4.301406088745175e+02,
      4.351552502298094e+02,
      4.401774223148202e+02,
      4.452500497108158e+02,
      4.503299610661473e+02,
      4.554605744971888e+02,
      4.605982251667677e+02,
      4.657646839606519e+02,
      4.709822146642819e+02,
      4.762064127721131e+02,
      4.814819295722471e+02,
      4.867638670342317e+02,
      4.920746126195583e+02,
      4.974370467592532e+02,
      5.028055317088541e+02,
      5.082028248088587e+02,
      5.136289260479356e+02,
      5.191072088380275e+02,
      5.245910494770286e+02,
  };

  if (eps < 1.0e-18) eps = 1e-18;
  double d = -log10(eps);
  int i    = static_cast<int>(d * 10 + 0.1);
  der3     = der3s[i - 1];
}

void prolc180(double eps, double &c) {
  static const std::array<double, 180> cs = {
      0.43368E-16, 0.10048E+01, 0.17298E+01, 0.22271E+01, 0.26382E+01, 0.30035E+01,
      0.33409E+01, 0.36598E+01, 0.39658E+01, 0.42621E+01, 0.45513E+01, 0.48347E+01,
      0.51136E+01, 0.53887E+01, 0.56606E+01, 0.59299E+01, 0.61968E+01, 0.64616E+01,
      0.67247E+01, 0.69862E+01, 0.72462E+01, 0.75049E+01, 0.77625E+01, 0.80189E+01,
      0.82744E+01, 0.85289E+01, 0.87826E+01, 0.90355E+01, 0.92877E+01, 0.95392E+01,
      0.97900E+01, 0.10040E+02, 0.10290E+02, 0.10539E+02, 0.10788E+02, 0.11036E+02,
      0.11284E+02, 0.11531E+02, 0.11778E+02, 0.12024E+02, 0.12270E+02, 0.12516E+02,
      0.12762E+02, 0.13007E+02, 0.13251E+02, 0.13496E+02, 0.13740E+02, 0.13984E+02,
      0.14228E+02, 0.14471E+02, 0.14714E+02, 0.14957E+02, 0.15200E+02, 0.15443E+02,
      0.15685E+02, 0.15927E+02, 0.16169E+02, 0.16411E+02, 0.16652E+02, 0.16894E+02,
      0.17135E+02, 0.17376E+02, 0.17617E+02, 0.17858E+02, 0.18098E+02, 0.18339E+02,
      0.18579E+02, 0.18819E+02, 0.19059E+02, 0.19299E+02, 0.19539E+02, 0.19778E+02,
      0.20018E+02, 0.20257E+02, 0.20496E+02, 0.20736E+02, 0.20975E+02, 0.21214E+02,
      0.21452E+02, 0.21691E+02, 0.21930E+02, 0.22168E+02, 0.22407E+02, 0.22645E+02,
      0.22884E+02, 0.23122E+02, 0.23360E+02, 0.23598E+02, 0.23836E+02, 0.24074E+02,
      0.24311E+02, 0.24549E+02, 0.24787E+02, 0.25024E+02, 0.25262E+02, 0.25499E+02,
      0.25737E+02, 0.25974E+02, 0.26211E+02, 0.26448E+02, 0.26685E+02, 0.26922E+02,
      0.27159E+02, 0.27396E+02, 0.27633E+02, 0.27870E+02, 0.28106E+02, 0.28343E+02,
      0.28580E+02, 0.28816E+02, 0.29053E+02, 0.29289E+02, 0.29526E+02, 0.29762E+02,
      0.29998E+02, 0.30234E+02, 0.30471E+02, 0.30707E+02, 0.30943E+02, 0.31179E+02,
      0.31415E+02, 0.31651E+02, 0.31887E+02, 0.32123E+02, 0.32358E+02, 0.32594E+02,
      0.32830E+02, 0.33066E+02, 0.33301E+02, 0.33537E+02, 0.33773E+02, 0.34008E+02,
      0.34244E+02, 0.34479E+02, 0.34714E+02, 0.34950E+02, 0.35185E+02, 0.35421E+02,
      0.35656E+02, 0.35891E+02, 0.36126E+02, 0.36362E+02, 0.36597E+02, 0.36832E+02,
      0.37067E+02, 0.37302E+02, 0.37537E+02, 0.37772E+02, 0.38007E+02, 0.38242E+02,
      0.38477E+02, 0.38712E+02, 0.38947E+02, 0.39181E+02, 0.39416E+02, 0.39651E+02,
      0.39886E+02, 0.40120E+02, 0.40355E+02, 0.40590E+02, 0.40824E+02, 0.41059E+02,
      0.41294E+02, 0.41528E+02, 0.41763E+02, 0.41997E+02, 0.42232E+02, 0.42466E+02,
      0.42700E+02, 0.42935E+02, 0.43169E+02, 0.43404E+02, 0.43638E+02, 0.43872E+02,
      0.44107E+02, 0.44341E+02, 0.44575E+02, 0.44809E+02, 0.45044E+02, 0.45278E+02};

  if (eps < 1.0e-18) eps = 1e-18;
  double d = -log10(eps);
  int i    = static_cast<int>(d * 10 + 0.1);
  c        = cs[i - 1];
}

static inline void prosinin(double c, const double *ts, const double *whts,
                            const double *fs, double x, int n, double &rint,
                            double &derrint) {
  rint    = 0.0;
  derrint = 0.0;

  for (int i = 0; i < n; ++i) {
    double diff     = x - ts[i];
    double sin_term = sin(c * diff);
    double cos_term = cos(c * diff);

    rint += whts[i] * fs[i] * sin_term / diff;

    derrint += whts[i] * fs[i] / (diff * diff) * (c * diff * cos_term - sin_term);
  }
}

static inline void prolcoef(double rlam, int k, double c, double &alpha0, double &beta0,
                            double &gamma0, double &alpha, double &beta, double &gamma) {
  double d  = k * (k - 1);
  d         = d / (2 * k + 1) / (2 * k - 1);
  double uk = d;

  d         = (k + 1) * (k + 1);
  d         = d / (2 * k + 3);
  double d2 = k * k;
  d2        = d2 / (2 * k - 1);
  double vk = (d + d2) / (2 * k + 1);

  d         = (k + 1) * (k + 2);
  d         = d / (2 * k + 1) / (2 * k + 3);
  double wk = d;

  alpha = -c * c * uk;
  beta  = rlam - k * (k + 1) - c * c * vk;
  gamma = -c * c * wk;

  alpha0 = uk;
  beta0  = vk;
  gamma0 = wk;
}

static inline void prolmatr(double *as, double *bs, double *cs, int n, double c,
                            double rlam, int ifsymm, int ifodd) {
  double done = 1.0;
  double half = done / 2.0;
  int k       = 0;

  if (ifodd > 0) {
    for (int k0 = 1; k0 <= n + 2; k0 += 2) {
      k++;
      double alpha0, beta0, gamma0, alpha, beta, gamma;
      prolcoef(rlam, k0, c, alpha0, beta0, gamma0, alpha, beta, gamma);

      as[k - 1] = alpha;
      bs[k - 1] = beta;
      cs[k - 1] = gamma;

      if (ifsymm != 0) {
        if (k0 > 1) {
          as[k - 1] = as[k - 1] / std::sqrt(k0 - 2 + half) * std::sqrt(k0 + half);
        }
        cs[k - 1] = cs[k - 1] * std::sqrt(k0 + half) / std::sqrt(k0 + half + 2);
      }
    }
  } else {
    for (int k0 = 0; k0 <= n + 2; k0 += 2) {
      k++;
      double alpha0, beta0, gamma0, alpha, beta, gamma;
      prolcoef(rlam, k0, c, alpha0, beta0, gamma0, alpha, beta, gamma);

      as[k - 1] = alpha;
      bs[k - 1] = beta;
      cs[k - 1] = gamma;

      if (ifsymm != 0) {
        if (k0 != 0) {
          as[k - 1] = as[k - 1] / std::sqrt(k0 - 2 + half) * std::sqrt(k0 + half);
        }
        cs[k - 1] = cs[k - 1] * std::sqrt(k0 + half) / std::sqrt(k0 + half + 2);
      }
    }
  }
}

static inline void prolql1(int n, double *d, double *e, int &ierr) {
  ierr = 0;
  if (n == 1) return;

  for (int i = 1; i < n; ++i) {
    e[i - 1] = e[i];
  }
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
      if (j == 30) {
        ierr = l + 1;
        return;
      }
      ++j;

      double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
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
        r        = (d[i] - g) * s + 2.0 * c * b;
        p        = s * r;
        d[i + 1] = g + p;
        g        = c * r - b;
      }

      if (r == 0.0) break;
      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
    }

    if (l == 0) continue;
    for (int i = l; i > 0; --i) {
      if (d[i] >= d[i - 1]) break;
      std::swap(d[i], d[i - 1]);
    }
  }
}

static inline void prolfact(double *a, double *b, double *c, int n, double *u, double *v,
                            double *w) {
  // Eliminate down
  for (int i = 0; i < n - 1; ++i) {
    double d = c[i + 1] / a[i];
    a[i + 1] -= b[i] * d;
    u[i] = d;
  }

  // Eliminate up
  for (int i = n - 1; i > 0; --i) {
    double d = b[i - 1] / a[i];
    v[i]     = d;
  }

  // Scale the diagonal
  double done = 1.0;
  for (int i = 0; i < n; ++i) {
    w[i] = done / a[i];
  }
}

static inline void prolsolv(const double *u, const double *v, const double *w, int n,
                            double *rhs) {
  // Eliminate down
  for (int i = 0; i < n - 1; ++i) {
    rhs[i + 1] -= u[i] * rhs[i];
  }

  // Eliminate up
  for (int i = n - 1; i > 0; --i) {
    rhs[i - 1] -= rhs[i] * v[i];
  }

  // Scale
  for (int i = 0; i < n; ++i) {
    rhs[i] *= w[i];
  }
}

static inline void prolfun0(int &ier, int n, double c, double *as, double *bs, double *cs,
                            double *xk, double *u, double *v, double *w, double eps,
                            int &nterms, double &rkhi) {
  ier          = 0;
  double delta = 1.0e-8;
  int ifsymm   = 1;
  int numit    = 4;
  double rlam  = 0;
  int ifodd    = -1;

  prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);

  prolql1(n / 2, bs, as, ier);
  if (ier != 0) {
    ier = 2048;
    return;
  }

  rkhi = -bs[n / 2 - 1];
  rlam = -bs[n / 2 - 1] + delta;

  std::fill(xk, xk + n, 1.0);

  prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);

  prolfact(bs, cs, as, n / 2, u, v, w);

  for (int ijk = 0; ijk < numit; ++ijk) {
    prolsolv(u, v, w, n / 2, xk);

    double d = 0;
    for (int j = 0; j < n / 2; ++j) {
      d += xk[j] * xk[j];
    }

    d = std::sqrt(d);
    for (int j = 0; j < n / 2; ++j) {
      xk[j] /= d;
    }

    double err = 0;
    for (int j = 0; j < n / 2; ++j) {
      err += (as[j] - xk[j]) * (as[j] - xk[j]);
      as[j] = xk[j];
    }
    err = std::sqrt(err);
  }

  double half = 0.5;
  for (int i = 0; i < n / 2; ++i) {
    if (std::abs(xk[i]) > eps) nterms = i + 1;
    xk[i] *= std::sqrt(i * 2 + half);
    cs[i] = xk[i];
  }

  int j = 0;
  for (int i = 0; i <= nterms; ++i) {
    xk[j++] = cs[i];
    xk[j++] = 0;
  }

  nterms *= 2;
}

static inline void prolps0i(int &ier, double c, double *w, int lenw, int &nterms,
                            int &ltot, double &rkhi) {
  static const std::array<int, 20> ns = {48,  64,  80,  92,  106, 120, 130,
                                         144, 156, 168, 178, 190, 202, 214,
                                         224, 236, 248, 258, 268, 280};

  double eps = 1.0e-16;
  int n      = static_cast<int>(c * 3);
  n          = n / 2;

  int i = static_cast<int>(c / 10);
  if (i <= 19) n = ns[i];

  ier     = 0;
  int ixk = 1;
  int lxk = n + 2;

  int ias = ixk + lxk;
  int las = n + 2;

  int ibs = ias + las;
  int lbs = n + 2;

  int ics = ibs + lbs;
  int lcs = n + 2;

  int iu = ics + lcs;
  int lu = n + 2;

  int iv = iu + lu;
  int lv = n + 2;

  int iw = iv + lv;
  int lw = n + 2;

  ltot = iw + lw;

  if (ltot >= lenw) {
    ier = 512;
    return;
  }

  // Call to prolfun0 (to be implemented)
  prolfun0(ier, n, c, w + ias - 1, w + ibs - 1, w + ics - 1, w + ixk - 1, w + iu - 1,
           w + iv - 1, w + iw - 1, eps, nterms, rkhi);

  if (ier != 0) return;
}

static inline void prol0ini(int &ier, double c, double *w, double &rlam20, double &rkhi,
                            int lenw, int &keep, int &ltot) {
  ier           = 0;
  double thresh = 45;
  int iw        = 11;
  w[0]          = iw + 0.1;
  w[8]          = thresh;

  // Create the data to be used in the evaluation of the function ψ^c_0(x) for x ∈ [-1,1]
  int nterms = 0;
  prolps0i(ier, c, w + iw - 1, lenw, nterms, ltot, rkhi);

  if (ier != 0) return;

  // If c >= thresh, do not prepare data for the evaluation of ψ^c_0 outside the interval
  // [-1,1]
  if (c >= thresh) {
    w[7] = c;
    w[4] = nterms + 0.1;
    keep = nterms + 3;
    return;
  }

  // Create the data to be used in the evaluation of the function ψ^c_0(x) for x outside
  // the interval [-1,1]
  int ngauss = nterms * 2;
  int lw     = nterms + 2;
  int its    = iw + lw;
  int lts    = ngauss + 2;
  int iwhts  = its + lts;
  int lwhts  = ngauss + 2;
  int ifs    = iwhts + lwhts;
  int lfs    = ngauss + 2;

  keep = ifs + lfs;
  if (keep > ltot) ltot = keep;
  if (keep >= lenw) {
    ier = 1024;
    return;
  }

  w[1] = its + 0.1;
  w[2] = iwhts + 0.1;
  w[3] = ifs + 0.1;

  int itype = 1;
  std::vector<std::vector<double>> u;
  std::vector<std::vector<double>> v;
  legeexps(itype, ngauss, w + its - 1, u, v, w + iwhts - 1);

  // Evaluate the prolate function at the Gaussian nodes
  for (int i = 0; i < ngauss; ++i) {
    legeexev(w[its + i - 2], w[ifs + i - 2], w + iw - 1, nterms - 1);
  }

  // Calculate the eigenvalue corresponding to ψ^c_0
  double rlam = 0;
  double x0   = 0;
  double f0;
  legeexev(x0, f0, w + iw - 1, nterms - 1);
  double der;
  prosinin(c, w + its - 1, w + iwhts - 1, w + ifs - 1, x0, ngauss, rlam, der);

  rlam   = rlam / f0;
  rlam20 = rlam;

  w[4] = nterms + 0.1;
  w[5] = ngauss + 0.1;
  w[6] = rlam;
  w[7] = c;
}

static inline void prol0eva(double x, const double *w, double &psi0, double &derpsi0) {
  int iw    = static_cast<int>(w[0]);
  int its   = static_cast<int>(w[1]);
  int iwhts = static_cast<int>(w[2]);
  int ifs   = static_cast<int>(w[3]);

  int nterms    = static_cast<int>(w[4]);
  int ngauss    = static_cast<int>(w[5]);
  double rlam   = w[6];
  double c      = w[7];
  double thresh = w[8];

  if (std::abs(x) > 1) {
    if (c >= thresh - 1.0e-10) {
      psi0    = 0;
      derpsi0 = 0;
      return;
    }

    prosinin(c, &w[its - 1], &w[iwhts - 1], &w[ifs - 1], x, ngauss, psi0, derpsi0);
    psi0 /= rlam;
    derpsi0 /= rlam;
    return;
  }

  legeFDER(x, psi0, derpsi0, &w[iw - 1], nterms - 2);
  // to match chebfun psi0, needs a factor of sqrt(2)
  // psi0 = sqrt(2.0) * psi0;
  // derpsi0 = sqrt(2.0) * derpsi0;
}

static inline void prol0int0r(const double *w, double r, double &val) {
  static int npts  = 200;
  static int itype = 1;
  double derpsi0;
  static std::vector<double> xs(npts, 0), ws(npts, 0), fvals(npts, 0);
  static int need_init = 1;
  std::vector<std::vector<double>> u;
  std::vector<std::vector<double>> v;

  // since xs, ws, fval of size 200 are static
  // only need to get nodes and weights once
  if (need_init) {
#pragma omp critical(PROL0INT0R)
    if (need_init) {
      legeexps(itype, npts, xs.data(), u, v, ws.data());
      need_init = 0;
    }
  }

  // Scale the nodes and weights to [0, r]
  double xs_r;
  for (int i = 0; i < npts; ++i) {
    xs_r = (xs[i] + 1) * r / 2;
    prol0eva(xs_r, w, fvals[i], derpsi0);
  }

  val = 0;
  for (int i = 0; i < npts; ++i) {
    val += ws[i] * r / 2 * fvals[i];
  }
}

struct Prolate0Fun {
  Prolate0Fun() = default;

  inline Prolate0Fun(double c_, int lenw_) : c(c_), lenw(lenw_) {
    int ier;
    workarray.resize(lenw);
    prol0ini(ier, c, workarray.data(), rlam20, rkhi, lenw, keep, ltot);
    // if (ier) error->all(FLERR,"Unable to init Prolate0Fun");
  }

  // evaluate prolate0 function val and derivative
  inline std::pair<double, double> eval_val_derivative(double x) const {
    double psi0, derpsi0;
    prol0eva(x, workarray.data(), psi0, derpsi0);
    // prol0eva_(&x, workarray.data(), &psi0, &derpsi0);
    // std::pair<double, double> psi0_derpsi0{psi0, derpsi0};
    return {psi0, derpsi0};
  }

  // evaluate prolate0 function value
  inline double eval_val(double x) const {
    auto [val, dum] = eval_val_derivative(x);
    return val;
  }

  // evaluate prolate0 function derivative
  inline double eval_derivative(double x) const {
    auto [dum, der] = eval_val_derivative(x);
    return der;
  }

  // int_0^r prolate0(x) dx
  inline double int_eval(double r) const {
    double val;
    prol0int0r(workarray.data(), r, val);
    // prol0int0r_(workarray.data(), &r, &val);
    return val;
  }

  double c;
  int lenw, keep, ltot;
  std::vector<double> workarray;
  double rlam20, rkhi;
};

/*
evaluate prolate0c derivative at x, i.e., \psi_0^c(x)
*/
double prolate0_eval_derivative(double c, double x) {
  static std::unordered_map<double, Prolate0Fun> prolate0_funcs_cache;
  if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_EVAL)
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
      prolate0_funcs_cache.emplace(c, Prolate0Fun(c, 10000));
    }
  }
  return prolate0_funcs_cache[c].eval_derivative(x);
}

/*
evaluate prolate0c at x, i.e., \psi_0^c(x)
*/
double prolate0_eval(double c, double x) {
  static std::unordered_map<double, Prolate0Fun> prolate0_funcs_cache;
  if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_EVAL)
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
      prolate0_funcs_cache.emplace(c, Prolate0Fun(c, 10000));
    }
  }
  return prolate0_funcs_cache[c].eval_val(x);
}

/*
evaluate prolate0c function integral of \int_0^r \psi_0^c(x) dx
*/
double prolate0_int_eval(double c, double r) {
  static std::unordered_map<double, Prolate0Fun> prolate0_funcs_cache;
  if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
#pragma omp critical(PROLATE0_INT_EVAL)
    if (prolate0_funcs_cache.find(c) == prolate0_funcs_cache.end()) {
      prolate0_funcs_cache.emplace(c, Prolate0Fun(c, 10000));
    }
  }
  return prolate0_funcs_cache[c].int_eval(r);
}
// end of prolate functions

// start of ducc pswf
double sdp_pswf_cipow(double base, int exp) {
  double result = 1;
  // This is never called with negative exponents. Avoids potential recursion.
  if (exp == 1) return base;
  while (exp) {
    if ((exp & 1) != 0) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

static inline void pswf_sdmn(const int m, const int n, const double c, const double cv,
                             const int kd, double *df) {
  /*       ===================================================== */
  /*       Purpose: Compute the expansion coefficients of the */
  /*                prolate and oblate spheroidal functions, dk */
  /*       Input :  m  --- Mode parameter */
  /*                n  --- Mode parameter */
  /*                c  --- Spheroidal parameter */
  /*                cv --- Characteristic value */
  /*                KD --- Function code */
  /*                       KD=1 for prolate; KD=-1 for oblate */
  /*       Output:  DF(k) --- Expansion coefficients dk; */
  /*                          DF(1), DF(2), ... correspond to */
  /*                          d0, d2, ... for even n-m and d1, */
  /*                          d3, ... for odd n-m */
  /*       ===================================================== */

  int nm = int((n - m) * .5 + c) + 25;
  if (c < 1e-10) {
    for (int i = 0; i < nm; ++i) df[i] = 0.;
    df[(n - m) / 2] = 1.;
    return;
  }
  double cs = c * c * kd;
  int ip    = 1;
  int k     = 0;
  if (n - m == (n - m) / 2 << 1) ip = 0;
  std::vector<double> a(nm + 2), d_(nm + 2), g(nm + 2);
  for (int i = 0; i < nm + 2; ++i) {
    if (ip == 0) k = i << 1;
    if (ip == 1) k = (i << 1) + 1;
    double dk0 = double(m + k);
    double dk1 = double(m + k + 1);
    double dk2 = double((m + k) << 1);
    double d2k = double((m << 1) + k);
    a[i]       = (d2k + 2.) * (d2k + 1.) / ((dk2 + 3.) * (dk2 + 5.)) * cs;
    d_[i] =
        dk0 * dk1 + (dk0 * 2. * dk1 - m * 2. * m - 1.) / ((dk2 - 1.) * (dk2 + 3.)) * cs;
    g[i] = k * (k - 1.) / ((dk2 - 3.) * (dk2 - 1.)) * cs;
  }
  double fs = 1.;
  double f1 = 0.;
  double f0 = 1e-100;
  int kb    = 0;
  df[nm]    = 0.;
  double fl = 0.;
  for (k = nm; k >= 1; --k) {
    double f = -((d_[k] - cv) * f0 + a[k] * f1) / g[k];
    if (std::abs(f) > std::abs(df[k])) {
      df[k - 1] = f;
      f1        = f0;
      f0        = f;
      if (std::abs(f) > 1e100) {
        for (int k1 = k; k1 <= nm; ++k1) df[k1 - 1] *= 1e-100;
        f1 *= 1e-100;
        f0 *= 1e-100;
      }
    } else {
      kb        = k;
      fl        = df[k];
      f1        = 1e-100;
      double f2 = -(d_[0] - cv) / a[0] * f1;
      df[0]     = f1;
      if (kb == 1)
        fs = f2;
      else if (kb == 2) {
        df[1] = f2;
        fs    = -((d_[1] - cv) * f2 + g[1] * f1) / a[1];
      } else {
        df[1] = f2;
        for (int j = 3; j <= kb + 1; ++j) {
          double f = -((d_[j - 2] - cv) * f2 + g[j - 2] * f1) / a[j - 2];
          if (j <= kb) df[j - 1] = f;
          if (std::abs(f) > 1e100) {
            for (int k1 = 0; k1 < j; ++k1) df[k1] *= 1e-100;
            f *= 1e-100;
            f2 *= 1e-100;
          }
          f1 = f2;
          f2 = f;
        }
        fs = f;
      }
      break;
    }
  }

  double r1 = 1.;
  for (int j = m + ip + 1; j <= (m + ip) << 1; ++j) r1 *= j;
  double su1 = df[0] * r1;
  for (k = 2; k <= kb; ++k) {
    r1 = -r1 * (k + m + ip - 1.5) / (k - 1.);
    su1 += r1 * df[k - 1];
  }
  double su2 = 0.;
  double sw  = 0.;
  for (k = kb + 1; k <= nm; ++k) {
    if (k != 1) r1 = -r1 * (k + m + ip - 1.5) / (k - 1.);
    su2 += r1 * df[k - 1];
    if (std::abs(sw - su2) < std::abs(su2) * 1e-14) break;
    sw = su2;
  }

  double r3 = 1.;
  for (int j = 1; j <= (m + n + ip) / 2; ++j) r3 *= j + (n + m + ip) * .5;
  double r4 = 1.;
  for (int j = 1; j <= (n - m - ip) / 2; ++j) r4 *= -4. * j;
  double s0 = r3 / (fl * (su1 / fs) + su2) / r4;
  for (int k = 0; k < kb; ++k) df[k] *= fl / fs * s0;
  for (k = kb; k < nm; ++k) df[k] *= s0;
}

static inline void pswf_sckb(int m, int n, double c, const double *df, double *ck) {
  /*       ====================================================== */
  /*       Purpose: Compute the expansion coefficients of the */
  /*                prolate and oblate spheroidal functions */
  /*       Input :  m  --- Mode parameter */
  /*                n  --- Mode parameter */
  /*                c  --- Spheroidal parameter */
  /*                DF(k) --- Expansion coefficients dk */
  /*       Output:  CK(k) --- Expansion coefficients ck; */
  /*                          CK(1), CK(2), ... correspond to */
  /*                          c0, c2, ... */
  /*       ====================================================== */
  // df[nm+1], ck[nm]
  if (c <= 1e-10) c = 1e-10;
  int nm = int((n - m) * .5 + c) + 25;
  int ip = 1;
  if (n - m == (n - m) / 2 << 1) ip = 0;
  double reg = 1.;
  if (m + nm > 80) reg = 1e-200;
  double fac = -sdp_pswf_cipow(.5, m);
  double sw  = 0.;
  for (int k = 0; k <= nm - 1; ++k) {
    fac      = -fac;
    int i1   = (k << 1) + ip + 1;
    double r = reg;
    for (int i = i1; i <= i1 + (m << 1) - 1; ++i) r *= i;
    int i2 = k + m + ip;
    for (int i = i2; i <= i2 + k - 1; ++i) r *= i + .5;
    double sum = r * df[k];
    for (int i = k + 1; i <= nm; ++i) {
      double d1 = i * 2. + ip;
      double d2 = m * 2. + d1;
      double d3 = i + m + ip - .5;
      r         = r * d2 * (d2 - 1.) * i * (d3 + k) / (d1 * (d1 - 1.) * (i - k) * d3);
      sum += r * df[i];
      if (std::abs(sw - sum) < std::abs(sum) * 1e-14) break;
      sw = sum;
    }

    double r1 = reg;
    for (int i = 2; i <= m + k; ++i) r1 *= i;
    ck[k] = fac * sum / r1;
  }
}

static inline void pswf_segv(const int m, const int n, const double c, const int kd,
                             double *cv, double *eg) {
  std::vector<double> a(300), b(100), d_(300), e(300), f(300), g(300), h_(100);
  std::vector<double> cv0(100);

  /*       ========================================================= */
  /*       Purpose: Compute the characteristic values of spheroidal */
  /*                wave functions */
  /*       Input :  m  --- Mode parameter */
  /*                n  --- Mode parameter */
  /*                c  --- Spheroidal parameter */
  /*                KD --- Function code */
  /*                       KD=1 for Prolate; KD=-1 for Oblate */
  /*       Output:  CV --- Characteristic value for given m, n and c */
  /*                EG(L) --- Characteristic value for mode m and n' */
  /*                          ( L = n' - m + 1 ) */
  /*       ========================================================= */

  if (c < 1e-10) {
    for (int i = 0; i < n - m + 1; ++i) eg[i] = (i + m + 1) * (i + m);
    *cv = eg[n - m];
    return;
  }
  int icm   = (n - m + 2) / 2;
  int nm    = int((n - m) * .5 + c) + 10;
  double cs = c * c * kd;
  int k     = 0;
  for (int l = 0; l <= 1; ++l) {
    for (int i = 1; i <= nm; ++i) {
      if (l == 0) k = (i - 1) << 1;
      if (l == 1) k = (i << 1) - 1;
      double dk0 = double(m + k);
      double dk1 = double(m + k + 1);
      double dk2 = double((m + k) << 1);
      double d2k = double((m << 1) + k);
      a[i - 1]   = (d2k + 2.) * (d2k + 1.) / ((dk2 + 3.) * (dk2 + 5.)) * cs;
      d_[i - 1] =
          dk0 * dk1 + (dk0 * 2. * dk1 - m * 2. * m - 1.) / ((dk2 - 1.) * (dk2 + 3.)) * cs;
      g[i - 1] = k * (k - 1.) / ((dk2 - 3.) * (dk2 - 1.)) * cs;
    }
    for (k = 1; k < nm; ++k) {
      e[k] = sqrt(a[k - 1] * g[k]);
      f[k] = e[k] * e[k];
    }
    f[0] = e[0] = 0.;
    double xa   = d_[nm - 1] + std::abs(e[nm - 1]);
    double xb   = d_[nm - 1] - std::abs(e[nm - 1]);
    int nm1     = nm - 1;
    for (int i = 0; i < nm1; ++i) {
      double t = std::abs(e[i]) + std::abs(e[i + 1]);
      xa       = std::max(xa, d_[i] + t);
      xb       = std::min(xb, d_[i] - t);
    }
    for (int i = 0; i < icm; ++i) {
      b[i]  = xa;
      h_[i] = xb;
    }
    for (k = 1; k <= icm; ++k) {
      for (int k1 = k; k1 <= icm; ++k1)
        if (b[k1 - 1] < b[k - 1]) {
          b[k - 1] = b[k1 - 1];
          break;
        }

      if (k != 1)
        if (h_[k - 1] < h_[k - 2]) h_[k - 1] = h_[k - 2];

      double x1;
      while (true) {
        x1         = (b[k - 1] + h_[k - 1]) / 2.;
        cv0[k - 1] = x1;
        if (std::abs((b[k - 1] - h_[k - 1]) / x1) < 1e-14) break;

        int j    = 0;
        double s = 1.;
        for (int i = 1; i <= nm; ++i) {
          if (s == 0.) s += 1e-30;
          s = d_[i - 1] - f[i - 1] / s - x1;
          if (s < 0.) ++j;
        }
        if (j < k)
          h_[k - 1] = x1;
        else {
          b[k - 1] = x1;
          if (j >= icm)
            b[icm - 1] = x1;
          else {
            h_[j]    = std::max(h_[j], x1);
            b[j - 1] = std::min(b[j - 1], x1);
          }
        }
      }

      cv0[k - 1] = x1;
      if (l == 0) eg[k * 2 - 2] = cv0[k - 1];
      if (l == 1) eg[k * 2 - 1] = cv0[k - 1];
    }
  }
  *cv = eg[n - m];
}

/**
 * @brief Evaluate PSWF at a specific point.
 *
 * Compute the prolate and oblate spheroidal angular functions of the first
 * kind and their derivatives.
 *
 * This function has been heavily specialised, as it it basically the
 * inner loop of PSWF generation.
 *
 * @param m Mode parameter, m = 0, 1, 2, ...
 * @param n Mode parameter, n = m, m + 1, ...
 * @param c Spheroidal parameter.
 * @param ck Expansion coefficients; CK(1), CK(2) ... correspond to c0, c2 ...
 * @param x Argument of angular function, |x| < 1.0
 * @return Angular function of the first kind.
 */
double sdp_pswf_aswfa(int m, int n, double c, const double *ck, double x) {
  const int nm    = (int)((n - m) / 2 + c) + 40;
  const int nm2   = nm / 2 - 2;
  const double x1 = 1.0 - x * x;
  const double a0 = (m == 0 && x1 == 0.0) ? 1.0 : pow(x1, m * 0.5);

  double su1 = ck[0];
  double x1p = x1;
  for (int k = 1; k <= nm2; ++k, x1p *= x1) {
    const double r = ck[k] * x1p;
    su1 += r;
    const double t = r / su1;
    if (k >= 10 && abs(t) < 1e-14) break;
  }
  return ((n - m) % 2 == 0) ? (a0 * su1) : (a0 * x * su1);
}
// end of ducc pswf

double pswf(double c, double x) {
  if (std::abs(x) > 1.0) return 0.0; // restrict support to [-1,1]

  return prolate0_eval(c, x) / prolate0_eval(c, 0.0);
}

double pswf_ducc(double c, double x) {
  int m  = 0;
  int n  = 0;
  int kd = 1;
  std::vector<double> coeff(2000);
  std::vector<double> df(2000);
  double cv = 0.0, eg[200] = {0.0, 0.0};
  pswf_segv(m, n, c, kd, &cv, eg);
  pswf_sdmn(m, n, c, cv, kd, df.data());
  pswf_sckb(m, n, c, df.data(), coeff.data());
  return sdp_pswf_aswfa(m, n, c, coeff.data(), x); // / sdp_pswf_aswfa(m, n, c,
                                                   // coeff.data(), 0.0);
}

} // namespace finufft::common

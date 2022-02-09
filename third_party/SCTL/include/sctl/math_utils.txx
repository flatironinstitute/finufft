#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

namespace SCTL_NAMESPACE {

template <class Real> inline Real machine_eps() {
  auto compute_machine_eps = [](){
    Real eps = (Real)1;
    while (eps + (Real)1 > 1) eps *= (Real)0.5;
    return eps;
  };
  static const Real eps = compute_machine_eps();
  return eps;
}

template <class Real> inline Real atoreal(const char* str) { // Warning: does not do correct rounding
  int i = 0;
  Real sign = 1.0;
  for (; str[i] != '\0'; i++) {
    char c = str[i];
    if (c == '-') sign = -sign;
    if (c >= '0' && c <= '9') break;
  }

  Real val = 0.0;
  for (; str[i] != '\0'; i++) {
    char c = str[i];
    if (c >= '0' && c <= '9')
      val = val * 10 + (c - '0');
    else
      break;
  }

  if (str[i] == '.') {
    i++;
    Real exp = 1.0;
    exp /= 10;
    for (; str[i] != '\0'; i++) {
      char c = str[i];
      if (c >= '0' && c <= '9')
        val = val + (c - '0') * exp;
      else
        break;
      exp /= 10;
    }
  }

  return sign * val;
}

template <class Real> static inline constexpr Real const_pi_generic() {
  return 113187804032455044LL*pow<-55,Real>(2) + 59412220499329904LL*pow<-112,Real>(2);
}

template <class Real> static inline constexpr Real const_e_generic() {
  return 97936424237889173LL*pow<-55,Real>(2) + 30046841068362992LL*pow<-112,Real>(2);
}

template <class Real> static inline constexpr Real fabs_generic(const Real a) {
  return (a<0?-a:a);
}

template <class Real> static inline Real sqrt_generic(const Real a) {
  Real b = ::sqrt((double)a);
  if (a > 0) { // Newton iterations for greater accuracy
    b = (b + a / b) * 0.5;
    b = (b + a / b) * 0.5;
  }
  return b;
}

template <class Real> static inline Real sin_generic(const Real a) {
  const int N = 200;
  static std::vector<Real> theta;
  static std::vector<Real> sinval;
  static std::vector<Real> cosval;
  if (theta.size() == 0) {
#pragma omp critical(SCTL_QUAD_SIN)
    if (theta.size() == 0) {
      sinval.resize(N);
      cosval.resize(N);

      Real t = 1.0;
      std::vector<Real> theta_(N);
      for (int i = 0; i < N; i++) {
        theta_[i] = t;
        t = t * 0.5;
      }

      sinval[N - 1] = theta_[N - 1];
      cosval[N - 1] = 1.0 - sinval[N - 1] * sinval[N - 1] / 2;
      for (int i = N - 2; i >= 0; i--) {
        sinval[i] = 2.0 * sinval[i + 1] * cosval[i + 1];
        cosval[i] = sqrt<Real>(1.0 - sinval[i] * sinval[i]);
      }
      theta_.swap(theta);
    }
  }

  Real t = (a < 0.0 ? -a : a);
  Real sval = 0.0;
  Real cval = 1.0;
  for (int i = 0; i < N; i++) {
    while (theta[i] <= t) {
      Real sval_ = sval * cosval[i] + cval * sinval[i];
      Real cval_ = cval * cosval[i] - sval * sinval[i];
      sval = sval_;
      cval = cval_;
      t = t - theta[i];
    }
  }
  return (a < 0.0 ? -sval : sval);
}

template <class Real> static inline Real cos_generic(const Real a) {
  const int N = 200;
  static std::vector<Real> theta;
  static std::vector<Real> sinval;
  static std::vector<Real> cosval;
  if (theta.size() == 0) {
#pragma omp critical(SCTL_QUAD_COS)
    if (theta.size() == 0) {
      sinval.resize(N);
      cosval.resize(N);

      Real t = 1.0;
      std::vector<Real> theta_(N);
      for (int i = 0; i < N; i++) {
        theta_[i] = t;
        t = t * 0.5;
      }

      sinval[N - 1] = theta_[N - 1];
      cosval[N - 1] = 1.0 - sinval[N - 1] * sinval[N - 1] / 2;
      for (int i = N - 2; i >= 0; i--) {
        sinval[i] = 2.0 * sinval[i + 1] * cosval[i + 1];
        cosval[i] = sqrt<Real>(1.0 - sinval[i] * sinval[i]);
      }

      theta_.swap(theta);
    }
  }

  Real t = (a < 0.0 ? -a : a);
  Real sval = 0.0;
  Real cval = 1.0;
  for (int i = 0; i < N; i++) {
    while (theta[i] <= t) {
      Real sval_ = sval * cosval[i] + cval * sinval[i];
      Real cval_ = cval * cosval[i] - sval * sinval[i];
      sval = sval_;
      cval = cval_;
      t = t - theta[i];
    }
  }
  return cval;
}

template <class Real> static inline Real acos_generic(const Real a) {
  Real b = ::acos((double)a);
  if (b > 0) { // Newton iterations for greater accuracy
    b += (cos<Real>(b)-a)/sin<Real>(b);
    b += (cos<Real>(b)-a)/sin<Real>(b);
  }
  return b;
}

template <class Real> static inline Real exp_generic(const Real a) {
  const int N = 200;
  static std::vector<Real> theta0;
  static std::vector<Real> theta1;
  static std::vector<Real> expval0;
  static std::vector<Real> expval1;
  if (theta0.size() == 0) {
#pragma omp critical(SCTL_QUAD_EXP)
    if (theta0.size() == 0) {
      std::vector<Real> theta0_(N);
      theta1.resize(N);
      expval0.resize(N);
      expval1.resize(N);

      theta0_[0] = 1.0;
      theta1[0] = 1.0;
      expval0[0] = const_e<Real>();
      expval1[0] = const_e<Real>();
      for (int i = 1; i < N; i++) {
        theta0_[i] = theta0_[i - 1] * 0.5;
        theta1[i] = theta1[i - 1] * 2.0;
        expval0[i] = sqrt<Real>(expval0[i - 1]);
        expval1[i] = expval1[i - 1] * expval1[i - 1];
      }
      theta0.swap(theta0_);
    }
  }

  Real t = (a < 0.0 ? -a : a);
  Real eval = 1.0;
  for (int i = N - 1; i > 0; i--) {
    while (theta1[i] <= t) {
      eval = eval * expval1[i];
      t = t - theta1[i];
    }
  }
  for (int i = 0; i < N; i++) {
    while (theta0[i] <= t) {
      eval = eval * expval0[i];
      t = t - theta0[i];
    }
  }
  eval = eval * (1.0 + t);
  return (a < 0.0 ? 1.0 / eval : eval);
}

template <class Real> static inline Real log_generic(const Real a) {
  if (a == 0) return (Real)NAN;
  Real y0 = ::log((double)a);
  { // Newton iterations
    y0 = y0 + (a / exp<Real>(y0) - 1.0);
    y0 = y0 + (a / exp<Real>(y0) - 1.0);
  }
  return y0;
}

template <class Real> static inline Real pow_generic(const Real b, const Real e) {
  if (e == 0) return 1;
  if (b == 0) return 0;
  if (b < 0) {
    Long e_ = (Long)e;
    SCTL_ASSERT(e == (Real)e_);
    return exp<Real>(log<Real>(-b) * e) * (e_ % 2 ? (Real)-1 : (Real)1.0);
  }
  return exp<Real>(log<Real>(b) * e);
}
template <class ValueType> static inline constexpr ValueType pow_integer_exp(ValueType b, Long e) {
  return (e > 0) ? ((e & 1) ? b : ValueType(1)) * pow_integer_exp(b*b, e>>1) : ValueType(1);
}
template <class Real, class ExpType> class pow_wrapper {
  public:
    static Real pow(Real b, ExpType e) {
      return (Real)std::pow(b, e);
    }
};
template <class ValueType> class pow_wrapper<ValueType,Long> {
  public:
    static constexpr ValueType pow(ValueType b, Long e) {
      return (e > 0) ? pow_integer_exp(b, e) : 1/pow_integer_exp(b, -e);
    }
};

template <Long e, class ValueType> inline constexpr ValueType pow(ValueType b) {
  return (e > 0) ? pow_integer_exp<ValueType>(b, e) : 1/pow_integer_exp<ValueType>(b, -e);
}

template <class Real> inline std::ostream& ostream_insertion_generic(std::ostream& output, const Real q_) {
  // int width=output.width();
  output << std::setw(1);

  Real q = q_;
  if (q < 0.0) {
    output << "-";
    q = -q;
  } else if (q > 0) {
    output << " ";
  } else {
    output << " ";
    output << "0.0";
    return output;
  }

  int exp = 0;
  static const Real ONETENTH = (Real)1 / 10;
  while (q < 1.0 && abs(exp) < 10000) {
    q = q * 10;
    exp--;
  }
  while (q >= 10 && abs(exp) < 10000) {
    q = q * ONETENTH;
    exp++;
  }

  for (int i = 0; i < 34; i++) {
    output << (int)q;
    if (i == 0) output << ".";
    q = (q - int(q)) * 10;
    if (q == 0 && i > 0) break;
  }

  if (exp > 0) {
    std::cout << "e+" << exp;
  } else if (exp < 0) {
    std::cout << "e" << exp;
  }

  return output;
}



#ifdef SCTL_QUAD_T
template <> inline constexpr QuadReal const_pi<QuadReal>() { return const_pi_generic<QuadReal>(); }

template <> inline constexpr QuadReal const_e<QuadReal>() { return const_e_generic<QuadReal>(); }

template <> inline QuadReal fabs<QuadReal>(const QuadReal a) { return fabs_generic(a); }

template <> inline QuadReal sqrt<QuadReal>(const QuadReal a) { return sqrt_generic(a); }

template <> inline QuadReal sin<QuadReal>(const QuadReal a) { return sin_generic(a); }

template <> inline QuadReal cos<QuadReal>(const QuadReal a) { return cos_generic(a); }

template <> inline QuadReal acos<QuadReal>(const QuadReal a) { return acos_generic(a); }

template <> inline QuadReal exp<QuadReal>(const QuadReal a) { return exp_generic(a); }

template <> inline QuadReal log<QuadReal>(const QuadReal a) { return log_generic(a); }

template <class ExpType> class pow_wrapper<QuadReal,ExpType> {
  public:
    static QuadReal pow(QuadReal b, ExpType e) {
      return pow_generic<QuadReal>(b, (QuadReal)e);
    }
};
template <> class pow_wrapper<QuadReal,Long> {
  public:
    static constexpr QuadReal pow(QuadReal b, Long e) {
      return (e > 0) ? pow_integer_exp(b, e) : 1/pow_integer_exp(b, -e);
    }
};

inline std::ostream& operator<<(std::ostream& output, const QuadReal& q) { return ostream_insertion_generic(output, q); }
inline std::istream& operator>>(std::istream& inputstream, QuadReal& x) {
  long double x_;
  inputstream>>x_;
  x = x_;
  return inputstream;
}
#endif



template <class Real, class ExpType> inline Real pow(const Real b, const ExpType e) {
  return pow_wrapper<Real,ExpType>::pow(b, e);
}

} // end namespace


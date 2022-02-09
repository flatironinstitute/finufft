#ifndef _SCTL_VEC_WRAPPER_HPP_
#define _SCTL_VEC_WRAPPER_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(intrin-wrapper.hpp)

#include <cassert>
#include <cstdint>
#include <ostream>

namespace SCTL_NAMESPACE {

  #if defined(__AVX512__) || defined(__AVX512F__)
  static_assert(SCTL_ALIGN_BYTES >= 64, "Insufficient memory alignment for SIMD vector types");
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 64/sizeof(ScalarType); }
  #elif defined(__AVX__)
  static_assert(SCTL_ALIGN_BYTES >= 32, "Insufficient memory alignment for SIMD vector types");
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 32/sizeof(ScalarType); }
  #elif defined(__SSE4_2__)
  static_assert(SCTL_ALIGN_BYTES >= 16, "Insufficient memory alignment for SIMD vector types");
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 16/sizeof(ScalarType); }
  #else
  static_assert(SCTL_ALIGN_BYTES >= 8, "Insufficient memory alignment for SIMD vector types");
  template <class ScalarType> constexpr Integer DefaultVecLen() { return 1; }
  #endif

  template <class ValueType, Integer N = DefaultVecLen<ValueType>()> class alignas(sizeof(ValueType) * N) Vec {
    public:
      using ScalarType = ValueType;
      using VData = VecData<ScalarType,N>;
      using MaskType = Mask<VData>;

      static constexpr Integer Size() {
        return N;
      }

      static Vec Zero() {
        Vec r;
        r.v = zero_intrin<VData>();
        return r;
      }

      static Vec Load1(ScalarType const* p) {
        Vec r;
        r.v = load1_intrin<VData>(p);
        return r;
      }
      static Vec Load(ScalarType const* p) {
        Vec r;
        r.v = loadu_intrin<VData>(p);
        return r;
      }
      static Vec LoadAligned(ScalarType const* p) {
        Vec r;
        r.v = load_intrin<VData>(p);
        return r;
      }

      Vec() = default;
      Vec(const Vec&) = default;
      Vec& operator=(const Vec&) = default;
      ~Vec() = default;

      Vec(const VData& v_) : v(v_) {}
      Vec(const ScalarType& a) : Vec(set1_intrin<VData>(a)) {}
      template <class T,class ...T1> Vec(T x, T1... args) : Vec(InitVec<T1...>::template apply<ScalarType>((ScalarType)x,args...)) {}

      void Store(ScalarType* p) const {
        storeu_intrin(p,v);
      }
      void StoreAligned(ScalarType* p) const {
        store_intrin(p,v);
      }

      // Conversion operators
      friend Mask<VData> convert2mask(const Vec& a) {
        return convert_vec2mask_intrin(a.v);
      }
      friend Vec RoundReal2Real(const Vec& x) {
        return round_real2real_intrin(x.v);
      }
      template <class IntVec, class RealVec> friend IntVec RoundReal2Int(const RealVec& x);
      template <class RealVec, class IntVec> friend RealVec ConvertInt2Real(const IntVec& x);

      // Element access
      ScalarType operator[](Integer i) const {
        return extract_intrin(v,i);
      }
      void insert(Integer i, ScalarType value) {
        insert_intrin(v,i,value);
      }

      // Arithmetic operators
      Vec operator+() const {
        return *this;
      }
      Vec operator-() const {
        return unary_minus_intrin(v); // Zero() - (*this);
      }
      friend Vec operator*(const Vec& a, const Vec& b) {
        return mul_intrin(a.v, b.v);
      }
      friend Vec operator/(const Vec& a, const Vec& b) {
        return div_intrin(a.v, b.v);
      }
      friend Vec operator+(const Vec& a, const Vec& b) {
        return add_intrin(a.v, b.v);
      }
      friend Vec operator-(const Vec& a, const Vec& b) {
        return sub_intrin(a.v, b.v);
      }
      friend Vec FMA(const Vec& a, const Vec& b, const Vec& c) {
        return fma_intrin(a.v, b.v, c.v);
      }

      // Comparison operators
      friend Mask<VData> operator< (const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::lt>(a.v, b.v);
      }
      friend Mask<VData> operator<=(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::le>(a.v, b.v);
      }
      friend Mask<VData> operator>=(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::ge>(a.v, b.v);
      }
      friend Mask<VData> operator> (const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::gt>(a.v, b.v);
      }
      friend Mask<VData> operator==(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::eq>(a.v, b.v);
      }
      friend Mask<VData> operator!=(const Vec& a, const Vec& b) {
        return comp_intrin<ComparisonType::ne>(a.v, b.v);
      }
      friend Vec select(const Mask<VData>& m, const Vec& a, const Vec& b) {
        return select_intrin(m, a.v, b.v);
      }

      // Bitwise operators
      Vec operator~() const {
        return not_intrin(v);
      }
      friend Vec operator&(const Vec& a, const Vec& b) {
        return and_intrin(a.v, b.v);
      }
      friend Vec operator^(const Vec& a, const Vec& b) {
        return xor_intrin(a.v, b.v);
      }
      friend Vec operator|(const Vec& a, const Vec& b) {
        return or_intrin(a.v, b.v);
      }
      friend Vec AndNot(const Vec& a, const Vec& b) { // return a & ~b
        return andnot_intrin(a.v, b.v);
      }

      // Bitshift
      friend Vec operator<<(const Vec& lhs, const Integer& rhs) {
        return bitshiftleft_intrin(lhs.v, rhs);
      }
      friend Vec operator>>(const Vec& lhs, const Integer& rhs) {
        return bitshiftright_intrin(lhs.v, rhs);
      }

      // Assignment operators
      Vec& operator=(const ScalarType& a) {
        v = set1_intrin<VData>(a);
        return *this;
      }
      Vec& operator*=(const Vec& rhs) {
        v = mul_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator/=(const Vec& rhs) {
        v = div_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator+=(const Vec& rhs) {
        v = add_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator-=(const Vec& rhs) {
        v = sub_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator&=(const Vec& rhs) {
        v = and_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator^=(const Vec& rhs) {
        v = xor_intrin(v, rhs.v);
        return *this;
      }
      Vec& operator|=(const Vec& rhs) {
        v = or_intrin(v, rhs.v);
        return *this;
      }

      // Other operators
      friend Vec max(const Vec& lhs, const Vec& rhs) {
        return max_intrin(lhs.v, rhs.v);
      }
      friend Vec min(const Vec& lhs, const Vec& rhs) {
        return min_intrin(lhs.v, rhs.v);
      }

      // Special functions
      template <Integer digits, class RealVec> friend RealVec approx_rsqrt(const RealVec& x);
      template <Integer digits, class RealVec> friend RealVec approx_rsqrt(const RealVec& x, const typename RealVec::MaskType& m);

      friend void sincos(Vec& sinx, Vec& cosx, const Vec& x) {
        sincos_intrin(sinx.v, cosx.v, x.v);
      }
      template <Integer digits, class RealVec> friend void approx_sincos(RealVec& sinx, RealVec& cosx, const RealVec& x);

      friend Vec exp(const Vec& x) {
        return exp_intrin(x.v);
      }
      template <Integer digits, class RealVec> friend RealVec approx_exp(const RealVec& x);


      //template <class Vec1, class Vec2> friend Vec1 reinterpret(const Vec2& x);
      //template <class Vec> friend Vec RoundReal2Real(const Vec& x);
      //template <class Vec> friend void exp_intrin(Vec& expx, const Vec& x);

      // Print
      friend std::ostream& operator<<(std::ostream& os, const Vec& in) {
        for (Integer i = 0; i < Size(); i++) os << in[i] << ' ';
        return os;
      }

      void set(const VData& v_) { v = v_; }
      const VData& get() const { return v; }

    private:

      template <class T, class... T2> struct InitVec {
        template <class... T1> static VData apply(T1... start, T x, T2... rest) {
          return InitVec<T2...>::template apply<ScalarType, T1...>(start..., (ScalarType)x, rest...);
        }
      };
      template <class T> struct InitVec<T> {
        template <class... T1> static VData apply(T1... start, T x) {
          return set_intrin<VData>(start..., (ScalarType)x);
        }
      };

      VData v;
  };

  // Conversion operators
  template <class RealVec, class IntVec> RealVec ConvertInt2Real(const IntVec& x) {
    return convert_int2real_intrin<typename RealVec::VData>(x.v);
  }
  template <class IntVec, class RealVec> IntVec RoundReal2Int(const RealVec& x) {
    return round_real2int_intrin<typename IntVec::VData>(x.v);
  }
  template <class MaskType> Vec<typename MaskType::ScalarType,MaskType::Size> convert2vec(const MaskType& a) {
    return convert_mask2vec_intrin(a);
  }

  // Special functions
  template <Integer digits, class RealVec> RealVec approx_rsqrt(const RealVec& x) {
    static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<typename RealVec::ScalarType>::SigBits*0.3010299957) : digits);
    return rsqrt_approx_intrin<digits_, typename RealVec::VData>::eval(x.v);
  }
  template <Integer digits, class RealVec> RealVec approx_rsqrt(const RealVec& x, const typename RealVec::MaskType& m) {
    static constexpr Integer digits_ = (digits==-1 ? (Integer)(TypeTraits<typename RealVec::ScalarType>::SigBits*0.3010299957) : digits);
    return rsqrt_approx_intrin<digits_, typename RealVec::VData>::eval(x.v, m);
  }

  template <Integer digits, class RealVec> void approx_sincos(RealVec& sinx, RealVec& cosx, const RealVec& x) {
    constexpr Integer ORDER = (digits>1?digits>9?digits>14?digits>17?digits-1:digits:digits+1:digits+2:1);
    if (digits == -1 || ORDER > 20) sincos(sinx, cosx, x);
    else approx_sincos_intrin<ORDER>(sinx.v, cosx.v, x.v);
  }

  template <Integer digits, class RealVec> RealVec approx_exp(const RealVec& x) {
    constexpr Integer ORDER = digits;
    if (digits == -1 || ORDER > 13) return exp(x);
    else return approx_exp_intrin<ORDER>(x.v);
  }

  // Other operators
  template <class ValueType> void printb(const ValueType& x) { // print binary
    union {
      ValueType v;
      uint8_t c[sizeof(ValueType)];
    } u = {x};
    //std::cout<<std::setw(10)<<x<<' ';
    for (Integer i = 0; i < (Integer)sizeof(ValueType); i++) {
      for (Integer j = 0; j < 8; j++) {
        std::cout<<((u.c[i] & (1U<<j))?'1':'0');
      }
    }
    std::cout<<'\n';
  }

}

#endif  //_SCTL_VEC_WRAPPER_HPP_

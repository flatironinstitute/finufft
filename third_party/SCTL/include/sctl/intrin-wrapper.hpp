#ifndef _SCTL_INTRIN_WRAPPER_HPP_
#define _SCTL_INTRIN_WRAPPER_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(math_utils.hpp)

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE4_2__
#include <smmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif
#if defined(__MIC__)
#include <immintrin.h>
#endif

// TODO: Check alignment when SCTL_MEMDEBUG is defined
// TODO: Replace pointers with iterators


namespace SCTL_NAMESPACE { // Traits

  enum class DataType {
    Integer,
    Real
  };

  template <class ValueType> class TypeTraits {
  };
  template <> class TypeTraits<int8_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int8_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
  template <> class TypeTraits<int16_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int16_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
  template <> class TypeTraits<int32_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int32_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
  template <> class TypeTraits<int64_t> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(int64_t);
      static constexpr Integer SigBits = Size * 8 - 1;
  };
#ifdef __SIZEOF_INT128__
  template <> class TypeTraits<__int128> {
    public:
      static constexpr DataType Type = DataType::Integer;
      static constexpr Integer Size = sizeof(__int128);
      static constexpr Integer SigBits = Size * 16 - 1;
  };
#endif

  template <class Real, Integer bits = sizeof(Real)*8> struct GetSigBits {
    static constexpr Integer value() {
      return (pow<bits>((Real)0.5)+1 == (Real)1 ? GetSigBits<Real,bits-1>::value() : bits);
    }
  };
  template <class Real> struct GetSigBits<Real,0> {
    static constexpr Integer value() {
      return 0;
    }
  };

  template <> class TypeTraits<float> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(float);
      static constexpr Integer SigBits = 23;
  };
  template <> class TypeTraits<double> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(double);
      static constexpr Integer SigBits = 52;
  };
  template <> class TypeTraits<long double> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(long double);
      static constexpr Integer SigBits = GetSigBits<long double>::value();
  };
#ifdef SCTL_QUAD_T
  template <> class TypeTraits<QuadReal> {
    public:
      static constexpr DataType Type = DataType::Real;
      static constexpr Integer Size = sizeof(QuadReal);
      static constexpr Integer SigBits = 112;
  };
#endif

  template <Integer N> struct IntegerType {};
  template <> struct IntegerType<sizeof( int8_t)> { using value =  int8_t; };
  template <> struct IntegerType<sizeof(int16_t)> { using value = int16_t; };
  template <> struct IntegerType<sizeof(int32_t)> { using value = int32_t; };
  template <> struct IntegerType<sizeof(int64_t)> { using value = int64_t; };
#ifdef __SIZEOF_INT128__
  template <> struct IntegerType<sizeof(__int128)> { using value = __int128; };
#endif
}

namespace SCTL_NAMESPACE { // Generic

  template <class ValueType, Integer N> struct alignas(sizeof(ValueType)*N>SCTL_ALIGN_BYTES?SCTL_ALIGN_BYTES:sizeof(ValueType)*N) VecData {
    using ScalarType = ValueType;
    static constexpr Integer Size = N;
    ScalarType v[N];
  };


  template <class VData> inline VData zero_intrin() {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } a_;
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = (typename VData::ScalarType)0;
    return a_.v;
  }
  template <class VData> inline VData set1_intrin(typename VData::ScalarType a) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } a_;
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = a;
    return a_.v;
  }

  template <Integer k, Integer Size, class Data, class T> inline void SetHelper(Data& vec, T x) {
    vec.x[Size-k-1] = x;
  }
  template <Integer k, Integer Size, class Data, class T, class... T2> inline void SetHelper(Data& vec, T x, T2... rest) {
    vec.x[Size-k-1] = x;
    SetHelper<k-1,Size>(vec, rest...);
  }
  template <class VData, class T, class ...T2> inline VData set_intrin(T x, T2 ...args) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec;
    SetHelper<VData::Size-1,VData::Size>(vec, x, args...);
    return vec.v;
  }

  template <class VData> inline VData load1_intrin(typename VData::ScalarType const* p) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec;
    for (Integer i = 0; i < VData::Size; i++) vec.x[i] = p[0];
    return vec.v;
  }
  template <class VData> inline VData loadu_intrin(typename VData::ScalarType const* p) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec;
    for (Integer i = 0; i < VData::Size; i++) vec.x[i] = p[i];
    return vec.v;
  }
  template <class VData> inline VData load_intrin(typename VData::ScalarType const* p) {
    return loadu_intrin<VData>(p);
  }

  template <class VData> inline void storeu_intrin(typename VData::ScalarType* p, VData vec) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    for (Integer i = 0; i < VData::Size; i++) p[i] = vec_.x[i];
  }
  template <class VData> inline void store_intrin(typename VData::ScalarType* p, VData vec) {
    storeu_intrin(p,vec);
  }

  template <class VData> inline typename VData::ScalarType extract_intrin(VData vec, Integer i) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    return vec_.x[i];
  }
  template <class VData> inline void insert_intrin(VData& vec, Integer i, typename VData::ScalarType value) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    vec_.x[i] = value;
    vec = vec_.v;
  }

  // Arithmetic operators
  template <class VData> inline VData unary_minus_intrin(const VData& vec) {
    union {
      VData v;
      typename VData::ScalarType x[VData::Size];
    } vec_ = {vec};
    for (Integer i = 0; i < VData::Size; i++) vec_.x[i] = -vec_.x[i];
    return vec_.v;
  }
  template <class VData> inline VData mul_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] *= b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData div_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] /= b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData add_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] += b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData sub_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] -= b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData fma_intrin(const VData& a, const VData& b, const VData& c) {
    return add_intrin(mul_intrin(a,b), c);
  }

  // Bitwise operators
  template <class VData> inline VData not_intrin(const VData& vec) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union {
      VData v;
      int8_t x[N];
    } vec_ = {vec};
    for (Integer i = 0; i < N; i++) vec_.x[i] = ~vec_.x[i];
    return vec_.v;
  }
  template <class VData> inline VData and_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] & b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData xor_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] ^ b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData or_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] | b_.x[i];
    return a_.v;
  }
  template <class VData> inline VData andnot_intrin(const VData& a, const VData& b) {
    static constexpr Integer N = VData::Size*sizeof(typename VData::ScalarType);
    union U {
      VData v;
      int8_t x[N];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] & (~b_.x[i]);
    return a_.v;
  }

  // Bitshift
  template <class VData> inline VData bitshiftleft_intrin(const VData& a, const Integer& rhs) {
    static constexpr Integer N = VData::Size;
    union {
      VData v;
      typename VData::ScalarType x[N];
    } a_ = {a};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] << rhs;
    return a_.v;
  }
  template <class VData> inline VData bitshiftright_intrin(const VData& a, const Integer& rhs) {
    static constexpr Integer N = VData::Size;
    union {
      VData v;
      typename VData::ScalarType x[N];
    } a_ = {a};
    for (Integer i = 0; i < N; i++) a_.x[i] = a_.x[i] >> rhs;
    return a_.v;
  }

  // Other functions
  template <class VData> inline VData max_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = (a_.x[i] < b_.x[i] ? b_.x[i] : a_.x[i]);
    return a_.v;
  }
  template <class VData> inline VData min_intrin(const VData& a, const VData& b) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] = (a_.x[i] < b_.x[i] ? a_.x[i] : b_.x[i]);
    return a_.v;
  }

  // Conversion operators
  template <class RetType, class ValueType, Integer N> RetType reinterpret_intrin(const VecData<ValueType,N>& v){
    static_assert(sizeof(RetType) == sizeof(VecData<ValueType,N>), "Illegal type cast -- size of types does not match.");
    union {
      VecData<ValueType,N> v;
      RetType r;
    } u = {v};
    return u.r;
  }
  template <class RealVec, class IntVec> RealVec convert_int2real_intrin(const IntVec& x) {
    using Real = typename RealVec::ScalarType;
    using Int = typename IntVec::ScalarType;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");
    static_assert(TypeTraits<Int>::Type == DataType::Integer, "Expected integer type!");
    static_assert(sizeof(RealVec) == sizeof(IntVec) && sizeof(Real) == sizeof(Int), "Real and integer types must have same size!");

    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    union {
      Int Cint = (((Int)1) << (SigBits - 1)) + ((SigBits + ((((Int)1)<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    IntVec l(add_intrin(x, set1_intrin<IntVec>(Cint)));
    return sub_intrin(reinterpret_intrin<RealVec>(l), set1_intrin<RealVec>(Creal));
  }
  template <class IntVec, class RealVec> IntVec round_real2int_intrin(const RealVec& x) {
    using Int = typename IntVec::ScalarType;
    using Real = typename RealVec::ScalarType;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");
    static_assert(TypeTraits<Int>::Type == DataType::Integer, "Expected integer type!");
    static_assert(sizeof(RealVec) == sizeof(IntVec) && sizeof(Real) == sizeof(Int), "Real and integer types must have same size!");

    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    union {
      Int Cint = (((Int)1) << (SigBits - 1)) + ((SigBits + ((((Int)1)<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    RealVec d(add_intrin(x, set1_intrin<RealVec>(Creal)));
    return sub_intrin(reinterpret_intrin<IntVec>(d), set1_intrin<IntVec>(Cint));
  }
  template <class VData> VData round_real2real_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");

    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    union {
      Int Cint = (((Int)1) << (SigBits - 1)) + ((SigBits + ((((Int)1)<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    VData Vreal(set1_intrin<VData>(Creal));
    return sub_intrin(add_intrin(x, Vreal), Vreal);
  }


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  template <class VData> struct Mask : public VData {
    using VDataType = VData;

    static Mask Zero() {
      return Mask<VData>(zero_intrin<VDataType>());
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    explicit Mask(const VData& v) : VData(v) {}
  };

  template <class RetType, class VData> RetType reinterpret_mask(const Mask<VData>& v){
    static_assert(sizeof(RetType) == sizeof(Mask<VData>), "Illegal type cast -- size of types does not match.");
    union {
      Mask<VData> v;
      RetType r;
    } u = {v};
    return u.r;
  }

  // Bitwise operators
  template <class VData> inline Mask<VData> operator~(const Mask<VData>& vec) {
    return Mask<VData>(not_intrin(vec));
  }
  template <class VData> inline Mask<VData> operator&(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(and_intrin(a,b));
  }
  template <class VData> inline Mask<VData> operator^(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(xor_intrin(a,b));
  }
  template <class VData> inline Mask<VData> operator|(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(or_intrin(a,b));
  }
  template <class VData> inline Mask<VData> AndNot(const Mask<VData>& a, const Mask<VData>& b) {
    return Mask<VData>(andnot_intrin(a,b));
  }

  template <class VData> inline VData convert_mask2vec_intrin(const Mask<VData>& v) {
    return v;
  }
  template <class VData> inline Mask<VData> convert_vec2mask_intrin(const VData& v) {
    return Mask<VData>(v);
  }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  enum class ComparisonType { lt, le, gt, ge, eq, ne};
  template <ComparisonType TYPE, class VData> Mask<VData> comp_intrin(const VData& a, const VData& b) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "Invalid operation on Mask");
    using ScalarType = typename VData::ScalarType;
    using IntType = typename IntegerType<sizeof(ScalarType)>::value;

    union U {
      VData v;
      Mask<VData> m;
      ScalarType x[VData::Size];
      IntType q[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    U c_;

    static constexpr IntType zero_const = (IntType)0;
    static constexpr IntType  one_const =~(IntType)0;
    if (TYPE == ComparisonType::lt) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] <  b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::le) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] <= b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::gt) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] >  b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::ge) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] >= b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::eq) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] == b_.x[i] ? one_const : zero_const);
    if (TYPE == ComparisonType::ne) for (Integer i = 0; i < VData::Size; i++) c_.q[i] = (a_.x[i] != b_.x[i] ? one_const : zero_const);
    return c_.m;
  }

  template <class VData> VData select_intrin(const Mask<VData>& s, const VData& a, const VData& b) {
    static_assert(sizeof(Mask<VData>) == sizeof(VData), "Invalid operation on Mask");
    union U {
      Mask<VData> m;
      VData v;
    } s_ = {s};
    return or_intrin(and_intrin(a,s_.v), andnot_intrin(b,s_.v));
  }

  // Special funtions
  template <Integer MAX_ITER, Integer ITER, class VData> struct rsqrt_newton_iter {
    static VData eval(const VData& y, const VData& x) {
      using ValueType = typename VData::ScalarType;
      constexpr ValueType c1 = 3 * pow<pow<MAX_ITER-ITER>(3)-1,ValueType>(2);
      return rsqrt_newton_iter<MAX_ITER,ITER-1,VData>::eval(mul_intrin(y, sub_intrin(set1_intrin<VData>(c1), mul_intrin(x,mul_intrin(y,y)))), x);
    }
  };
  template <Integer MAX_ITER, class VData> struct rsqrt_newton_iter<MAX_ITER,1,VData> {
    static VData eval(const VData& y, const VData& x) {
      using ValueType = typename VData::ScalarType;
      constexpr ValueType c1 = 3 * pow<pow<MAX_ITER-1>(3)-1,ValueType>(2);
      constexpr ValueType c2 = pow<(pow<MAX_ITER-1>(3)-1)*3/2+1,ValueType>(0.5);
      return mul_intrin(mul_intrin(y, sub_intrin(set1_intrin<VData>(c1), mul_intrin(x,mul_intrin(y,y)))), set1_intrin<VData>(c2));
    }
  };
  template <class VData> struct rsqrt_newton_iter<0,0,VData> {
    static VData eval(const VData& y, const VData& x) {
      return y;
    }
  };
  static constexpr Integer mylog2(Integer x) {
    return ((x<1) ? 0 : 1+mylog2(x/2));
  }
  template <Integer digits, class VData> struct rsqrt_approx_intrin {
    static inline VData eval(const VData& a) {
      union {
        VData v;
        typename VData::ScalarType x[VData::Size];
      } a_ = {a}, b_;
      for (Integer i = 0; i < VData::Size; i++) b_.x[i] = (typename VData::ScalarType)1/sqrt<float>((float)a_.x[i]);

      constexpr Integer newton_iter = mylog2((Integer)(digits/7.2247198959));
      return rsqrt_newton_iter<newton_iter,newton_iter,VData>::eval(b_.v, a);
    }
    static inline VData eval(const VData& a, const Mask<VData>& m) {
      return and_intrin(rsqrt_approx_intrin<digits,VData>::eval(a), convert_mask2vec_intrin(m));
    }
  };

  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0) {
    return set1_intrin<VData>(c0);
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1) {
    return fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2) {
    VData x2(mul_intrin<VData>(x1,x1));
    return fma_intrin(x2, set1_intrin<VData>(c2), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3) {
    VData x2(mul_intrin<VData>(x1,x1));
    return fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, set1_intrin<VData>(c4), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4)), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, fma_intrin(x2, set1_intrin<VData>(c6), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    return fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, set1_intrin<VData>(c8), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x2, set1_intrin<VData>(c10), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, set1_intrin<VData>(c12) , fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, fma_intrin(x1, set1_intrin<VData>(c13), set1_intrin<VData>(c12)) , fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13, const CType& c14) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, fma_intrin(x2, set1_intrin<VData>(c14), fma_intrin(x1, set1_intrin<VData>(c13), set1_intrin<VData>(c12))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }
  template <class VData, class CType> VData EvalPolynomial(const VData& x1, const CType& c0, const CType& c1, const CType& c2, const CType& c3, const CType& c4, const CType& c5, const CType& c6, const CType& c7, const CType& c8, const CType& c9, const CType& c10, const CType& c11, const CType& c12, const CType& c13, const CType& c14, const CType& c15) {
    VData x2(mul_intrin<VData>(x1,x1));
    VData x4(mul_intrin<VData>(x2,x2));
    VData x8(mul_intrin<VData>(x4,x4));
    return fma_intrin(x8, fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c15), set1_intrin<VData>(c14)), fma_intrin(x1, set1_intrin<VData>(c13), set1_intrin<VData>(c12))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c11), set1_intrin<VData>(c10)), fma_intrin(x1, set1_intrin<VData>(c9), set1_intrin<VData>(c8)))), fma_intrin(x4, fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c7), set1_intrin<VData>(c6)), fma_intrin(x1, set1_intrin<VData>(c5), set1_intrin<VData>(c4))), fma_intrin(x2, fma_intrin(x1, set1_intrin<VData>(c3), set1_intrin<VData>(c2)), fma_intrin(x1, set1_intrin<VData>(c1), set1_intrin<VData>(c0)))));
  }

  template <Integer ORDER, class VData> void approx_sincos_intrin(VData& sinx, VData& cosx, const VData& x) {
    // ORDER    ERROR
    //     1 8.81e-02
    //     3 2.45e-03
    //     5 3.63e-05
    //     7 3.11e-07
    //     9 1.75e-09
    //    11 6.93e-12
    //    13 2.09e-14
    //    15 6.66e-16
    //    17 6.66e-16

    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    using IntVec = VecData<Int, VData::Size>;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");

    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Real coeff1  =  1;
    static constexpr Real coeff3  = -1/(((Real)2)*3);
    static constexpr Real coeff5  =  1/(((Real)2)*3*4*5);
    static constexpr Real coeff7  = -1/(((Real)2)*3*4*5*6*7);
    static constexpr Real coeff9  =  1/(((Real)2)*3*4*5*6*7*8*9);
    static constexpr Real coeff11 = -1/(((Real)2)*3*4*5*6*7*8*9*10*11);
    static constexpr Real coeff13 =  1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13);
    static constexpr Real coeff15 = -1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13*14*15);
    static constexpr Real coeff17 =  1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13*14*15*16*17);
    static constexpr Real coeff19 = -1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13*14*15*16*17*18*19); // err = 2^-72.7991
    static constexpr Real pi_over_2 = const_pi<Real>()/2;
    static constexpr Real neg_pi_over_2 = -const_pi<Real>()/2;
    static constexpr Real inv_pi_over_2 = 1 / pi_over_2;

    union {
      Int Cint = 0 + (((Int)1) << (SigBits - 1)) + ((SigBits + ((((Int)1)<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    VData real_offset(set1_intrin<VData>(Creal));

    VData x_int(fma_intrin(x, set1_intrin<VData>(inv_pi_over_2), real_offset));
    VData x_(sub_intrin(x_int, real_offset)); // x_ <-- round(x*inv_pi_over_2)
    VData x1 = fma_intrin(x_, set1_intrin<VData>(neg_pi_over_2), x);

    VData s1;
    VData x2 = mul_intrin(x1,x1);
    if      (ORDER >= 19) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5, coeff7, coeff9, coeff11, coeff13, coeff15, coeff17, coeff19), x1);
    else if (ORDER >= 17) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5, coeff7, coeff9, coeff11, coeff13, coeff15, coeff17), x1);
    else if (ORDER >= 15) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5, coeff7, coeff9, coeff11, coeff13, coeff15), x1);
    else if (ORDER >= 13) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5, coeff7, coeff9, coeff11, coeff13), x1);
    else if (ORDER >= 11) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5, coeff7, coeff9, coeff11), x1);
    else if (ORDER >=  9) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5, coeff7, coeff9), x1);
    else if (ORDER >=  7) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5, coeff7), x1);
    else if (ORDER >=  5) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3, coeff5), x1);
    else if (ORDER >=  3) s1 = mul_intrin(EvalPolynomial(x2, coeff1, coeff3), x1);
    else                  s1 = mul_intrin(EvalPolynomial(x2, coeff1), x1);

    VData cos_squared = sub_intrin(set1_intrin<VData>(1), mul_intrin(s1, s1));
    VData inv_cos = rsqrt_approx_intrin<ORDER,VData>::eval(cos_squared, comp_intrin<ComparisonType::ne>(cos_squared,zero_intrin<VData>()));
    VData c1 = mul_intrin(cos_squared, inv_cos);

    IntVec vec_zero(zero_intrin<IntVec>());
    auto xAnd1 = reinterpret_mask<Mask<VData>>(comp_intrin<ComparisonType::eq>(and_intrin(reinterpret_intrin<IntVec>(x_int), set1_intrin<IntVec>(1)), vec_zero));
    auto xAnd2 = reinterpret_mask<Mask<VData>>(comp_intrin<ComparisonType::eq>(and_intrin(reinterpret_intrin<IntVec>(x_int), set1_intrin<IntVec>(2)), vec_zero));

    VData s2(select_intrin(xAnd1, s1,                    c1 ));
    VData c2(select_intrin(xAnd1, c1, unary_minus_intrin(s1)));
    sinx = select_intrin(xAnd2, s2, unary_minus_intrin(s2));
    cosx = select_intrin(xAnd2, c2, unary_minus_intrin(c2));
  }
  template <class VData> void sincos_intrin(VData& sinx, VData& cosx, const VData& x) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U sinx_, cosx_, x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) {
      sinx_.x[i] = sin(x_.x[i]);
      cosx_.x[i] = cos(x_.x[i]);
    }
    sinx = sinx_.v;
    cosx = cosx_.v;
  }

  template <Integer ORDER, class VData> VData approx_exp_intrin(const VData& x) {
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    using IntVec = VecData<Int, VData::Size>;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");

    static constexpr Int SigBits = TypeTraits<Real>::SigBits;
    static constexpr Real coeff2  = 1/(((Real)2));
    static constexpr Real coeff3  = 1/(((Real)2)*3);
    static constexpr Real coeff4  = 1/(((Real)2)*3*4);
    static constexpr Real coeff5  = 1/(((Real)2)*3*4*5);
    static constexpr Real coeff6  = 1/(((Real)2)*3*4*5*6);
    static constexpr Real coeff7  = 1/(((Real)2)*3*4*5*6*7);
    static constexpr Real coeff8  = 1/(((Real)2)*3*4*5*6*7*8);
    static constexpr Real coeff9  = 1/(((Real)2)*3*4*5*6*7*8*9);
    static constexpr Real coeff10 = 1/(((Real)2)*3*4*5*6*7*8*9*10);
    static constexpr Real coeff11 = 1/(((Real)2)*3*4*5*6*7*8*9*10*11);
    static constexpr Real coeff12 = 1/(((Real)2)*3*4*5*6*7*8*9*10*11*12);
    static constexpr Real coeff13 = 1/(((Real)2)*3*4*5*6*7*8*9*10*11*12*13); // err = 2^-57.2759
    static constexpr Real x0 = -(Real)0.693147180559945309417232121458l; // -ln(2)
    static constexpr Real invx0 = -1 / x0; // 1/ln(2)

    VData x_(round_real2real_intrin(mul_intrin(x, set1_intrin<VData>(invx0))));
    IntVec int_x_ = round_real2int_intrin<IntVec>(x_);
    VData x1 = fma_intrin(x_, set1_intrin<VData>(x0), x);

    VData e1;
    if      (ORDER >= 13) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10, coeff11, coeff12, coeff13);
    else if (ORDER >= 12) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10, coeff11, coeff12);
    else if (ORDER >= 11) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10, coeff11);
    else if (ORDER >= 10) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10);
    else if (ORDER >=  9) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9);
    else if (ORDER >=  8) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8);
    else if (ORDER >=  7) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7);
    else if (ORDER >=  6) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5, coeff6);
    else if (ORDER >=  5) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4, coeff5);
    else if (ORDER >=  4) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3, coeff4);
    else if (ORDER >=  3) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2, coeff3);
    else if (ORDER >=  2) e1 = EvalPolynomial(x1, (Real)1, (Real)1, coeff2);
    else if (ORDER >=  1) e1 = EvalPolynomial(x1, (Real)1, (Real)1);
    else if (ORDER >=  0) e1 = set1_intrin<VData>(1);

    VData e2;
    { // set e2 = 2 ^ x_
      union {
        Real real_one = 1.0;
        Int int_one;
      };
      IntVec int_e2 = add_intrin(set1_intrin<IntVec>(int_one), bitshiftleft_intrin(int_x_, SigBits));

      // Handle underflow
      static constexpr Int max_exp = -(Int)(((Int)1)<<((sizeof(Real)*8-SigBits-2)));
      e2 = reinterpret_intrin<VData>(select_intrin(comp_intrin<ComparisonType::gt>(int_x_, set1_intrin<IntVec>(max_exp)) , int_e2, zero_intrin<IntVec>()));
    }

    return mul_intrin(e1, e2);
  }
  template <class VData> VData exp_intrin(const VData& x) {
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U expx_, x_ = {x};
    for (Integer i = 0; i < VData::Size; i++) expx_.x[i] = exp(x_.x[i]);
    return expx_.v;
  }

}

namespace SCTL_NAMESPACE { // SSE
#ifdef __SSE4_2__
  template <> struct alignas(sizeof(int8_t) * 16) VecData<int8_t,16> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 16;
    VecData() = default;
    VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(int16_t) * 8) VecData<int16_t,8> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 8;
    VecData() = default;
    VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(int32_t) * 4) VecData<int32_t,4> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 4;
    VecData() = default;
    VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(int64_t) * 2) VecData<int64_t,2> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 2;
    VecData() = default;
    VecData(__m128i v_) : v(v_) {}
    __m128i v;
  };
  template <> struct alignas(sizeof(float) * 4) VecData<float,4> {
    using ScalarType = float;
    static constexpr Integer Size = 4;
    VecData() = default;
    VecData(__m128 v_) : v(v_) {}
    __m128 v;
  };
  template <> struct alignas(sizeof(double) * 2) VecData<double,2> {
    using ScalarType = double;
    static constexpr Integer Size = 2;
    VecData() = default;
    VecData(__m128d v_) : v(v_) {}
    __m128d v;
  };

  // Select between two sources, byte by byte. Used in various functions and operators
  // Corresponds to this pseudocode:
  // for (int i = 0; i < 16; i++) result[i] = s[i] ? a[i] : b[i];
  // Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
  // The implementation depends on the instruction set:
  // If SSE4.1 is supported then only bit 7 in each byte of s is checked,
  // otherwise all bits in s are used.
  static inline __m128i selectb (__m128i const & s, __m128i const & a, __m128i const & b) {
    #if defined(__SSE4_1__)
    return _mm_blendv_epi8 (b, a, s);
    #else
    return _mm_or_si128(_mm_and_si128(s,a), _mm_andnot_si128(s,b));
    #endif
  }


  template <> inline VecData<int8_t,16> zero_intrin<VecData<int8_t,16>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<int16_t,8> zero_intrin<VecData<int16_t,8>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<int32_t,4> zero_intrin<VecData<int32_t,4>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<int64_t,2> zero_intrin<VecData<int64_t,2>>() {
    return _mm_setzero_si128();
  }
  template <> inline VecData<float,4> zero_intrin<VecData<float,4>>() {
    return _mm_setzero_ps();
  }
  template <> inline VecData<double,2> zero_intrin<VecData<double,2>>() {
    return _mm_setzero_pd();
  }

  template <> inline VecData<int8_t,16> set1_intrin<VecData<int8_t,16>>(int8_t a) {
    return _mm_set1_epi8(a);
  }
  template <> inline VecData<int16_t,8> set1_intrin<VecData<int16_t,8>>(int16_t a) {
    return _mm_set1_epi16(a);
  }
  template <> inline VecData<int32_t,4> set1_intrin<VecData<int32_t,4>>(int32_t a) {
    return _mm_set1_epi32(a);
  }
  template <> inline VecData<int64_t,2> set1_intrin<VecData<int64_t,2>>(int64_t a) {
    return _mm_set1_epi64x(a);
  }
  template <> inline VecData<float,4> set1_intrin<VecData<float,4>>(float a) {
    return _mm_set1_ps(a);
  }
  template <> inline VecData<double,2> set1_intrin<VecData<double,2>>(double a) {
    return _mm_set1_pd(a);
  }

  template <> inline VecData<int8_t,16> set_intrin<VecData<int8_t,16>,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t>(int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16) {
    return _mm_set_epi8(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int16_t,8> set_intrin<VecData<int16_t,8>,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t>(int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7, int16_t v8) {
    return _mm_set_epi16(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int32_t,4> set_intrin<VecData<int32_t,4>,int32_t,int32_t,int32_t,int32_t>(int32_t v1, int32_t v2, int32_t v3, int32_t v4) {
    return _mm_set_epi32(v4,v3,v2,v1);
  }
  template <> inline VecData<int64_t,2> set_intrin<VecData<int64_t,2>,int64_t,int64_t>(int64_t v1, int64_t v2) {
    return _mm_set_epi64x(v2,v1);
  }
  template <> inline VecData<float,4> set_intrin<VecData<float,4>,float,float,float,float>(float v1, float v2, float v3, float v4) {
    return _mm_set_ps(v4,v3,v2,v1);
  }
  template <> inline VecData<double,2> set_intrin<VecData<double,2>,double,double>(double v1, double v2) {
    return _mm_set_pd(v2,v1);
  }

  template <> inline VecData<int8_t,16> load1_intrin<VecData<int8_t,16>>(int8_t const* p) {
    return _mm_set1_epi8(p[0]);
  }
  template <> inline VecData<int16_t,8> load1_intrin<VecData<int16_t,8>>(int16_t const* p) {
    return _mm_set1_epi16(p[0]);
  }
  template <> inline VecData<int32_t,4> load1_intrin<VecData<int32_t,4>>(int32_t const* p) {
    return _mm_set1_epi32(p[0]);
  }
  template <> inline VecData<int64_t,2> load1_intrin<VecData<int64_t,2>>(int64_t const* p) {
    return _mm_set1_epi64x(p[0]);
  }
  template <> inline VecData<float,4> load1_intrin<VecData<float,4>>(float const* p) {
    return _mm_load1_ps(p);
  }
  template <> inline VecData<double,2> load1_intrin<VecData<double,2>>(double const* p) {
    return _mm_load1_pd(p);
  }

  template <> inline VecData<int8_t,16> loadu_intrin<VecData<int8_t,16>>(int8_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<int16_t,8> loadu_intrin<VecData<int16_t,8>>(int16_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<int32_t,4> loadu_intrin<VecData<int32_t,4>>(int32_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<int64_t,2> loadu_intrin<VecData<int64_t,2>>(int64_t const* p) {
    return _mm_loadu_si128((__m128i const*)p);
  }
  template <> inline VecData<float,4> loadu_intrin<VecData<float,4>>(float const* p) {
    return _mm_loadu_ps(p);
  }
  template <> inline VecData<double,2> loadu_intrin<VecData<double,2>>(double const* p) {
    return _mm_loadu_pd(p);
  }

  template <> inline VecData<int8_t,16> load_intrin<VecData<int8_t,16>>(int8_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<int16_t,8> load_intrin<VecData<int16_t,8>>(int16_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<int32_t,4> load_intrin<VecData<int32_t,4>>(int32_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<int64_t,2> load_intrin<VecData<int64_t,2>>(int64_t const* p) {
    return _mm_load_si128((__m128i const*)p);
  }
  template <> inline VecData<float,4> load_intrin<VecData<float,4>>(float const* p) {
    return _mm_load_ps(p);
  }
  template <> inline VecData<double,2> load_intrin<VecData<double,2>>(double const* p) {
    return _mm_load_pd(p);
  }

  template <> inline void storeu_intrin<VecData<int8_t,16>>(int8_t* p, VecData<int8_t,16> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int16_t,8>>(int16_t* p, VecData<int16_t,8> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<float,4>>(float* p, VecData<float,4> vec) {
    _mm_storeu_ps(p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<double,2>>(double* p, VecData<double,2> vec) {
    _mm_storeu_pd(p, vec.v);
  }

  template <> inline void store_intrin<VecData<int8_t,16>>(int8_t* p, VecData<int8_t,16> vec) {
    _mm_storeu_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int16_t,8>>(int16_t* p, VecData<int16_t,8> vec) {
    _mm_store_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int32_t,4>>(int32_t* p, VecData<int32_t,4> vec) {
    _mm_store_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int64_t,2>>(int64_t* p, VecData<int64_t,2> vec) {
    _mm_store_si128((__m128i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<float,4>>(float* p, VecData<float,4> vec) {
    _mm_store_ps(p, vec.v);
  }
  template <> inline void store_intrin<VecData<double,2>>(double* p, VecData<double,2> vec) {
    _mm_store_pd(p, vec.v);
  }

#if defined(__AVX512VBMI2__)
  template <> inline int8_t extract_intrin<VecData<int8_t,16>>(VecData<int8_t,16> vec, Integer i) {
    __m128i x = _mm_maskz_compress_epi8(__mmask16(1u<<i), vec.v);
    return (int8_t)_mm_cvtsi128_si32(x);
  }
  template <> inline int16_t extract_intrin<VecData<int16_t,8>>(VecData<int16_t,8> vec, Integer i) {
    __m128i x = _mm_maskz_compress_epi16(__mmask8(1u<<i), vec.v);
    return (int16_t)_mm_cvtsi128_si32(x);
  }
  template <> inline int32_t extract_intrin<VecData<int32_t,4>>(VecData<int32_t,4> vec, Integer i) {
    __m128i x = _mm_maskz_compress_epi32(__mmask8(1u<<i), vec.v);
    return (int32_t)_mm_cvtsi128_si32(x);
  }
  //template <> inline int64_t extract_intrin<VecData<int64_t,2>>(VecData<int64_t,2> vec, Integer i) {}
  template <> inline float extract_intrin<VecData<float,4>>(VecData<float,4> vec, Integer i) {
    __m128 x = _mm_maskz_compress_ps(__mmask8(1u<<i), vec.v);
    return _mm_cvtss_f32(x);
  }
  template <> inline double extract_intrin<VecData<double,2>>(VecData<double,2> vec, Integer i) {
    __m128d x = _mm_mask_unpackhi_pd(vec.v, __mmask8(i), vec.v, vec.v);
    return _mm_cvtsd_f64(x);
  }
#endif

#if defined(__AVX512BW__)
  template <> inline void insert_intrin<VecData<int8_t,16>>(VecData<int8_t,16>& vec, Integer i, int8_t value) {
    vec.v = _mm_mask_set1_epi8(vec.v, __mmask16(1u<<i), value);
  }
  template <> inline void insert_intrin<VecData<int16_t,8>>(VecData<int16_t,8>& vec, Integer i, int16_t value) {
    vec.v = _mm_mask_set1_epi16(vec.v, __mmask8(1u<<i), value);
  }
  template <> inline void insert_intrin<VecData<int32_t,4>>(VecData<int32_t,4>& vec, Integer i, int32_t value) {
    vec.v = _mm_mask_set1_epi32(vec.v, __mmask8(1u<<i), value);
  }
  template <> inline void insert_intrin<VecData<int64_t,2>>(VecData<int64_t,2>& vec, Integer i, int64_t value) {
    vec.v = _mm_mask_set1_epi64(vec.v, __mmask8(1u<<i), value);
  }
  template <> inline void insert_intrin<VecData<float,4>>(VecData<float,4>& vec, Integer i, float value) {
    vec.v = _mm_mask_broadcastss_ps(vec.v, __mmask8(1u<<i), _mm_set_ss(value));
  }
  template <> inline void insert_intrin<VecData<double,2>>(VecData<double,2>& vec, Integer i, double value) {
    vec.v = _mm_mask_movedup_pd(vec.v, __mmask8(1u<<i), _mm_set_sd(value));
  }
#endif

  // Arithmetic operators
  template <> inline VecData<int8_t,16> unary_minus_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a) {
    return _mm_sub_epi8(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<int16_t,8> unary_minus_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a) {
    return _mm_sub_epi16(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<int32_t,4> unary_minus_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a) {
    return _mm_sub_epi32(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<int64_t,2> unary_minus_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a) {
    return _mm_sub_epi64(_mm_setzero_si128(), a.v);
  }
  template <> inline VecData<float,4> unary_minus_intrin<VecData<float,4>>(const VecData<float,4>& a) {
    return _mm_xor_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
  }
  template <> inline VecData<double,2> unary_minus_intrin<VecData<double,2>>(const VecData<double,2>& a) {
    return _mm_xor_pd(a.v, _mm_castsi128_pd(_mm_setr_epi32(0,0x80000000,0,0x80000000)));
  }

  template <> inline VecData<int8_t,16> mul_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    // There is no 8-bit multiply in SSE2. Split into two 16-bit multiplies
    __m128i aodd    = _mm_srli_epi16(a.v,8);               // odd numbered elements of a
    __m128i bodd    = _mm_srli_epi16(b.v,8);               // odd numbered elements of b
    __m128i muleven = _mm_mullo_epi16(a.v,b.v);            // product of even numbered elements
    __m128i mulodd  = _mm_mullo_epi16(aodd,bodd);          // product of odd  numbered elements
            mulodd  = _mm_slli_epi16(mulodd,8);            // put odd numbered elements back in place
    #if defined(__AVX512VL__) && defined(__AVX512BW__)
    return _mm_mask_mov_epi8(mulodd, 0x5555, muleven);
    #else
    __m128i mask    = _mm_set1_epi32(0x00FF00FF);          // mask for even positions
    return selectb(mask,muleven,mulodd);                   // interleave even and odd
    #endif
  }
  template <> inline VecData<int16_t,8> mul_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_mullo_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> mul_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    #if defined(__SSE4_1__)
    return _mm_mullo_epi32(a.v, b.v);
    #else
    __m128i a13    = _mm_shuffle_epi32(a.v, 0xF5);        // (-,a3,-,a1)
    __m128i b13    = _mm_shuffle_epi32(b.v, 0xF5);        // (-,b3,-,b1)
    __m128i prod02 = _mm_mul_epu32(a.v, b.v);             // (-,a2*b2,-,a0*b0)
    __m128i prod13 = _mm_mul_epu32(a13, b13);             // (-,a3*b3,-,a1*b1)
    __m128i prod01 = _mm_unpacklo_epi32(prod02,prod13);   // (-,-,a1*b1,a0*b0)
    __m128i prod23 = _mm_unpackhi_epi32(prod02,prod13);   // (-,-,a3*b3,a2*b2)
    return           _mm_unpacklo_epi64(prod01,prod23);   // (ab3,ab2,ab1,ab0)
    #endif
  }
  template <> inline VecData<int64_t,2> mul_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    #if defined(__AVX512DQ__) && defined(__AVX512VL__)
    return _mm_mullo_epi64(a.v, b.v);
    #elif defined(__SSE4_1__)
    // Split into 32-bit multiplies
    __m128i bswap   = _mm_shuffle_epi32(b.v,0xB1);         // b0H,b0L,b1H,b1L (swap H<->L)
    __m128i prodlh  = _mm_mullo_epi32(a.v,bswap);          // a0Lb0H,a0Hb0L,a1Lb1H,a1Hb1L, 32 bit L*H products
    __m128i zero    = _mm_setzero_si128();                 // 0
    __m128i prodlh2 = _mm_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m128i prodlh3 = _mm_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m128i prodll  = _mm_mul_epu32(a.v,b.v);              // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m128i prod    = _mm_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return  prod;
    #else               // SSE2
    union U {
      VData v;
      typename VData::ScalarType x[VData::Size];
    };
    U a_ = {a};
    U b_ = {b};
    for (Integer i = 0; i < VData::Size; i++) a_.x[i] *= b_.x[i];
    return a_.v;
    #endif
  }
  template <> inline VecData<float,4> mul_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_mul_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> mul_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_mul_pd(a.v, b.v);
  }

  template <> inline VecData<float,4> div_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_div_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> div_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_div_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> add_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_add_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> add_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_add_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> add_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_add_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> add_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_add_epi64(a.v, b.v);
  }
  template <> inline VecData<float,4> add_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_add_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> add_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_add_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> sub_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_sub_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> sub_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_sub_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> sub_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_sub_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> sub_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_sub_epi64(a.v, b.v);
  }
  template <> inline VecData<float,4> sub_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_sub_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> sub_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_sub_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,16> fma_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b, const VecData<int8_t,16>& c) {}
  //template <> inline VecData<int16_t,8> fma_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b, const VecData<int16_t,8>& c) {}
  //template <> inline VecData<int32_t,4> sub_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b, const VecData<int32_t,4>& c) {}
  //template <> inline VecData<int64_t,2> sub_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b, const VecData<int64_t,2>& c) {}
  template <> inline VecData<float,4> fma_intrin(const VecData<float,4>& a, const VecData<float,4>& b, const VecData<float,4>& c) {
    #ifdef __FMA__
    return _mm_fmadd_ps(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm_macc_ps(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }
  template <> inline VecData<double,2> fma_intrin(const VecData<double,2>& a, const VecData<double,2>& b, const VecData<double,2>& c) {
    #ifdef __FMA__
    return _mm_fmadd_pd(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm_macc_pd(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }

  // Bitwise operators
  template <> inline VecData<int8_t,16> not_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<int16_t,8> not_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<int32_t,4> not_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<int64_t,2> not_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a) {
    return _mm_xor_si128(a.v, _mm_set1_epi32(-1));
  }
  template <> inline VecData<float,4> not_intrin<VecData<float,4>>(const VecData<float,4>& a) {
    return _mm_xor_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(-1)));
  }
  template <> inline VecData<double,2> not_intrin<VecData<double,2>>(const VecData<double,2>& a) {
    return _mm_xor_pd(a.v, _mm_castsi128_pd(_mm_set1_epi32(-1)));
  }

  template <> inline VecData<int8_t,16> and_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> and_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> and_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> and_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_and_si128(a.v, b.v);
  }
  template <> inline VecData<float,4> and_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_and_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> and_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_and_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> xor_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> xor_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> xor_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> xor_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_xor_si128(a.v, b.v);
  }
  template <> inline VecData<float,4> xor_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_xor_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> xor_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_xor_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> or_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> or_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> or_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<int64_t,2> or_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_or_si128(a.v, b.v);
  }
  template <> inline VecData<float,4> or_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_or_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> or_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_or_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> andnot_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<int16_t,8> andnot_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<int32_t,4> andnot_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<int64_t,2> andnot_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_andnot_si128(b.v, a.v);
  }
  template <> inline VecData<float,4> andnot_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_andnot_ps(b.v, a.v);
  }
  template <> inline VecData<double,2> andnot_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_andnot_pd(b.v, a.v);
  }

  // Bitshift
  //template <> inline VecData<int8_t,16> bitshiftleft_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a, const Integer& rhs) { }
  template <> inline VecData<int16_t,8> bitshiftleft_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a, const Integer& rhs) { return _mm_slli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,4> bitshiftleft_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a, const Integer& rhs) { return _mm_slli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t,2> bitshiftleft_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a, const Integer& rhs) { return _mm_slli_epi64(a.v , rhs); }
  template <> inline VecData<float  ,4> bitshiftleft_intrin<VecData<float  ,4>>(const VecData<float  ,4>& a, const Integer& rhs) { return _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(a.v), rhs)); }
  template <> inline VecData<double ,2> bitshiftleft_intrin<VecData<double ,2>>(const VecData<double ,2>& a, const Integer& rhs) { return _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(a.v), rhs)); }

  //template <> inline VecData<int8_t,16> bitshiftright_intrin<VecData<int8_t,16>>(const VecData<int8_t,16>& a, const Integer& rhs) { }
  template <> inline VecData<int16_t,8> bitshiftright_intrin<VecData<int16_t,8>>(const VecData<int16_t,8>& a, const Integer& rhs) { return _mm_srli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,4> bitshiftright_intrin<VecData<int32_t,4>>(const VecData<int32_t,4>& a, const Integer& rhs) { return _mm_srli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t,2> bitshiftright_intrin<VecData<int64_t,2>>(const VecData<int64_t,2>& a, const Integer& rhs) { return _mm_srli_epi64(a.v , rhs); }
  template <> inline VecData<float  ,4> bitshiftright_intrin<VecData<float  ,4>>(const VecData<float  ,4>& a, const Integer& rhs) { return _mm_castsi128_ps(_mm_srli_epi32(_mm_castps_si128(a.v), rhs)); }
  template <> inline VecData<double ,2> bitshiftright_intrin<VecData<double ,2>>(const VecData<double ,2>& a, const Integer& rhs) { return _mm_castsi128_pd(_mm_srli_epi64(_mm_castpd_si128(a.v), rhs)); }

  // Other functions
  template <> inline VecData<int8_t,16> max_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_max_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> max_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_max_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> max_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_max_epi32(a.v, b.v);
  }
  #if defined(__AVX512F__) || defined(__AVX512VL__)
  template <> inline VecData<int64_t,2> max_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_max_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,4> max_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_max_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> max_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_max_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,16> min_intrin(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) {
    return _mm_min_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,8> min_intrin(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) {
    return _mm_min_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,4> min_intrin(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) {
    return _mm_min_epi32(a.v, b.v);
  }
  #if defined(__AVX512F__) || defined(__AVX512VL__)
  template <> inline VecData<int64_t,2> min_intrin(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) {
    return _mm_min_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,4> min_intrin(const VecData<float,4>& a, const VecData<float,4>& b) {
    return _mm_min_ps(a.v, b.v);
  }
  template <> inline VecData<double,2> min_intrin(const VecData<double,2>& a, const VecData<double,2>& b) {
    return _mm_min_pd(a.v, b.v);
  }

  // Conversion operators
  template <> inline VecData<float ,4> convert_int2real_intrin<VecData<float ,4>,VecData<int32_t,4>>(const VecData<int32_t,4>& x) {
    return _mm_cvtepi32_ps(x.v);
  }
  #if defined(__AVX512F__) || defined(__AVX512VL__)
  template <> inline VecData<double,2> convert_int2real_intrin<VecData<double,2>,VecData<int64_t,2>>(const VecData<int64_t,2>& x) {
    return _mm_cvtepi64_pd(x.v);
  }
  #endif

  template <> inline VecData<int32_t,4> round_real2int_intrin<VecData<int32_t,4>,VecData<float ,4>>(const VecData<float ,4>& x) {
    return _mm_cvtps_epi32(x.v);
  }
  template <> inline VecData<int64_t,2> round_real2int_intrin<VecData<int64_t,2>,VecData<double,2>>(const VecData<double,2>& x) {
  #if defined(__AVX512F__) || defined(__AVX512VL__)
    return _mm_cvtpd_epi64(x.v);
  #else
    return _mm_cvtepi32_epi64(_mm_cvtpd_epi32(x.v));
  #endif
  }

  template <> inline VecData<float ,4> round_real2real_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_round_ps(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }
  template <> inline VecData<double,2> round_real2real_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _mm_round_pd(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::lt>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return Mask<VecData<int8_t,16>>(_mm_cmplt_epi8(a.v,b.v)); }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::le>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::gt>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return Mask<VecData<int8_t,16>>(_mm_cmpgt_epi8(a.v,b.v)); }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::ge>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::eq>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return Mask<VecData<int8_t,16>>(_mm_cmpeq_epi8(a.v,b.v)); }
  template <> inline Mask<VecData<int8_t,16>> comp_intrin<ComparisonType::ne>(const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::lt>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return Mask<VecData<int16_t,8>>(_mm_cmplt_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::le>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::gt>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return Mask<VecData<int16_t,8>>(_mm_cmpgt_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::ge>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::eq>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return Mask<VecData<int16_t,8>>(_mm_cmpeq_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,8>> comp_intrin<ComparisonType::ne>(const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::lt>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return Mask<VecData<int32_t,4>>(_mm_cmplt_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::le>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::gt>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return Mask<VecData<int32_t,4>>(_mm_cmpgt_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::ge>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::eq>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return Mask<VecData<int32_t,4>>(_mm_cmpeq_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,4>> comp_intrin<ComparisonType::ne>(const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::lt>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return Mask<VecData<int64_t,2>>(_mm_cmpgt_epi64(b.v,a.v));}
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::le>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));           }
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::gt>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return Mask<VecData<int64_t,2>>(_mm_cmpgt_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::ge>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));           }
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::eq>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return Mask<VecData<int64_t,2>>(_mm_cmpeq_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,2>> comp_intrin<ComparisonType::ne>(const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));           }

  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::lt>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmplt_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::le>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmple_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::gt>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpgt_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::ge>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpge_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::eq>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpeq_ps(a.v,b.v)); }
  template <> inline Mask<VecData<float,4>> comp_intrin<ComparisonType::ne>(const VecData<float,4>& a, const VecData<float,4>& b) { return Mask<VecData<float,4>>(_mm_cmpneq_ps(a.v,b.v));}

  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::lt>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmplt_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::le>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmple_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::gt>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpgt_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::ge>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpge_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::eq>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpeq_pd(a.v,b.v)); }
  template <> inline Mask<VecData<double,2>> comp_intrin<ComparisonType::ne>(const VecData<double,2>& a, const VecData<double,2>& b) { return Mask<VecData<double,2>>(_mm_cmpneq_pd(a.v,b.v));}

  template <> inline VecData<int8_t,16> select_intrin(const Mask<VecData<int8_t,16>>& s, const VecData<int8_t,16>& a, const VecData<int8_t,16>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int16_t,8> select_intrin(const Mask<VecData<int16_t,8>>& s, const VecData<int16_t,8>& a, const VecData<int16_t,8>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int32_t,4> select_intrin(const Mask<VecData<int32_t,4>>& s, const VecData<int32_t,4>& a, const VecData<int32_t,4>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int64_t,2> select_intrin(const Mask<VecData<int64_t,2>>& s, const VecData<int64_t,2>& a, const VecData<int64_t,2>& b) { return _mm_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<float  ,4> select_intrin(const Mask<VecData<float  ,4>>& s, const VecData<float  ,4>& a, const VecData<float  ,4>& b) { return _mm_blendv_ps  (b.v, a.v, s.v); }
  template <> inline VecData<double ,2> select_intrin(const Mask<VecData<double ,2>>& s, const VecData<double ,2>& a, const VecData<double ,2>& b) { return _mm_blendv_pd  (b.v, a.v, s.v); }


  // Special functions
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<float,4>> {
    static VecData<float,4> eval(const VecData<float,4>& a) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(_mm_maskz_rsqrt14_ps(~__mmask8(0), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(_mm_rsqrt_ps(a.v), a.v);
      #endif
    }
    static VecData<float,4> eval(const VecData<float,4>& a, const Mask<VecData<float,4>>& m) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(_mm_maskz_rsqrt14_ps(_mm_movepi32_mask(_mm_castps_si128(m.v)), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,4>>::eval(and_intrin(VecData<float,4>(_mm_rsqrt_ps(a.v)), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<double,2>> {
    static VecData<double,2> eval(const VecData<double,2>& a) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,2>>::eval(_mm_maskz_rsqrt14_pd(~__mmask8(0), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,2>>::eval(_mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(a.v))), a.v);
      #endif
    }
    static VecData<double,2> eval(const VecData<double,2>& a, const Mask<VecData<double,2>>& m) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,2>>::eval(_mm_maskz_rsqrt14_pd(_mm_movepi64_mask(_mm_castpd_si128(m.v)), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,2>>::eval(and_intrin(VecData<double,2>(_mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(a.v)))), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };

  #ifdef SCTL_HAVE_SVML
  template <> inline void sincos_intrin<VecData<float ,4>>(VecData<float ,4>& sinx, VecData<float ,4>& cosx, const VecData<float ,4>& x) { sinx = _mm_sincos_ps(&cosx.v, x.v); }
  template <> inline void sincos_intrin<VecData<double,2>>(VecData<double,2>& sinx, VecData<double,2>& cosx, const VecData<double,2>& x) { sinx = _mm_sincos_pd(&cosx.v, x.v); }

  template <> inline VecData<float ,4> exp_intrin<VecData<float ,4>>(const VecData<float ,4>& x) { return _mm_exp_ps(x.v); }
  template <> inline VecData<double,2> exp_intrin<VecData<double,2>>(const VecData<double,2>& x) { return _mm_exp_pd(x.v); }
  #else
  template <> inline void sincos_intrin<VecData<float ,4>>(VecData<float ,4>& sinx, VecData<float ,4>& cosx, const VecData<float ,4>& x) {
    approx_sincos_intrin<(Integer)(TypeTraits<float>::SigBits/3.2)>(sinx, cosx, x); // TODO: determine constants more precisely
  }
  template <> inline void sincos_intrin<VecData<double,2>>(VecData<double,2>& sinx, VecData<double,2>& cosx, const VecData<double,2>& x) {
    approx_sincos_intrin<(Integer)(TypeTraits<double>::SigBits/3.2)>(sinx, cosx, x); // TODO: determine constants more precisely
  }

  template <> inline VecData<float ,4> exp_intrin<VecData<float ,4>>(const VecData<float ,4>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<float>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  template <> inline VecData<double,2> exp_intrin<VecData<double,2>>(const VecData<double,2>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<double>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  #endif


#endif
}

namespace SCTL_NAMESPACE { // AVX
#ifdef __AVX__
  template <> struct alignas(sizeof(int8_t) * 32) VecData<int8_t,32> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 32;
    VecData() = default;
    VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(int16_t) * 16) VecData<int16_t,16> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 16;
    VecData() = default;
    VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(int32_t) * 8) VecData<int32_t,8> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 8;
    VecData() = default;
    VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(int64_t) * 4) VecData<int64_t,4> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 4;
    VecData() = default;
    VecData(__m256i v_) : v(v_) {}
    __m256i v;
  };
  template <> struct alignas(sizeof(float) * 8) VecData<float,8> {
    using ScalarType = float;
    static constexpr Integer Size = 8;
    VecData() = default;
    VecData(__m256 v_) : v(v_) {}
    __m256 v;
  };
  template <> struct alignas(sizeof(double) * 4) VecData<double,4> {
    using ScalarType = double;
    static constexpr Integer Size = 4;
    VecData() = default;
    VecData(__m256d v_) : v(v_) {}
    __m256d v;
  };

  // Select between two sources, byte by byte. Used in various functions and operators
  // Corresponds to this pseudocode:
  // for (int i = 0; i < 32; i++) result[i] = s[i] ? a[i] : b[i];
  // Each byte in s must be either 0 (false) or 0xFF (true). No other values are allowed.
  // Only bit 7 in each byte of s is checked,
  #if defined(__AVX2__)
  static inline __m256i selectb (__m256i const & s, __m256i const & a, __m256i const & b) {
    return _mm256_blendv_epi8(b, a, s);

    //union U {
    //  __m256i  v;
    //  int8_t x[32];
    //};
    //U s_ = {s};
    //U a_ = {a};
    //U b_ = {b};
    //for (Integer i = 0; i < 32; i++) {
    //  a_.x[i] = (s_.x[i] ? a_.x[i] : b_.x[i]);
    //}
    //return a_.v;
  }
  #endif


  template <> inline VecData<int8_t,32> zero_intrin<VecData<int8_t,32>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<int16_t,16> zero_intrin<VecData<int16_t,16>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<int32_t,8> zero_intrin<VecData<int32_t,8>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<int64_t,4> zero_intrin<VecData<int64_t,4>>() {
    return _mm256_setzero_si256();
  }
  template <> inline VecData<float,8> zero_intrin<VecData<float,8>>() {
    return _mm256_setzero_ps();
  }
  template <> inline VecData<double,4> zero_intrin<VecData<double,4>>() {
    return _mm256_setzero_pd();
  }

  template <> inline VecData<int8_t,32> set1_intrin<VecData<int8_t,32>>(int8_t a) {
    return _mm256_set1_epi8(a);
  }
  template <> inline VecData<int16_t,16> set1_intrin<VecData<int16_t,16>>(int16_t a) {
    return _mm256_set1_epi16(a);
  }
  template <> inline VecData<int32_t,8> set1_intrin<VecData<int32_t,8>>(int32_t a) {
    return _mm256_set1_epi32(a);
  }
  template <> inline VecData<int64_t,4> set1_intrin<VecData<int64_t,4>>(int64_t a) {
    return _mm256_set1_epi64x(a);
  }
  template <> inline VecData<float,8> set1_intrin<VecData<float,8>>(float a) {
    return _mm256_set1_ps(a);
  }
  template <> inline VecData<double,4> set1_intrin<VecData<double,4>>(double a) {
    return _mm256_set1_pd(a);
  }

  template <> inline VecData<int8_t,32> set_intrin<VecData<int8_t,32>,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t>(int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23, int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31, int8_t v32) {
    return _mm256_set_epi8(v32,v31,v30,v29,v28,v27,v26,v25,v24,v23,v22,v21,v20,v19,v18,v17,v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int16_t,16> set_intrin<VecData<int16_t,16>,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t>(int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7, int16_t v8, int16_t v9, int16_t v10, int16_t v11, int16_t v12, int16_t v13, int16_t v14, int16_t v15, int16_t v16) {
    return _mm256_set_epi16(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int32_t,8> set_intrin<VecData<int32_t,8>,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t>(int32_t v1, int32_t v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8) {
    return _mm256_set_epi32(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int64_t,4> set_intrin<VecData<int64_t,4>,int64_t,int64_t,int64_t,int64_t>(int64_t v1, int64_t v2, int64_t v3, int64_t v4) {
    return _mm256_set_epi64x(v4,v3,v2,v1);
  }
  template <> inline VecData<float,8> set_intrin<VecData<float,8>,float,float,float,float,float,float,float,float>(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8) {
    return _mm256_set_ps(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<double,4> set_intrin<VecData<double,4>,double,double,double,double>(double v1, double v2, double v3, double v4) {
    return _mm256_set_pd(v4,v3,v2,v1);
  }

  template <> inline VecData<int8_t,32> load1_intrin<VecData<int8_t,32>>(int8_t const* p) {
    return _mm256_set1_epi8(p[0]);
  }
  template <> inline VecData<int16_t,16> load1_intrin<VecData<int16_t,16>>(int16_t const* p) {
    return _mm256_set1_epi16(p[0]);
  }
  template <> inline VecData<int32_t,8> load1_intrin<VecData<int32_t,8>>(int32_t const* p) {
    return _mm256_set1_epi32(p[0]);
  }
  template <> inline VecData<int64_t,4> load1_intrin<VecData<int64_t,4>>(int64_t const* p) {
    return _mm256_set1_epi64x(p[0]);
  }
  template <> inline VecData<float,8> load1_intrin<VecData<float,8>>(float const* p) {
    return _mm256_broadcast_ss(p);
  }
  template <> inline VecData<double,4> load1_intrin<VecData<double,4>>(double const* p) {
    return _mm256_broadcast_sd(p);
  }

  template <> inline VecData<int8_t,32> loadu_intrin<VecData<int8_t,32>>(int8_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<int16_t,16> loadu_intrin<VecData<int16_t,16>>(int16_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<int32_t,8> loadu_intrin<VecData<int32_t,8>>(int32_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<int64_t,4> loadu_intrin<VecData<int64_t,4>>(int64_t const* p) {
    return _mm256_loadu_si256((__m256i const*)p);
  }
  template <> inline VecData<float,8> loadu_intrin<VecData<float,8>>(float const* p) {
    return _mm256_loadu_ps(p);
  }
  template <> inline VecData<double,4> loadu_intrin<VecData<double,4>>(double const* p) {
    return _mm256_loadu_pd(p);
  }

  template <> inline VecData<int8_t,32> load_intrin<VecData<int8_t,32>>(int8_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<int16_t,16> load_intrin<VecData<int16_t,16>>(int16_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<int32_t,8> load_intrin<VecData<int32_t,8>>(int32_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<int64_t,4> load_intrin<VecData<int64_t,4>>(int64_t const* p) {
    return _mm256_load_si256((__m256i const*)p);
  }
  template <> inline VecData<float,8> load_intrin<VecData<float,8>>(float const* p) {
    return _mm256_load_ps(p);
  }
  template <> inline VecData<double,4> load_intrin<VecData<double,4>>(double const* p) {
    return _mm256_load_pd(p);
  }

  template <> inline void storeu_intrin<VecData<int8_t,32>>(int8_t* p, VecData<int8_t,32> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int16_t,16>>(int16_t* p, VecData<int16_t,16> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec) {
    _mm256_storeu_si256((__m256i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<float,8>>(float* p, VecData<float,8> vec) {
    _mm256_storeu_ps(p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<double,4>>(double* p, VecData<double,4> vec) {
    _mm256_storeu_pd(p, vec.v);
  }

  template <> inline void store_intrin<VecData<int8_t,32>>(int8_t* p, VecData<int8_t,32> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int16_t,16>>(int16_t* p, VecData<int16_t,16> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int32_t,8>>(int32_t* p, VecData<int32_t,8> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int64_t,4>>(int64_t* p, VecData<int64_t,4> vec) {
    _mm256_store_si256((__m256i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<float,8>>(float* p, VecData<float,8> vec) {
    _mm256_store_ps(p, vec.v);
  }
  template <> inline void store_intrin<VecData<double,4>>(double* p, VecData<double,4> vec) {
    _mm256_store_pd(p, vec.v);
  }

  //template <> inline int8_t extract_intrin<VecData<int8_t,32>>(VecData<int8_t,32> vec, Integer i) {}
  //template <> inline int16_t extract_intrin<VecData<int16_t,16>>(VecData<int16_t,16> vec, Integer i) {}
  //template <> inline int32_t extract_intrin<VecData<int32_t,8>>(VecData<int32_t,8> vec, Integer i) {}
  //template <> inline int64_t extract_intrin<VecData<int64_t,4>>(VecData<int64_t,4> vec, Integer i) {}
  //template <> inline float extract_intrin<VecData<float,8>>(VecData<float,8> vec, Integer i) {}
  //template <> inline double extract_intrin<VecData<double,4>>(VecData<double,4> vec, Integer i) {}

  //template <> inline void insert_intrin<VecData<int8_t,32>>(VecData<int8_t,32>& vec, Integer i, int8_t value) {}
  //template <> inline void insert_intrin<VecData<int16_t,16>>(VecData<int16_t,16>& vec, Integer i, int16_t value) {}
  //template <> inline void insert_intrin<VecData<int32_t,8>>(VecData<int32_t,8>& vec, Integer i, int32_t value) {}
  //template <> inline void insert_intrin<VecData<int64_t,4>>(VecData<int64_t,4>& vec, Integer i, int64_t value) {}
  //template <> inline void insert_intrin<VecData<float,8>>(VecData<float,8>& vec, Integer i, float value) {}
  //template <> inline void insert_intrin<VecData<double,4>>(VecData<double,4>& vec, Integer i, double value) {}

  // Arithmetic operators
  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> unary_minus_intrin<VecData<int8_t,32>>(const VecData<int8_t,32>& a) {
    return _mm256_sub_epi8(_mm256_setzero_si256(), a.v);
  }
  template <> inline VecData<int16_t,16> unary_minus_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a) {
    return _mm256_sub_epi16(_mm256_setzero_si256(), a.v);
  }
  template <> inline VecData<int32_t,8> unary_minus_intrin<VecData<int32_t,8>>(const VecData<int32_t,8>& a) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), a.v);
  }
  template <> inline VecData<int64_t,4> unary_minus_intrin<VecData<int64_t,4>>(const VecData<int64_t,4>& a) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), a.v);
  }
  #endif
  template <> inline VecData<float,8> unary_minus_intrin<VecData<float,8>>(const VecData<float,8>& a) {
    return _mm256_xor_ps(a.v, _mm256_set1_ps(-0.0f));
  }
  template <> inline VecData<double,4> unary_minus_intrin<VecData<double,4>>(const VecData<double,4>& a) {
    static constexpr union {
      int32_t i[8];
      __m256  ymm;
    } u = {{0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000,0,(int)0x80000000}};
    return _mm256_xor_pd(a.v, _mm256_castps_pd(u.ymm));
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> mul_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    // There is no 8-bit multiply in SSE2. Split into two 16-bit multiplies
    __m256i aodd    = _mm256_srli_epi16(a.v,8);               // odd numbered elements of a
    __m256i bodd    = _mm256_srli_epi16(b.v,8);               // odd numbered elements of b
    __m256i muleven = _mm256_mullo_epi16(a.v,b.v);            // product of even numbered elements
    __m256i mulodd  = _mm256_mullo_epi16(aodd,bodd);          // product of odd  numbered elements
            mulodd  = _mm256_slli_epi16(mulodd,8);            // put odd numbered elements back in place
    #if defined(__AVX512VL__) && defined(__AVX512BW__)
    return _mm256_mask_mov_epi8(mulodd, 0x55555555, muleven);
    #else
    __m256i mask    = _mm256_set1_epi32(0x00FF00FF);          // mask for even positions
    __m256i product = selectb(mask,muleven,mulodd);           // interleave even and odd
    return product;
    #endif
  }
  template <> inline VecData<int16_t,16> mul_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_mullo_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> mul_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_mullo_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> mul_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    #if defined(__AVX512DQ__) && defined(__AVX512VL__)
    return _mm256_mullo_epi64(a.v, b.v);
    #else
    // Split into 32-bit multiplies
    __m256i bswap   = _mm256_shuffle_epi32(b.v,0xB1);         // swap H<->L
    __m256i prodlh  = _mm256_mullo_epi32(a.v,bswap);          // 32 bit L*H products
    __m256i zero    = _mm256_setzero_si256();                 // 0
    __m256i prodlh2 = _mm256_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m256i prodll  = _mm256_mul_epu32(a.v,b.v);              // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m256i prod    = _mm256_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return  prod;
    #endif
  }
  #endif
  template <> inline VecData<float,8> mul_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_mul_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> mul_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_mul_pd(a.v, b.v);
  }

  template <> inline VecData<float,8> div_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_div_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> div_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_div_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> add_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_add_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> add_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_add_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> add_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_add_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> add_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_add_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> add_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_add_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> add_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_add_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> sub_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_sub_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> sub_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_sub_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> sub_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_sub_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> sub_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_sub_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> sub_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_sub_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> sub_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_sub_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,32> fma_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b, const VecData<int8_t,32>& c) {}
  //template <> inline VecData<int16_t,16> fma_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b, const VecData<int16_t,16>& c) {}
  //template <> inline VecData<int32_t,8> sub_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b, const VecData<int32_t,8>& c) {}
  //template <> inline VecData<int64_t,4> sub_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b, const VecData<int64_t,4>& c) {}
  template <> inline VecData<float,8> fma_intrin(const VecData<float,8>& a, const VecData<float,8>& b, const VecData<float,8>& c) {
    #ifdef __FMA__
    return _mm256_fmadd_ps(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm256_macc_ps(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }
  template <> inline VecData<double,4> fma_intrin(const VecData<double,4>& a, const VecData<double,4>& b, const VecData<double,4>& c) {
    #ifdef __FMA__
    return _mm256_fmadd_pd(a.v, b.v, c.v);
    #elif defined(__FMA4__)
    return _mm256_macc_pd(a.v, b.v, c.v);
    #else
    return add_intrin(mul_intrin(a,b), c);
    #endif
  }

  // Bitwise operators
  template <> inline VecData<int8_t,32> not_intrin<VecData<int8_t,32>>(const VecData<int8_t,32>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<int16_t,16> not_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<int32_t,8> not_intrin<VecData<int32_t,8>>(const VecData<int32_t,8>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<int64_t,4> not_intrin<VecData<int64_t,4>>(const VecData<int64_t,4>& a) {
    #ifdef __AVX2__
    return _mm256_xor_si256(a.v, _mm256_set1_epi32(-1));
    #else
    return _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(a.v), _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
    #endif
  }
  template <> inline VecData<float,8> not_intrin<VecData<float,8>>(const VecData<float,8>& a) {
    return _mm256_xor_ps(a.v, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
  }
  template <> inline VecData<double,4> not_intrin<VecData<double,4>>(const VecData<double,4>& a) {
    return _mm256_xor_pd(a.v, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> and_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> and_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> and_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> and_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_and_si256(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> and_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_and_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> and_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_and_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> xor_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> xor_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> xor_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> xor_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_xor_si256(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> xor_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_xor_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> xor_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_xor_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> or_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> or_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> or_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  template <> inline VecData<int64_t,4> or_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_or_si256(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> or_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_or_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> or_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_or_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> andnot_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  template <> inline VecData<int16_t,16> andnot_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  template <> inline VecData<int32_t,8> andnot_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  template <> inline VecData<int64_t,4> andnot_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_andnot_si256(b.v, a.v);
  }
  #endif
  template <> inline VecData<float,8> andnot_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_andnot_ps(b.v, a.v);
  }
  template <> inline VecData<double,4> andnot_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_andnot_pd(b.v, a.v);
  }

  // Bitshift
  //template <> inline VecData<int8_t ,32> bitshiftleft_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a, const Integer& rhs) { }
  template <> inline VecData<int16_t,16> bitshiftleft_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const Integer& rhs) { return _mm256_slli_epi16(a.v , rhs); }
  #ifdef __AVX2__
  template <> inline VecData<int32_t ,8> bitshiftleft_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const Integer& rhs) { return _mm256_slli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t ,4> bitshiftleft_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const Integer& rhs) { return _mm256_slli_epi64(a.v , rhs); }
  #endif
  template <> inline VecData<float   ,8> bitshiftleft_intrin<VecData<float   ,8>>(const VecData<float   ,8>& a, const Integer& rhs) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(a.v), rhs)); }
  template <> inline VecData<double  ,4> bitshiftleft_intrin<VecData<double  ,4>>(const VecData<double  ,4>& a, const Integer& rhs) { return _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_castpd_si256(a.v), rhs)); }

  //template <> inline VecData<int8_t ,32> bitshiftright_intrin<VecData<int8_t ,32>>(const VecData<int8_t ,32>& a, const Integer& rhs) { }
  template <> inline VecData<int16_t,16> bitshiftright_intrin<VecData<int16_t,16>>(const VecData<int16_t,16>& a, const Integer& rhs) { return _mm256_srli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t ,8> bitshiftright_intrin<VecData<int32_t ,8>>(const VecData<int32_t ,8>& a, const Integer& rhs) { return _mm256_srli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t ,4> bitshiftright_intrin<VecData<int64_t ,4>>(const VecData<int64_t ,4>& a, const Integer& rhs) { return _mm256_srli_epi64(a.v , rhs); }
  template <> inline VecData<float   ,8> bitshiftright_intrin<VecData<float   ,8>>(const VecData<float   ,8>& a, const Integer& rhs) { return _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_castps_si256(a.v), rhs)); }
  template <> inline VecData<double  ,4> bitshiftright_intrin<VecData<double  ,4>>(const VecData<double  ,4>& a, const Integer& rhs) { return _mm256_castsi256_pd(_mm256_srli_epi64(_mm256_castpd_si256(a.v), rhs)); }

  // Other functions
  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> max_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_max_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> max_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_max_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> max_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_max_epi32(a.v, b.v);
  }
  #endif
  #if defined(__AVX512F__) || defined(__AVX512VL__)
  template <> inline VecData<int64_t,4> max_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_max_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> max_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_max_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> max_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_max_pd(a.v, b.v);
  }

  #ifdef __AVX2__
  template <> inline VecData<int8_t,32> min_intrin(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) {
    return _mm256_min_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,16> min_intrin(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) {
    return _mm256_min_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,8> min_intrin(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) {
    return _mm256_min_epi32(a.v, b.v);
  }
  #endif
  #if defined(__AVX512F__) || defined(__AVX512VL__)
  template <> inline VecData<int64_t,4> min_intrin(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) {
    return _mm256_min_epi64(a.v, b.v);
  }
  #endif
  template <> inline VecData<float,8> min_intrin(const VecData<float,8>& a, const VecData<float,8>& b) {
    return _mm256_min_ps(a.v, b.v);
  }
  template <> inline VecData<double,4> min_intrin(const VecData<double,4>& a, const VecData<double,4>& b) {
    return _mm256_min_pd(a.v, b.v);
  }

  // Conversion operators
  template <> inline VecData<float ,8> convert_int2real_intrin<VecData<float ,8>,VecData<int32_t,8>>(const VecData<int32_t,8>& x) {
    return _mm256_cvtepi32_ps(x.v);
  }
  #if defined(__AVX512F__) || defined(__AVX512VL__)
  template <> inline VecData<double,4> convert_int2real_intrin<VecData<double,4>,VecData<int64_t,4>>(const VecData<int64_t,4>& x) {
    return _mm256_cvtepi64_pd(x.v);
  }
  #endif

  template <> inline VecData<int32_t,8> round_real2int_intrin<VecData<int32_t,8>,VecData<float ,8>>(const VecData<float ,8>& x) {
    return _mm256_cvtps_epi32(x.v);
  }
  #if defined(__AVX512F__) || defined(__AVX512VL__)
  template <> inline VecData<int64_t,4> round_real2int_intrin<VecData<int64_t,4>,VecData<double,4>>(const VecData<double,4>& x) {
    return _mm256_cvtpd_epi64(x.v);
  }
  #elif defined(__AVX2__)
  template <> inline VecData<int64_t,4> round_real2int_intrin<VecData<int64_t,4>,VecData<double,4>>(const VecData<double,4>& x) {
    return _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(x.v));
  }
  #endif

  template <> inline VecData<float ,8> round_real2real_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_round_ps(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }
  template <> inline VecData<double,4> round_real2real_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _mm256_round_pd(x.v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)); }


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  #ifdef __AVX2__
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::lt>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(_mm256_cmpgt_epi8(b.v,a.v));}
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::le>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));             }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::gt>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(_mm256_cmpgt_epi8(a.v,b.v));}
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::ge>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));             }
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::eq>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return Mask<VecData<int8_t,32>>(_mm256_cmpeq_epi8(a.v,b.v));}
  template <> inline Mask<VecData<int8_t,32>> comp_intrin<ComparisonType::ne>(const VecData<int8_t,32>& a, const VecData<int8_t,32>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));             }

  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::lt>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(_mm256_cmpgt_epi16(b.v,a.v));}
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::le>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));               }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::gt>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(_mm256_cmpgt_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::ge>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));               }
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::eq>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return Mask<VecData<int16_t,16>>(_mm256_cmpeq_epi16(a.v,b.v));}
  template <> inline Mask<VecData<int16_t,16>> comp_intrin<ComparisonType::ne>(const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));               }

  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::lt>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(_mm256_cmpgt_epi32(b.v,a.v));}
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::le>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));              }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::gt>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(_mm256_cmpgt_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::ge>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));              }
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::eq>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return Mask<VecData<int32_t,8>>(_mm256_cmpeq_epi32(a.v,b.v));}
  template <> inline Mask<VecData<int32_t,8>> comp_intrin<ComparisonType::ne>(const VecData<int32_t,8>& a, const VecData<int32_t,8>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));              }

  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::lt>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(_mm256_cmpgt_epi64(b.v,a.v));}
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::le>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::lt>(b,a));              }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::gt>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(_mm256_cmpgt_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::ge>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::gt>(b,a));              }
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::eq>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return Mask<VecData<int64_t,4>>(_mm256_cmpeq_epi64(a.v,b.v));}
  template <> inline Mask<VecData<int64_t,4>> comp_intrin<ComparisonType::ne>(const VecData<int64_t,4>& a, const VecData<int64_t,4>& b) { return ~(comp_intrin<ComparisonType::eq>(a,b));              }
  #endif

  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::lt>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::le>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::gt>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::ge>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::eq>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OS)); }
  template <> inline Mask<VecData<float,8>> comp_intrin<ComparisonType::ne>(const VecData<float,8>& a, const VecData<float,8>& b) { return Mask<VecData<float,8>>(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_OS));}

  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::lt>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::le>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::gt>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::ge>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::eq>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OS)); }
  template <> inline Mask<VecData<double,4>> comp_intrin<ComparisonType::ne>(const VecData<double,4>& a, const VecData<double,4>& b) { return Mask<VecData<double,4>>(_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_OS));}

  #if defined(__AVX2__)
  template <> inline VecData<int8_t ,32> select_intrin(const Mask<VecData<int8_t ,32>>& s, const VecData<int8_t ,32>& a, const VecData<int8_t ,32>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int16_t,16> select_intrin(const Mask<VecData<int16_t,16>>& s, const VecData<int16_t,16>& a, const VecData<int16_t,16>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int32_t ,8> select_intrin(const Mask<VecData<int32_t ,8>>& s, const VecData<int32_t ,8>& a, const VecData<int32_t ,8>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  template <> inline VecData<int64_t ,4> select_intrin(const Mask<VecData<int64_t ,4>>& s, const VecData<int64_t ,4>& a, const VecData<int64_t ,4>& b) { return _mm256_blendv_epi8(b.v, a.v, s.v); }
  #endif
  template <> inline VecData<float   ,8> select_intrin(const Mask<VecData<float   ,8>>& s, const VecData<float   ,8>& a, const VecData<float   ,8>& b) { return _mm256_blendv_ps  (b.v, a.v, s.v); }
  template <> inline VecData<double  ,4> select_intrin(const Mask<VecData<double  ,4>>& s, const VecData<double  ,4>& a, const VecData<double  ,4>& b) { return _mm256_blendv_pd  (b.v, a.v, s.v); }



  // Special functions
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<float,8>> {
    static VecData<float,8> eval(const VecData<float,8>& a) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(_mm256_maskz_rsqrt14_ps(~__mmask8(0), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(_mm256_rsqrt_ps(a.v), a.v);
      #endif
    }
    static VecData<float,8> eval(const VecData<float,8>& a, const Mask<VecData<float,8>>& m) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(_mm256_maskz_rsqrt14_ps(_mm256_movepi32_mask(_mm256_castps_si256(m.v)), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,8>>::eval(and_intrin(VecData<float,8>(_mm256_rsqrt_ps(a.v)), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<double,4>> {
    static VecData<double,4> eval(const VecData<double,4>& a) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,4>>::eval(_mm256_maskz_rsqrt14_pd(~__mmask8(0), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,4>>::eval(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(a.v))), a.v);
      #endif
    }
    static VecData<double,4> eval(const VecData<double,4>& a, const Mask<VecData<double,4>>& m) {
      #if defined(__AVX512F__) || defined(__AVX512VL__)
      constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,4>>::eval(_mm256_maskz_rsqrt14_pd(_mm256_movepi64_mask(_mm256_castpd_si256(m.v)), a.v), a.v);
      #else
      constexpr Integer newton_iter = mylog2((Integer)(digits/3.4362686889));
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,4>>::eval(and_intrin(VecData<double,4>(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(a.v)))), convert_mask2vec_intrin(m)), a.v);
      #endif
    }
  };

  #ifdef SCTL_HAVE_SVML
  template <> inline void sincos_intrin<VecData<float ,8>>(VecData<float ,8>& sinx, VecData<float ,8>& cosx, const VecData<float ,8>& x) { sinx = _mm256_sincos_ps(&cosx.v, x.v); }
  template <> inline void sincos_intrin<VecData<double,4>>(VecData<double,4>& sinx, VecData<double,4>& cosx, const VecData<double,4>& x) { sinx = _mm256_sincos_pd(&cosx.v, x.v); }

  template <> inline VecData<float ,8> exp_intrin<VecData<float ,8>>(const VecData<float ,8>& x) { return _mm256_exp_ps(x.v); }
  template <> inline VecData<double,4> exp_intrin<VecData<double,4>>(const VecData<double,4>& x) { return _mm256_exp_pd(x.v); }
  #else
  template <> inline void sincos_intrin<VecData<float ,8>>(VecData<float ,8>& sinx, VecData<float ,8>& cosx, const VecData<float ,8>& x) {
    approx_sincos_intrin<(Integer)(TypeTraits<float>::SigBits/3.2)>(sinx, cosx, x); // TODO: determine constants more precisely
  }
  template <> inline void sincos_intrin<VecData<double,4>>(VecData<double,4>& sinx, VecData<double,4>& cosx, const VecData<double,4>& x) {
    approx_sincos_intrin<(Integer)(TypeTraits<double>::SigBits/3.2)>(sinx, cosx, x); // TODO: determine constants more precisely
  }

  template <> inline VecData<float ,8> exp_intrin<VecData<float ,8>>(const VecData<float ,8>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<float>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  template <> inline VecData<double,4> exp_intrin<VecData<double,4>>(const VecData<double,4>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<double>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  #endif


#endif
}

namespace SCTL_NAMESPACE { // AVX512
#if defined(__AVX512__) || defined(__AVX512F__)
  template <> struct alignas(sizeof(int8_t) * 64) VecData<int8_t,64> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 64;
    VecData() = default;
    VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(int16_t) * 32) VecData<int16_t,32> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 32;
    VecData() = default;
    VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(int32_t) * 16) VecData<int32_t,16> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 16;
    VecData() = default;
    VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(int64_t) * 8) VecData<int64_t,8> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 8;
    VecData() = default;
    VecData(__m512i v_) : v(v_) {}
    __m512i v;
  };
  template <> struct alignas(sizeof(float) * 16) VecData<float,16> {
    using ScalarType = float;
    static constexpr Integer Size = 16;
    VecData() = default;
    VecData(__m512 v_) : v(v_) {}
    __m512 v;
  };
  template <> struct alignas(sizeof(double) * 8) VecData<double,8> {
    using ScalarType = double;
    static constexpr Integer Size = 8;
    VecData(__m512d v_) : v(v_) {}
    VecData() = default;
    __m512d v;
  };



  template <> inline VecData<int8_t,64> zero_intrin<VecData<int8_t,64>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<int16_t,32> zero_intrin<VecData<int16_t,32>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<int32_t,16> zero_intrin<VecData<int32_t,16>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<int64_t,8> zero_intrin<VecData<int64_t,8>>() {
    return _mm512_setzero_si512();
  }
  template <> inline VecData<float,16> zero_intrin<VecData<float,16>>() {
    return _mm512_setzero_ps();
  }
  template <> inline VecData<double,8> zero_intrin<VecData<double,8>>() {
    return _mm512_setzero_pd();
  }

  template <> inline VecData<int8_t,64> set1_intrin<VecData<int8_t,64>>(int8_t a) {
    return _mm512_set1_epi8(a);
  }
  template <> inline VecData<int16_t,32> set1_intrin<VecData<int16_t,32>>(int16_t a) {
    return _mm512_set1_epi16(a);
  }
  template <> inline VecData<int32_t,16> set1_intrin<VecData<int32_t,16>>(int32_t a) {
    return _mm512_set1_epi32(a);
  }
  template <> inline VecData<int64_t,8> set1_intrin<VecData<int64_t,8>>(int64_t a) {
    return _mm512_set1_epi64(a);
  }
  template <> inline VecData<float,16> set1_intrin<VecData<float,16>>(float a) {
    return _mm512_set1_ps(a);
  }
  template <> inline VecData<double,8> set1_intrin<VecData<double,8>>(double a) {
    return _mm512_set1_pd(a);
  }

  template <> inline VecData<int8_t,64> set_intrin<VecData<int8_t,64>,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t,int8_t>(int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23, int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31, int8_t v32, int8_t v33, int8_t v34, int8_t v35, int8_t v36, int8_t v37, int8_t v38, int8_t v39, int8_t v40, int8_t v41, int8_t v42, int8_t v43, int8_t v44, int8_t v45, int8_t v46, int8_t v47, int8_t v48, int8_t v49, int8_t v50, int8_t v51, int8_t v52, int8_t v53, int8_t v54, int8_t v55, int8_t v56, int8_t v57, int8_t v58, int8_t v59, int8_t v60, int8_t v61, int8_t v62, int8_t v63, int8_t v64) {
    return _mm512_set_epi8(v64,v63,v62,v61,v60,v59,v58,v57,v56,v55,v54,v53,v52,v51,v50,v49,v48,v47,v46,v45,v44,v43,v42,v41,v40,v39,v38,v37,v36,v35,v34,v33,v32,v31,v30,v29,v28,v27,v26,v25,v24,v23,v22,v21,v20,v19,v18,v17,v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int16_t,32> set_intrin<VecData<int16_t,32>,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t>(int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7, int16_t v8, int16_t v9, int16_t v10, int16_t v11, int16_t v12, int16_t v13, int16_t v14, int16_t v15, int16_t v16, int16_t v17, int16_t v18, int16_t v19, int16_t v20, int16_t v21, int16_t v22, int16_t v23, int16_t v24, int16_t v25, int16_t v26, int16_t v27, int16_t v28, int16_t v29, int16_t v30, int16_t v31, int16_t v32) {
    return _mm512_set_epi16(v32,v31,v30,v29,v28,v27,v26,v25,v24,v23,v22,v21,v20,v19,v18,v17,v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int32_t,16> set_intrin<VecData<int32_t,16>,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t>(int32_t v1, int32_t v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11, int32_t v12, int32_t v13, int32_t v14, int32_t v15, int32_t v16) {
    return _mm512_set_epi32(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<int64_t,8> set_intrin<VecData<int64_t,8>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t>(int64_t v1, int64_t v2, int64_t v3, int64_t v4, int64_t v5, int64_t v6, int64_t v7, int64_t v8) {
    return _mm512_set_epi64(v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<float,16> set_intrin<VecData<float,16>,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16) {
    return _mm512_set_ps(v16,v15,v14,v13,v12,v11,v10,v9,v8,v7,v6,v5,v4,v3,v2,v1);
  }
  template <> inline VecData<double,8> set_intrin<VecData<double,8>,double,double,double,double,double,double,double,double>(double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8) {
    return _mm512_set_pd(v8,v7,v6,v5,v4,v3,v2,v1);
  }

  template <> inline VecData<int8_t,64> load1_intrin<VecData<int8_t,64>>(int8_t const* p) {
    return _mm512_set1_epi8(p[0]);
  }
  template <> inline VecData<int16_t,32> load1_intrin<VecData<int16_t,32>>(int16_t const* p) {
    return _mm512_set1_epi16(p[0]);
  }
  template <> inline VecData<int32_t,16> load1_intrin<VecData<int32_t,16>>(int32_t const* p) {
    return _mm512_set1_epi32(p[0]);
  }
  template <> inline VecData<int64_t,8> load1_intrin<VecData<int64_t,8>>(int64_t const* p) {
    return _mm512_set1_epi64(p[0]);
  }
  template <> inline VecData<float,16> load1_intrin<VecData<float,16>>(float const* p) {
    return _mm512_set1_ps(p[0]);
  }
  template <> inline VecData<double,8> load1_intrin<VecData<double,8>>(double const* p) {
    return _mm512_set1_pd(p[0]);
  }

  template <> inline VecData<int8_t,64> loadu_intrin<VecData<int8_t,64>>(int8_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<int16_t,32> loadu_intrin<VecData<int16_t,32>>(int16_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<int32_t,16> loadu_intrin<VecData<int32_t,16>>(int32_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<int64_t,8> loadu_intrin<VecData<int64_t,8>>(int64_t const* p) {
    return _mm512_loadu_si512((__m512i const*)p);
  }
  template <> inline VecData<float,16> loadu_intrin<VecData<float,16>>(float const* p) {
    return _mm512_loadu_ps(p);
  }
  template <> inline VecData<double,8> loadu_intrin<VecData<double,8>>(double const* p) {
    return _mm512_loadu_pd(p);
  }

  template <> inline VecData<int8_t,64> load_intrin<VecData<int8_t,64>>(int8_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<int16_t,32> load_intrin<VecData<int16_t,32>>(int16_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<int32_t,16> load_intrin<VecData<int32_t,16>>(int32_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<int64_t,8> load_intrin<VecData<int64_t,8>>(int64_t const* p) {
    return _mm512_load_si512((__m512i const*)p);
  }
  template <> inline VecData<float,16> load_intrin<VecData<float,16>>(float const* p) {
    return _mm512_load_ps(p);
  }
  template <> inline VecData<double,8> load_intrin<VecData<double,8>>(double const* p) {
    return _mm512_load_pd(p);
  }

  template <> inline void storeu_intrin<VecData<int8_t,64>>(int8_t* p, VecData<int8_t,64> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int16_t,32>>(int16_t* p, VecData<int16_t,32> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int32_t,16>>(int32_t* p, VecData<int32_t,16> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<int64_t,8>>(int64_t* p, VecData<int64_t,8> vec) {
    _mm512_storeu_si512((__m512i*)p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<float,16>>(float* p, VecData<float,16> vec) {
    _mm512_storeu_ps(p, vec.v);
  }
  template <> inline void storeu_intrin<VecData<double,8>>(double* p, VecData<double,8> vec) {
    _mm512_storeu_pd(p, vec.v);
  }

  template <> inline void store_intrin<VecData<int8_t,64>>(int8_t* p, VecData<int8_t,64> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int16_t,32>>(int16_t* p, VecData<int16_t,32> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int32_t,16>>(int32_t* p, VecData<int32_t,16> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<int64_t,8>>(int64_t* p, VecData<int64_t,8> vec) {
    _mm512_store_si512((__m512i*)p, vec.v);
  }
  template <> inline void store_intrin<VecData<float,16>>(float* p, VecData<float,16> vec) {
    _mm512_store_ps(p, vec.v);
  }
  template <> inline void store_intrin<VecData<double,8>>(double* p, VecData<double,8> vec) {
    _mm512_store_pd(p, vec.v);
  }

  //template <> inline int8_t extract_intrin<VecData<int8_t,64>>(VecData<int8_t,64> vec, Integer i) {}
  //template <> inline int16_t extract_intrin<VecData<int16_t,32>>(VecData<int16_t,32> vec, Integer i) {}
  //template <> inline int32_t extract_intrin<VecData<int32_t,16>>(VecData<int32_t,16> vec, Integer i) {}
  //template <> inline int64_t extract_intrin<VecData<int64_t,8>>(VecData<int64_t,8> vec, Integer i) {}
  //template <> inline float extract_intrin<VecData<float,16>>(VecData<float,16> vec, Integer i) {}
  //template <> inline double extract_intrin<VecData<double,8>>(VecData<double,8> vec, Integer i) {}

  //template <> inline void insert_intrin<VecData<int8_t,64>>(VecData<int8_t,64>& vec, Integer i, int8_t value) {}
  //template <> inline void insert_intrin<VecData<int16_t,32>>(VecData<int16_t,32>& vec, Integer i, int16_t value) {}
  //template <> inline void insert_intrin<VecData<int32_t,16>>(VecData<int32_t,16>& vec, Integer i, int32_t value) {}
  //template <> inline void insert_intrin<VecData<int64_t,8>>(VecData<int64_t,8>& vec, Integer i, int64_t value) {}
  //template <> inline void insert_intrin<VecData<float,16>>(VecData<float,16>& vec, Integer i, float value) {}
  //template <> inline void insert_intrin<VecData<double,8>>(VecData<double,8>& vec, Integer i, double value) {}

  // Arithmetic operators
  //template <> inline VecData<int8_t,64> unary_minus_intrin<VecData<int8_t,64>>(const VecData<int8_t,64>& a) {}
  //template <> inline VecData<int16_t,32> unary_minus_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a) {}
  template <> inline VecData<int32_t,16> unary_minus_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a) {
    return _mm512_sub_epi32(_mm512_setzero_epi32(), a.v);
  }
  template <> inline VecData<int64_t,8> unary_minus_intrin<VecData<int64_t,8>>(const VecData<int64_t,8>& a) {
    return _mm512_sub_epi64(_mm512_setzero_epi32(), a.v);
  }
  template <> inline VecData<float,16> unary_minus_intrin<VecData<float,16>>(const VecData<float,16>& a) {
    return _mm512_xor_ps(a.v, _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000)));
  }
  template <> inline VecData<double,8> unary_minus_intrin<VecData<double,8>>(const VecData<double,8>& a) {
    return _mm512_xor_pd(a.v, _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000)));
  }

  //template <> inline VecData<int8_t,64> mul_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {}
  //template <> inline VecData<int16_t,32> mul_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {}
  template <> inline VecData<int32_t,16> mul_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_mullo_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> mul_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    #if defined(__AVX512DQ__)
    return _mm512_mullo_epi64(a.v, b.v);
    #elif defined (__INTEL_COMPILER)
    return _mm512_mullox_epi64(a.v, b.v);                  // _mm512_mullox_epi64 missing in gcc
    #else
    // instruction does not exist. Split into 32-bit multiplies
    //__m512i ahigh = _mm512_shuffle_epi32(a, 0xB1);       // swap H<->L
    __m512i ahigh   = _mm512_srli_epi64(a.v, 32);          // high 32 bits of each a
    __m512i bhigh   = _mm512_srli_epi64(b.v, 32);          // high 32 bits of each b
    __m512i prodahb = _mm512_mul_epu32(ahigh, b.v);        // ahigh*b
    __m512i prodbha = _mm512_mul_epu32(bhigh, a.v);        // bhigh*a
    __m512i prodhl  = _mm512_add_epi64(prodahb, prodbha);  // sum of high*low products
    __m512i prodhi  = _mm512_slli_epi64(prodhl, 32);       // same, shifted high
    __m512i prodll  = _mm512_mul_epu32(a.v, b.v);          // alow*blow = 64 bit unsigned products
    __m512i prod    = _mm512_add_epi64(prodll, prodhi);    // low*low+(high*low)<<32
    return  prod;
    #endif
  }
  template <> inline VecData<float,16> mul_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_mul_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> mul_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_mul_pd(a.v, b.v);
  }

  template <> inline VecData<float,16> div_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_div_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> div_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_div_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,64> add_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {}
  //template <> inline VecData<int16_t,32> add_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {}
  template <> inline VecData<int32_t,16> add_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_add_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> add_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_add_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> add_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_add_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> add_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_add_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,64> sub_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {}
  //template <> inline VecData<int16_t,32> sub_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {}
  template <> inline VecData<int32_t,16> sub_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_sub_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> sub_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_sub_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> sub_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_sub_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> sub_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_sub_pd(a.v, b.v);
  }

  //template <> inline VecData<int8_t,64> fma_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b, const VecData<int8_t,64>& c) {}
  //template <> inline VecData<int16_t,32> fma_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b, const VecData<int16_t,32>& c) {}
  //template <> inline VecData<int32_t,16> sub_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b, const VecData<int32_t,16>& c) {}
  //template <> inline VecData<int64_t,8> sub_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b, const VecData<int64_t,8>& c) {}
  template <> inline VecData<float,16> fma_intrin(const VecData<float,16>& a, const VecData<float,16>& b, const VecData<float,16>& c) {
    return _mm512_fmadd_ps(a.v, b.v, c.v);
  }
  template <> inline VecData<double,8> fma_intrin(const VecData<double,8>& a, const VecData<double,8>& b, const VecData<double,8>& c) {
    return _mm512_fmadd_pd(a.v, b.v, c.v);
  }

  // Bitwise operators
  template <> inline VecData<int8_t,64> not_intrin<VecData<int8_t,64>>(const VecData<int8_t,64>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<int16_t,32> not_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<int32_t,16> not_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<int64_t,8> not_intrin<VecData<int64_t,8>>(const VecData<int64_t,8>& a) {
    return _mm512_xor_si512(a.v, _mm512_set1_epi32(-1));
  }
  template <> inline VecData<float,16> not_intrin<VecData<float,16>>(const VecData<float,16>& a) {
    #ifdef __AVX512DQ__
    return _mm512_xor_ps(a.v, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
    #else
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_set1_epi32(-1)));
    #endif
  }
  template <> inline VecData<double,8> not_intrin<VecData<double,8>>(const VecData<double,8>& a) {
    #ifdef __AVX512DQ__
    return _mm512_xor_pd(a.v, _mm512_castsi512_pd(_mm512_set1_epi32(-1)));
    #else
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a.v), _mm512_set1_epi32(-1)));
    #endif
  }

  template <> inline VecData<int8_t,64> and_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> and_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> and_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> and_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_and_epi32(a.v, b.v);
  }
  template <> inline VecData<float,16> and_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_and_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> and_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_and_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,64> xor_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> xor_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> xor_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> xor_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_xor_epi32(a.v, b.v);
  }
  template <> inline VecData<float,16> xor_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_xor_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> xor_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_xor_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,64> or_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> or_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> or_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> or_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_or_epi32(a.v, b.v);
  }
  template <> inline VecData<float,16> or_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_or_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> or_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_or_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,64> andnot_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<int16_t,32> andnot_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<int32_t,16> andnot_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<int64_t,8> andnot_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_andnot_epi32(b.v, a.v);
  }
  template <> inline VecData<float,16> andnot_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_andnot_ps(b.v, a.v);
  }
  template <> inline VecData<double,8> andnot_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_andnot_pd(b.v, a.v);
  }

  // Bitshift
  //template <> inline VecData<int8_t ,64> bitshiftleft_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& a, const Integer& rhs) { }
  template <> inline VecData<int16_t,32> bitshiftleft_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a, const Integer& rhs) { return _mm512_slli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,16> bitshiftleft_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a, const Integer& rhs) { return _mm512_slli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t ,8> bitshiftleft_intrin<VecData<int64_t ,8>>(const VecData<int64_t ,8>& a, const Integer& rhs) { return _mm512_slli_epi64(a.v , rhs); }
  template <> inline VecData<float  ,16> bitshiftleft_intrin<VecData<float  ,16>>(const VecData<float  ,16>& a, const Integer& rhs) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(a.v), rhs)); }
  template <> inline VecData<double  ,8> bitshiftleft_intrin<VecData<double  ,8>>(const VecData<double  ,8>& a, const Integer& rhs) { return _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_castpd_si512(a.v), rhs)); }

  //template <> inline VecData<int8_t ,64> bitshiftright_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& a, const Integer& rhs) { }
  template <> inline VecData<int16_t,32> bitshiftright_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a, const Integer& rhs) { return _mm512_srli_epi16(a.v , rhs); }
  template <> inline VecData<int32_t,16> bitshiftright_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a, const Integer& rhs) { return _mm512_srli_epi32(a.v , rhs); }
  template <> inline VecData<int64_t ,8> bitshiftright_intrin<VecData<int64_t ,8>>(const VecData<int64_t ,8>& a, const Integer& rhs) { return _mm512_srli_epi64(a.v , rhs); }
  template <> inline VecData<float  ,16> bitshiftright_intrin<VecData<float  ,16>>(const VecData<float  ,16>& a, const Integer& rhs) { return _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(a.v), rhs)); }
  template <> inline VecData<double  ,8> bitshiftright_intrin<VecData<double  ,8>>(const VecData<double  ,8>& a, const Integer& rhs) { return _mm512_castsi512_pd(_mm512_srli_epi64(_mm512_castpd_si512(a.v), rhs)); }

  // Other functions
  template <> inline VecData<int8_t,64> max_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_max_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> max_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_max_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> max_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_max_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> max_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_max_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> max_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_max_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> max_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_max_pd(a.v, b.v);
  }

  template <> inline VecData<int8_t,64> min_intrin(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) {
    return _mm512_min_epi8(a.v, b.v);
  }
  template <> inline VecData<int16_t,32> min_intrin(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) {
    return _mm512_min_epi16(a.v, b.v);
  }
  template <> inline VecData<int32_t,16> min_intrin(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) {
    return _mm512_min_epi32(a.v, b.v);
  }
  template <> inline VecData<int64_t,8> min_intrin(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) {
    return _mm512_min_epi64(a.v, b.v);
  }
  template <> inline VecData<float,16> min_intrin(const VecData<float,16>& a, const VecData<float,16>& b) {
    return _mm512_min_ps(a.v, b.v);
  }
  template <> inline VecData<double,8> min_intrin(const VecData<double,8>& a, const VecData<double,8>& b) {
    return _mm512_min_pd(a.v, b.v);
  }

  // Conversion operators
  template <> inline VecData<float,16> convert_int2real_intrin<VecData<float,16>,VecData<int32_t,16>>(const VecData<int32_t,16>& x) { return _mm512_cvtepi32_ps(x.v); }
  template <> inline VecData<double,8> convert_int2real_intrin<VecData<double,8>,VecData<int64_t, 8>>(const VecData<int64_t, 8>& x) { return _mm512_cvtepi64_pd(x.v); }
  template <> inline VecData<int32_t,16> round_real2int_intrin<VecData<int32_t,16>,VecData<float,16>>(const VecData<float,16>& x) { return _mm512_cvtps_epi32(x.v); }
  template <> inline VecData<int64_t, 8> round_real2int_intrin<VecData<int64_t, 8>,VecData<double,8>>(const VecData<double,8>& x) { return _mm512_cvtpd_epi64(x.v); }
  template <> inline VecData<float,16> round_real2real_intrin<VecData<float,16>>(const VecData<float,16>& x) { return _mm512_roundscale_ps(x.v, _MM_FROUND_TO_NEAREST_INT); }
  template <> inline VecData<double,8> round_real2real_intrin<VecData<double,8>>(const VecData<double,8>& x) { return _mm512_roundscale_pd(x.v, _MM_FROUND_TO_NEAREST_INT); }


  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  template <> struct Mask<VecData<int8_t ,64>> {
    using ScalarType = int8_t;
    static constexpr Integer Size = 64;

    static Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    explicit Mask(const __mmask64& v_) : v(v_) {}

    __mmask64 v;
  };
  template <> struct Mask<VecData<int16_t,32>> {
    using ScalarType = int16_t;
    static constexpr Integer Size = 32;

    static Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    explicit Mask(const __mmask32& v_) : v(v_) {}

    __mmask32 v;
  };
  template <> struct Mask<VecData<int32_t,16>> {
    using ScalarType = int32_t;
    static constexpr Integer Size = 16;

    static Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    explicit Mask(const __mmask16& v_) : v(v_) {}

    __mmask16 v;
  };
  template <> struct Mask<VecData<int64_t ,8>> {
    using ScalarType = int64_t;
    static constexpr Integer Size = 8;

    static Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    explicit Mask(const __mmask8& v_) : v(v_) {}

    __mmask8  v;
  };
  template <> struct Mask<VecData<float  ,16>> {
    using ScalarType = float;
    static constexpr Integer Size = 16;

    static Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    explicit Mask(const __mmask16& v_) : v(v_) {}

    __mmask16 v;
  };
  template <> struct Mask<VecData<double  ,8>> {
    using ScalarType = double;
    static constexpr Integer Size = 8;

    static Mask Zero() {
      return Mask(0);
    }

    Mask() = default;
    Mask(const Mask&) = default;
    Mask& operator=(const Mask&) = default;
    ~Mask() = default;

    explicit Mask(const __mmask8& v_) : v(v_) {}

    __mmask8  v;
  };

  // Bitwise operators
  template <> inline Mask<VecData<int8_t ,64>> operator~<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& vec) { return Mask<VecData<int8_t ,64>>(_knot_mask64(vec.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator~<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& vec) { return Mask<VecData<int16_t,32>>(_knot_mask32(vec.v)); }
  template <> inline Mask<VecData<int32_t,16>> operator~<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& vec) { return Mask<VecData<int32_t,16>>(_knot_mask16(vec.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator~<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& vec) { return Mask<VecData<int64_t ,8>>(_knot_mask8 (vec.v)); }
  template <> inline Mask<VecData<float  ,16>> operator~<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& vec) { return Mask<VecData<float  ,16>>(_knot_mask16(vec.v)); }
  template <> inline Mask<VecData<double  ,8>> operator~<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& vec) { return Mask<VecData<double  ,8>>(_knot_mask8 (vec.v)); }

  template <> inline Mask<VecData<int8_t ,64>> operator&<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kand_mask64(a.v,b.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator&<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kand_mask32(a.v,b.v)); }
  template <> inline Mask<VecData<int32_t,16>> operator&<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kand_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator&<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kand_mask8 (a.v,b.v)); }
  template <> inline Mask<VecData<float  ,16>> operator&<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kand_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<double  ,8>> operator&<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kand_mask8 (a.v,b.v)); }

  template <> inline Mask<VecData<int8_t ,64>> operator^<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kxor_mask64(a.v,b.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator^<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kxor_mask32(a.v,b.v)); }
  template <> inline Mask<VecData<int32_t,16>> operator^<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kxor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator^<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kxor_mask8 (a.v,b.v)); }
  template <> inline Mask<VecData<float  ,16>> operator^<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kxor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<double  ,8>> operator^<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kxor_mask8 (a.v,b.v)); }

  template <> inline Mask<VecData<int8_t ,64>> operator|<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kor_mask64(a.v,b.v)); }
  template <> inline Mask<VecData<int16_t,32>> operator|<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kor_mask32(a.v,b.v)); }
  template <> inline Mask<VecData<int32_t,16>> operator|<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<int64_t ,8>> operator|<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kor_mask8 (a.v,b.v)); }
  template <> inline Mask<VecData<float  ,16>> operator|<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kor_mask16(a.v,b.v)); }
  template <> inline Mask<VecData<double  ,8>> operator|<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kor_mask8 (a.v,b.v)); }

  template <> inline Mask<VecData<int8_t ,64>> AndNot   <VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a, const Mask<VecData<int8_t ,64>>& b) { return Mask<VecData<int8_t ,64>>(_kandn_mask64(b.v,a.v)); }
  template <> inline Mask<VecData<int16_t,32>> AndNot   <VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a, const Mask<VecData<int16_t,32>>& b) { return Mask<VecData<int16_t,32>>(_kandn_mask32(b.v,a.v)); }
  template <> inline Mask<VecData<int32_t,16>> AndNot   <VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a, const Mask<VecData<int32_t,16>>& b) { return Mask<VecData<int32_t,16>>(_kandn_mask16(b.v,a.v)); }
  template <> inline Mask<VecData<int64_t ,8>> AndNot   <VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a, const Mask<VecData<int64_t ,8>>& b) { return Mask<VecData<int64_t ,8>>(_kandn_mask8 (b.v,a.v)); }
  template <> inline Mask<VecData<float  ,16>> AndNot   <VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a, const Mask<VecData<float  ,16>>& b) { return Mask<VecData<float  ,16>>(_kandn_mask16(b.v,a.v)); }
  template <> inline Mask<VecData<double  ,8>> AndNot   <VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a, const Mask<VecData<double  ,8>>& b) { return Mask<VecData<double  ,8>>(_kandn_mask8 (b.v,a.v)); }


  template <> inline VecData<int8_t ,64> convert_mask2vec_intrin<VecData<int8_t ,64>>(const Mask<VecData<int8_t ,64>>& a) { return _mm512_movm_epi8 (a.v); }
  template <> inline VecData<int16_t,32> convert_mask2vec_intrin<VecData<int16_t,32>>(const Mask<VecData<int16_t,32>>& a) { return _mm512_movm_epi16(a.v); }
  template <> inline VecData<int32_t,16> convert_mask2vec_intrin<VecData<int32_t,16>>(const Mask<VecData<int32_t,16>>& a) { return _mm512_movm_epi32(a.v); }
  template <> inline VecData<int64_t ,8> convert_mask2vec_intrin<VecData<int64_t ,8>>(const Mask<VecData<int64_t ,8>>& a) { return _mm512_movm_epi64(a.v); }
  template <> inline VecData<float  ,16> convert_mask2vec_intrin<VecData<float  ,16>>(const Mask<VecData<float  ,16>>& a) { return _mm512_castsi512_ps(_mm512_movm_epi32(a.v)); }
  template <> inline VecData<double  ,8> convert_mask2vec_intrin<VecData<double  ,8>>(const Mask<VecData<double  ,8>>& a) { return _mm512_castsi512_pd(_mm512_movm_epi64(a.v)); }

  template <> inline Mask<VecData<int8_t ,64>> convert_vec2mask_intrin<VecData<int8_t ,64>>(const VecData<int8_t ,64>& a) { return Mask<VecData<int8_t ,64>>(_mm512_movepi8_mask (a.v)); }
  template <> inline Mask<VecData<int16_t,32>> convert_vec2mask_intrin<VecData<int16_t,32>>(const VecData<int16_t,32>& a) { return Mask<VecData<int16_t,32>>(_mm512_movepi16_mask(a.v)); }
  template <> inline Mask<VecData<int32_t,16>> convert_vec2mask_intrin<VecData<int32_t,16>>(const VecData<int32_t,16>& a) { return Mask<VecData<int32_t,16>>(_mm512_movepi32_mask(a.v)); }
  template <> inline Mask<VecData<int64_t ,8>> convert_vec2mask_intrin<VecData<int64_t ,8>>(const VecData<int64_t ,8>& a) { return Mask<VecData<int64_t ,8>>(_mm512_movepi64_mask(a.v)); }
  template <> inline Mask<VecData<float  ,16>> convert_vec2mask_intrin<VecData<float  ,16>>(const VecData<float  ,16>& a) { return Mask<VecData<float  ,16>>(_mm512_movepi32_mask(_mm512_castps_si512(a.v))); }
  template <> inline Mask<VecData<double  ,8>> convert_vec2mask_intrin<VecData<double  ,8>>(const VecData<double  ,8>& a) { return Mask<VecData<double  ,8>>(_mm512_movepi64_mask(_mm512_castpd_si512(a.v))); }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  // Comparison operators
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::lt>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::le>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::gt>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::ge>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::eq>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int8_t,64>> comp_intrin<ComparisonType::ne>(const VecData<int8_t,64>& a, const VecData<int8_t,64>& b) { return Mask<VecData<int8_t,64>>(_mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE)); }

  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::lt>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::le>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::gt>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::ge>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::eq>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int16_t,32>> comp_intrin<ComparisonType::ne>(const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return Mask<VecData<int16_t,32>>(_mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE)); }

  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::lt>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::le>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::gt>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::ge>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::eq>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int32_t,16>> comp_intrin<ComparisonType::ne>(const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return Mask<VecData<int32_t,16>>(_mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE)); }

  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::lt>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT)); }
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::le>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE)); }
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::gt>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE));}
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::ge>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT));}
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::eq>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ)); }
  template <> inline Mask<VecData<int64_t,8>> comp_intrin<ComparisonType::ne>(const VecData<int64_t,8>& a, const VecData<int64_t,8>& b) { return Mask<VecData<int64_t,8>>(_mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE)); }

  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::lt>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::le>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::gt>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::ge>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::eq>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OS)); }
  template <> inline Mask<VecData<float,16>> comp_intrin<ComparisonType::ne>(const VecData<float,16>& a, const VecData<float,16>& b) { return Mask<VecData<float,16>>(_mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OS));}

  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::lt>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::le>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::gt>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::ge>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::eq>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OS)); }
  template <> inline Mask<VecData<double,8>> comp_intrin<ComparisonType::ne>(const VecData<double,8>& a, const VecData<double,8>& b) { return Mask<VecData<double,8>>(_mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OS));}

  template <> inline VecData<int8_t ,64> select_intrin(const Mask<VecData<int8_t ,64>>& s, const VecData<int8_t ,64>& a, const VecData<int8_t ,64>& b) { return _mm512_mask_blend_epi8 (s.v, b.v, a.v); }
  template <> inline VecData<int16_t,32> select_intrin(const Mask<VecData<int16_t,32>>& s, const VecData<int16_t,32>& a, const VecData<int16_t,32>& b) { return _mm512_mask_blend_epi16(s.v, b.v, a.v); }
  template <> inline VecData<int32_t,16> select_intrin(const Mask<VecData<int32_t,16>>& s, const VecData<int32_t,16>& a, const VecData<int32_t,16>& b) { return _mm512_mask_blend_epi32(s.v, b.v, a.v); }
  template <> inline VecData<int64_t ,8> select_intrin(const Mask<VecData<int64_t ,8>>& s, const VecData<int64_t ,8>& a, const VecData<int64_t ,8>& b) { return _mm512_mask_blend_epi64(s.v, b.v, a.v); }
  template <> inline VecData<float  ,16> select_intrin(const Mask<VecData<float  ,16>>& s, const VecData<float  ,16>& a, const VecData<float  ,16>& b) { return _mm512_mask_blend_ps   (s.v, b.v, a.v); }
  template <> inline VecData<double  ,8> select_intrin(const Mask<VecData<double  ,8>>& s, const VecData<double  ,8>& a, const VecData<double  ,8>& b) { return _mm512_mask_blend_pd   (s.v, b.v, a.v); }


  // Special functions
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<float,16>> {
    static constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
    static VecData<float,16> eval(const VecData<float,16>& a) {
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,16>>::eval(_mm512_rsqrt14_ps(a.v), a.v);
    }
    static VecData<float,16> eval(const VecData<float,16>& a, const Mask<VecData<float,16>>& m) {
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<float,16>>::eval(_mm512_maskz_rsqrt14_ps(m.v, a.v), a.v);
    }
  };
  template <Integer digits> struct rsqrt_approx_intrin<digits, VecData<double,8>> {
    static constexpr Integer newton_iter = mylog2((Integer)(digits/4.2144199393));
    static VecData<double,8> eval(const VecData<double,8>& a) {
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,8>>::eval(_mm512_rsqrt14_pd(a.v), a.v);
    }
    static VecData<double,8> eval(const VecData<double,8>& a, const Mask<VecData<double,8>>& m) {
      return rsqrt_newton_iter<newton_iter,newton_iter,VecData<double,8>>::eval(_mm512_maskz_rsqrt14_pd(m.v, a.v), a.v);
    }
  };

  #ifdef SCTL_HAVE_SVML
  template <> inline void sincos_intrin<VecData<float,16>>(VecData<float,16>& sinx, VecData<float,16>& cosx, const VecData<float,16>& x) { sinx = _mm512_sincos_ps(&cosx.v, x.v); }
  template <> inline void sincos_intrin<VecData<double,8>>(VecData<double,8>& sinx, VecData<double,8>& cosx, const VecData<double,8>& x) { sinx = _mm512_sincos_pd(&cosx.v, x.v); }

  template <> inline VecData<float,16> exp_intrin<VecData<float,16>>(const VecData<float,16>& x) { return _mm512_exp_ps(x.v); }
  template <> inline VecData<double,8> exp_intrin<VecData<double,8>>(const VecData<double,8>& x) { return _mm512_exp_pd(x.v); }
  #else
  template <> inline void sincos_intrin<VecData<float,16>>(VecData<float,16>& sinx, VecData<float,16>& cosx, const VecData<float,16>& x) {
    approx_sincos_intrin<(Integer)(TypeTraits<float>::SigBits/3.2)>(sinx, cosx, x); // TODO: determine constants more precisely
  }
  template <> inline void sincos_intrin<VecData<double,8>>(VecData<double,8>& sinx, VecData<double,8>& cosx, const VecData<double,8>& x) {
    approx_sincos_intrin<(Integer)(TypeTraits<double>::SigBits/3.2)>(sinx, cosx, x); // TODO: determine constants more precisely
  }

  template <> inline VecData<float,16> exp_intrin<VecData<float,16>>(const VecData<float,16>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<float>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  template <> inline VecData<double,8> exp_intrin<VecData<double,8>>(const VecData<double,8>& x) {
    return approx_exp_intrin<(Integer)(TypeTraits<double>::SigBits/3.8)>(x); // TODO: determine constants more precisely
  }
  #endif

#endif
}

#endif  //_SCTL_INTRIN_WRAPPER_HPP_

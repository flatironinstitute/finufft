#ifndef _SCTL_TENSOR_HPP_
#define _SCTL_TENSOR_HPP_

#include <sctl/common.hpp>
#include SCTL_INCLUDE(mem_mgr.hpp)
#include SCTL_INCLUDE(math_utils.hpp)

#include <iostream>
#include <iomanip>

namespace SCTL_NAMESPACE {

template <class ValueType, bool own_data, Long... Args> class Tensor {

    template <Long k> static constexpr Long SizeHelper() {
      return 1;
    }
    template <Long k, Long d, Long... dd> static constexpr Long SizeHelper() {
      return (k >= 0 ? d : 1) * SizeHelper<k+1, dd...>();
    }

    template <Long k> static constexpr Long DimHelper() {
      return 1;
    }
    template <Long k, Long d0, Long... dd> static constexpr Long DimHelper() {
      return k==0 ? d0 : DimHelper<k-1,dd...>();
    }

    template <typename T> static constexpr Long OrderHelper() {
      return 0;
    }
    template <typename T, Long d, Long... dd> static constexpr Long OrderHelper() {
      return 1 + OrderHelper<void, dd...>();
    }

    template <Long k, bool own_data_, Long... dd> struct RotateType {
      using Value = Tensor<ValueType,own_data_,dd...>;
    };
    template <bool own_data_, Long d, Long... dd> struct RotateType<0,own_data_,d,dd...> {
      using Value = Tensor<ValueType,own_data_,d,dd...>;
    };
    template <Long k, bool own_data_, Long d, Long... dd> struct RotateType<k,own_data_,d,dd...> {
      using Value = typename RotateType<k-1,own_data_,dd...,d>::Value;
    };

  public:

    static constexpr Long Order() {
      return OrderHelper<void, Args...>();
    }

    static constexpr Long Size() {
      return SizeHelper<0,Args...>();
    }

    template <Long k> static constexpr Long Dim() {
      return DimHelper<k,Args...>();
    }


    Tensor(Iterator<ValueType> src_iter = NullIterator<ValueType>()) {
      Init(src_iter);
    }

    Tensor(const Tensor &M) {
      Init((Iterator<ValueType>)M.begin());
    }

    explicit Tensor(const ValueType& v) {
      static_assert(own_data || Size() == 0, "Memory pointer must be provided to initialize Tensor types with own_data=false");
      Init(NullIterator<ValueType>());
      for (auto& x : *this) x = v;
    }

    template <bool own_data_> Tensor(const Tensor<ValueType,own_data_,Args...> &M) {
      Init((Iterator<ValueType>)M.begin());
    }

    Tensor &operator=(const Tensor &M) {
      memcopy(begin(), M.begin(), Size());
      return *this;
    }

    Tensor &operator=(const ValueType& v) {
      for (auto& x : *this) x = v;
      return *this;
    }


    Iterator<ValueType> begin() {
      return own_data ? (Iterator<ValueType>)buff : iter_[0];
    }

    ConstIterator<ValueType> begin() const {
      return own_data ? (ConstIterator<ValueType>)buff : (ConstIterator<ValueType>)iter_[0];
    }

    Iterator<ValueType> end() {
      return begin() + Size();
    }

    ConstIterator<ValueType> end() const {
      return begin() + Size();
    }


    template <class ...PackedLong> ValueType& operator()(PackedLong... ii) {
      return begin()[offset<0>(ii...)];
    }

    template <class ...PackedLong> ValueType operator()(PackedLong... ii) const {
      return begin()[offset<0>(ii...)];
    }


    typename RotateType<1,true,Args...>::Value RotateLeft() const {
      typename RotateType<1,true,Args...>::Value Tr;
      const auto& T = *this;

      constexpr Long N0 = Dim<0>();
      constexpr Long N1 = Size() / N0;
      for (Long i = 0; i < N0; i++) {
        for (Long j = 0; j < N1; j++) {
          Tr.begin()[j*N0+i] = T.begin()[i*N1+j];
        }
      }
      return Tr;
    }

    typename RotateType<Order()-1,true,Args...>::Value RotateRight() const {
      typename RotateType<Order()-1,true,Args...>::Value Tr;
      const auto& T = *this;

      constexpr Long N0 = Dim<Order()-1>();
      constexpr Long N1 = Size() / N0;
      for (Long i = 0; i < N0; i++) {
        for (Long j = 0; j < N1; j++) {
          Tr.begin()[i*N1+j] = T.begin()[j*N0+i];
        }
      }
      return Tr;
    }


    Tensor<ValueType, true, Args...> operator*(const ValueType &s) const {
      Tensor<ValueType, true, Args...> M0;
      const auto &M1 = *this;

      for (Long i = 0; i < Size(); i++) {
          M0.begin()[i] = M1.begin()[i]*s;
      }
      return M0;
    }

    template <bool own_data_> Tensor<ValueType, true, Args...> operator+(const Tensor<ValueType, own_data_, Args...> &M2) const {
      Tensor<ValueType, true, Args...> M0;
      const auto &M1 = *this;

      for (Long i = 0; i < Size(); i++) {
          M0.begin()[i] = M1.begin()[i] + M2.begin()[i];
      }
      return M0;
    }

    template <bool own_data_> Tensor<ValueType, true, Args...> operator-(const Tensor<ValueType, own_data_, Args...> &M2) const {
      Tensor<ValueType, true, Args...> M0;
      const auto &M1 = *this;

      for (Long i = 0; i < Size(); i++) {
          M0.begin()[i] = M1.begin()[i] - M2.begin()[i];
      }
      return M0;
    }

    template <bool own_data_, Long N1, Long N2> Tensor<ValueType, true, Dim<0>(), N2> operator*(const Tensor<ValueType, own_data_, N1, N2> &M2) const {
      static_assert(Order() == 2, "Multiplication is only defined for tensors of order two.");
      static_assert(Dim<1>() == N1, "Tensor dimensions dont match for multiplication.");
      Tensor<ValueType, true, Dim<0>(), N2> M0;
      const auto &M1 = *this;

      for (Long i = 0; i < Dim<0>(); i++) {
        for (Long j = 0; j < N2; j++) {
          ValueType Mij = 0;
          for (Long k = 0; k < N1; k++) {
            Mij += M1(i,k)*M2(k,j);
          }
          M0(i,j) = Mij;
        }
      }
      return M0;
    }

  private:

    template <Integer k> static Long offset() {
      return 0;
    }
    template <Integer k, class ...PackedLong> static Long offset(Long i, PackedLong... ii) {
      return i * SizeHelper<-(k+1),Args...>() + offset<k+1>(ii...);
    }

    void Init(Iterator<ValueType> src_iter) {
      if (own_data) {
        if (src_iter != NullIterator<ValueType>()) {
          memcopy((Iterator<ValueType>)buff, src_iter, Size());
        }
      } else {
        if (Size()) {
          SCTL_UNUSED(src_iter[0]);
          SCTL_UNUSED(src_iter[Size()-1]);
          iter_[0] = Ptr2Itr<ValueType>(&src_iter[0], Size());
        } else {
          iter_[0] = NullIterator<ValueType>();
        }
      }
    }

    StaticArray<ValueType,own_data?Size():0> buff;
    StaticArray<Iterator<ValueType>,own_data?0:1> iter_;
};

template <class ValueType, bool own_data, Long N1, Long N2> std::ostream& operator<<(std::ostream &output, const Tensor<ValueType, own_data, N1, N2> &M) {
  std::ios::fmtflags f(std::cout.flags());
  output << std::fixed << std::setprecision(4) << std::setiosflags(std::ios::left);
  for (Long i = 0; i < N1; i++) {
    for (Long j = 0; j < N2; j++) {
      float f = ((float)M(i,j));
      if (sctl::fabs<float>(f) < 1e-25) f = 0;
      output << std::setw(10) << ((double)f) << ' ';
    }
    output << ";\n";
  }
  std::cout.flags(f);
  return output;
}

}  // end namespace

#endif  //_SCTL_TENSOR_HPP_

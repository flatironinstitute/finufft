#ifndef _SCTL_VECTOR_HPP_
#define _SCTL_VECTOR_HPP_

#include <sctl/common.hpp>

#include <vector>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>

namespace SCTL_NAMESPACE {

template <class ValueType> Iterator<ValueType> NullIterator();

template <class ValueType> class Vector {
 public:
  typedef ValueType value_type;
  typedef ValueType& reference;
  typedef const ValueType& const_reference;
  typedef Iterator<ValueType> iterator;
  typedef ConstIterator<ValueType> const_iterator;
  typedef Long difference_type;
  typedef Long size_type;

  Vector();

  explicit Vector(Long dim_, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  Vector(const Vector& V);

  explicit Vector(const std::vector<ValueType>& V);

  explicit Vector(std::initializer_list<ValueType> V);

  ~Vector();

  void Swap(Vector<ValueType>& v1);

  void ReInit(Long dim_, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  void Write(const char* fname) const;

  void Read(const char* fname);

  Long Dim() const;

  Long Capacity() const;

  void SetZero();

  Iterator<ValueType> begin();

  ConstIterator<ValueType> begin() const;

  Iterator<ValueType> end();

  ConstIterator<ValueType> end() const;

  void PushBack(const ValueType& x);

  // Element access

  ValueType& operator[](Long j);

  const ValueType& operator[](Long j) const;

  // Vector-Vector operations

  Vector& operator=(const std::vector<ValueType>& V);

  Vector& operator=(const Vector& V);

  Vector& operator+=(const Vector& V);

  Vector& operator-=(const Vector& V);

  Vector& operator*=(const Vector& V);

  Vector& operator/=(const Vector& V);

  Vector operator+(const Vector& V) const;

  Vector operator-(const Vector& V) const;

  Vector operator*(const Vector& V) const;

  Vector operator/(const Vector& V) const;

  Vector operator-() const ;

  // Vector-Scalar operations

  template <class VType> Vector& operator=(VType s);

  template <class VType> Vector& operator+=(VType s);

  template <class VType> Vector& operator-=(VType s);

  template <class VType> Vector& operator*=(VType s);

  template <class VType> Vector& operator/=(VType s);

  template <class VType> Vector operator+(VType s) const;

  template <class VType> Vector operator-(VType s) const;

  template <class VType> Vector operator*(VType s) const;

  template <class VType> Vector operator/(VType s) const;

 private:
  void Init(Long dim_, Iterator<ValueType> data_ = NullIterator<ValueType>(), bool own_data_ = true);

  Long dim;
  Long capacity;
  Iterator<ValueType> data_ptr;
  bool own_data;
};

template <class VType, class ValueType> Vector<ValueType> operator+(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> Vector<ValueType> operator-(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> Vector<ValueType> operator*(VType s, const Vector<ValueType>& V);

template <class VType, class ValueType> Vector<ValueType> operator/(VType s, const Vector<ValueType>& V);

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Vector<ValueType>& V);

}  // end namespace

#include SCTL_INCLUDE(vector.txx)

#endif  //_SCTL_VECTOR_HPP_

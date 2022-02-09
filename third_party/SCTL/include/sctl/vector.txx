#include <cassert>
#include <iostream>
#include <iomanip>

#include SCTL_INCLUDE(mem_mgr.hpp)
#include SCTL_INCLUDE(profile.hpp)

namespace SCTL_NAMESPACE {

template <class ValueType> void Vector<ValueType>::Init(Long dim_, Iterator<ValueType> data_, bool own_data_) {
  dim = dim_;
  capacity = dim;
  own_data = own_data_;
  if (own_data) {
    if (dim > 0) {
      data_ptr = aligned_new<ValueType>(capacity);
      if (data_ != NullIterator<ValueType>()) {
        memcopy(data_ptr, data_, dim);
      }
    } else
      data_ptr = NullIterator<ValueType>();
  } else
    data_ptr = data_;
}

template <class ValueType> Vector<ValueType>::Vector() {
  Init(0);
}

template <class ValueType> Vector<ValueType>::Vector(Long dim_, Iterator<ValueType> data_, bool own_data_) {
  Init(dim_, data_, own_data_);
}

template <class ValueType> Vector<ValueType>::Vector(const Vector<ValueType>& V) {
  Init(V.Dim(), (Iterator<ValueType>)V.begin());
}

template <class ValueType> Vector<ValueType>::Vector(const std::vector<ValueType>& V) {
  Init(V.size(), Ptr2Itr<ValueType>((ValueType*)(V.size()?&V[0]:nullptr), V.size()));
}

template <class ValueType> Vector<ValueType>::Vector(std::initializer_list<ValueType> V) {
  Init(V.size(), Ptr2Itr<ValueType>((ValueType*)(V.size()?&(V.begin()[0]):nullptr), V.size()));
}

template <class ValueType> Vector<ValueType>::~Vector() {
  if (own_data) {
    if (data_ptr != NullIterator<ValueType>()) {
      aligned_delete(data_ptr);
    }
  }
  data_ptr = NullIterator<ValueType>();
  capacity = 0;
  dim = 0;
}

template <class ValueType> void Vector<ValueType>::Swap(Vector<ValueType>& v1) {
  Long dim_ = dim;
  Long capacity_ = capacity;
  Iterator<ValueType> data_ptr_ = data_ptr;
  bool own_data_ = own_data;

  dim = v1.dim;
  capacity = v1.capacity;
  data_ptr = v1.data_ptr;
  own_data = v1.own_data;

  v1.dim = dim_;
  v1.capacity = capacity_;
  v1.data_ptr = data_ptr_;
  v1.own_data = own_data_;
}

template <class ValueType> void Vector<ValueType>::ReInit(Long dim_, Iterator<ValueType> data_, bool own_data_) {
#ifdef SCTL_MEMDEBUG
  Vector<ValueType> tmp(dim_, data_, own_data_);
  this->Swap(tmp);
#else
  if (own_data_ && own_data && dim_ <= capacity) {
    dim = dim_;
    if (dim && (data_ptr != NullIterator<ValueType>()) && (data_ != NullIterator<ValueType>())) {
      memcopy(data_ptr, data_, dim);
    }
  } else {
    Vector<ValueType> tmp(dim_, data_, own_data_);
    this->Swap(tmp);
  }
#endif
}

template <class ValueType> void Vector<ValueType>::Write(const char* fname) const {
  FILE* f1 = fopen(fname, "wb+");
  if (f1 == nullptr) {
    std::cout << "Unable to open file for writing:" << fname << '\n';
    return;
  }
  StaticArray<uint64_t, 2> dim_;
  dim_[0] = (uint64_t)Dim();
  dim_[1] = 1;
  fwrite(&dim_[0], sizeof(uint64_t), 2, f1);
  if (dim_[0] && dim_[1]) fwrite(&data_ptr[0], sizeof(ValueType), dim_[0] * dim_[1], f1);
  fclose(f1);
}

template <class ValueType> void Vector<ValueType>::Read(const char* fname) {
  FILE* f1 = fopen(fname, "r");
  if (f1 == nullptr) {
    std::cout << "Unable to open file for reading:" << fname << '\n';
    return;
  }
  StaticArray<uint64_t, 2> dim_;
  Long readlen = fread(&dim_[0], sizeof(uint64_t), 2, f1);
  assert(readlen == 2);
  SCTL_UNUSED(readlen);

  if (Dim() != (Long)(dim_[0] * dim_[1])) ReInit(dim_[0] * dim_[1]);
  if (dim_[0] && dim_[1]) readlen = fread(&data_ptr[0], sizeof(ValueType), dim_[0] * dim_[1], f1);
  assert(readlen == (Long)(dim_[0] * dim_[1]));
  fclose(f1);
}

template <class ValueType> inline Long Vector<ValueType>::Dim() const { return dim; }

//template <class ValueType> inline Long Vector<ValueType>::Capacity() const { return capacity; }

template <class ValueType> void Vector<ValueType>::SetZero() {
  if (dim > 0) memset<ValueType>(data_ptr, 0, dim);
}

template <class ValueType> Iterator<ValueType> Vector<ValueType>::begin() { return data_ptr; }

template <class ValueType> ConstIterator<ValueType> Vector<ValueType>::begin() const { return data_ptr; }

template <class ValueType> Iterator<ValueType> Vector<ValueType>::end() { return data_ptr + dim; }

template <class ValueType> ConstIterator<ValueType> Vector<ValueType>::end() const { return data_ptr + dim; }

template <class ValueType> void Vector<ValueType>::PushBack(const ValueType& x) {
  if (capacity > dim) {
    data_ptr[dim] = x;
    dim++;
  } else {
    Vector<ValueType> v((Long)(capacity * 1.6) + 1);
    memcopy(v.data_ptr, data_ptr, dim);
    v.dim = dim;
    Swap(v);

    assert(capacity > dim);
    data_ptr[dim] = x;
    dim++;
  }
}

// Element access

template <class ValueType> inline ValueType& Vector<ValueType>::operator[](Long j) {
  assert(j >= 0 && j < dim);
  return data_ptr[j];
}

template <class ValueType> inline const ValueType& Vector<ValueType>::operator[](Long j) const {
  assert(j >= 0 && j < dim);
  return data_ptr[j];
}

// Vector-Vector operations

template <class ValueType> Vector<ValueType>& Vector<ValueType>::operator=(const std::vector<ValueType>& V) {
  if (dim != V.size()) ReInit(V.size());
  memcopy(data_ptr, Ptr2ConstItr<ValueType>(&V[0], V.size()), dim);
  return *this;
}

template <class ValueType> Vector<ValueType>& Vector<ValueType>::operator=(const Vector<ValueType>& V) {
  if (this != &V) {
    if (dim != V.dim) ReInit(V.dim);
    memcopy(data_ptr, V.data_ptr, dim);
  }
  return *this;
}

template <class ValueType> Vector<ValueType>& Vector<ValueType>::operator+=(const Vector<ValueType>& V) {
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) data_ptr[i] += V[i];
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> Vector<ValueType>& Vector<ValueType>::operator-=(const Vector<ValueType>& V) {
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) data_ptr[i] -= V[i];
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> Vector<ValueType>& Vector<ValueType>::operator*=(const Vector<ValueType>& V) {
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) data_ptr[i] *= V[i];
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> Vector<ValueType>& Vector<ValueType>::operator/=(const Vector<ValueType>& V) {
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) data_ptr[i] /= V[i];
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> Vector<ValueType> Vector<ValueType>::operator+(const Vector<ValueType>& V) const {
  Vector<ValueType> Vr(dim);
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] + V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> Vector<ValueType> Vector<ValueType>::operator-(const Vector<ValueType>& V) const {
  Vector<ValueType> Vr(dim);
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] - V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> Vector<ValueType> Vector<ValueType>::operator*(const Vector<ValueType>& V) const {
  Vector<ValueType> Vr(dim);
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] * V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> Vector<ValueType> Vector<ValueType>::operator/(const Vector<ValueType>& V) const {
  Vector<ValueType> Vr(dim);
  SCTL_ASSERT(V.Dim() == dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] / V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> Vector<ValueType> Vector<ValueType>::operator-() const {
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = -data_ptr[i];
  return Vr;
}

// Vector-Scalar operations

template <class ValueType> template <class VType> Vector<ValueType>& Vector<ValueType>::operator=(VType s) {
  for (Long i = 0; i < dim; i++) data_ptr[i] = s;
  return *this;
}

template <class ValueType> template <class VType> Vector<ValueType>& Vector<ValueType>::operator+=(VType s) {
  for (Long i = 0; i < dim; i++) data_ptr[i] += s;
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> template <class VType> Vector<ValueType>& Vector<ValueType>::operator-=(VType s) {
  for (Long i = 0; i < dim; i++) data_ptr[i] -= s;
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> template <class VType> Vector<ValueType>& Vector<ValueType>::operator*=(VType s) {
  for (Long i = 0; i < dim; i++) data_ptr[i] *= s;
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> template <class VType> Vector<ValueType>& Vector<ValueType>::operator/=(VType s) {
  for (Long i = 0; i < dim; i++) data_ptr[i] /= s;
  Profile::Add_FLOP(dim);
  return *this;
}

template <class ValueType> template <class VType> Vector<ValueType> Vector<ValueType>::operator+(VType s) const {
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] + s;
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> template <class VType> Vector<ValueType> Vector<ValueType>::operator-(VType s) const {
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] - s;
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> template <class VType> Vector<ValueType> Vector<ValueType>::operator*(VType s) const {
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] * s;
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> template <class VType> Vector<ValueType> Vector<ValueType>::operator/(VType s) const {
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = data_ptr[i] / s;
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class VType, class ValueType> Vector<ValueType> operator+(VType s, const Vector<ValueType>& V) {
  Long dim = V.Dim();
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = s + V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class VType, class ValueType> Vector<ValueType> operator-(VType s, const Vector<ValueType>& V) {
  Long dim = V.Dim();
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = s - V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class VType, class ValueType> Vector<ValueType> operator*(VType s, const Vector<ValueType>& V) {
  Long dim = V.Dim();
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = s * V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class VType, class ValueType> Vector<ValueType> operator/(VType s, const Vector<ValueType>& V) {
  Long dim = V.Dim();
  Vector<ValueType> Vr(dim);
  for (Long i = 0; i < dim; i++) Vr[i] = s / V[i];
  Profile::Add_FLOP(dim);
  return Vr;
}

template <class ValueType> std::ostream& operator<<(std::ostream& output, const Vector<ValueType>& V) {
  std::ios::fmtflags f(std::cout.flags());
  output << std::fixed << std::setprecision(4) << std::setiosflags(std::ios::left);
  for (Long i = 0; i < V.Dim(); i++) output << std::setw(10) << V[i] << ' ';
  output << ";\n";
  std::cout.flags(f);
  return output;
}

}  // end namespace

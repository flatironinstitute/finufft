#include <omp.h>
#include <cstring>
#include <algorithm>
#include SCTL_INCLUDE(vector.hpp)
#include SCTL_INCLUDE(mem_mgr.hpp)

namespace SCTL_NAMESPACE {

template <class ConstIter, class Iter, class Int, class StrictWeakOrdering> inline void omp_par::merge(ConstIter A_, ConstIter A_last, ConstIter B_, ConstIter B_last, Iter C_, Int p, StrictWeakOrdering comp) {
  typedef typename std::iterator_traits<Iter>::difference_type _DiffType;
  typedef typename std::iterator_traits<Iter>::value_type _ValType;

  _DiffType N1 = A_last - A_;
  _DiffType N2 = B_last - B_;
  if (N1 == 0 && N2 == 0) return;
  if (N1 == 0 || N2 == 0) {
    ConstIter A = (N1 == 0 ? B_ : A_);
    _DiffType N = (N1 == 0 ? N2 : N1);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p; i++) {
      _DiffType indx1 = (i * N) / p;
      _DiffType indx2 = ((i + 1) * N) / p;
      if (indx2 - indx1 > 0) memcpy(&C_[indx1], &A[indx1], (indx2 - indx1) * sizeof(_ValType));
    }
    return;
  }

  // Split both arrays ( A and B ) into n equal parts.
  // Find the position of each split in the final merged array.
  int n = 10;
  Vector<_ValType> split;
  split.ReInit(p * n * 2);
  Vector<_DiffType> split_size;
  split_size.ReInit(p * n * 2);
#pragma omp parallel for
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < n; j++) {
      int indx = i * n + j;
      _DiffType indx1 = (indx * N1) / (p * n);
      split[indx] = A_[indx1];
      split_size[indx] = indx1 + (std::lower_bound(B_, B_last, split[indx], comp) - B_);

      indx1 = (indx * N2) / (p * n);
      indx += p * n;
      split[indx] = B_[indx1];
      split_size[indx] = indx1 + (std::lower_bound(A_, A_last, split[indx], comp) - A_);
    }
  }

  // Find the closest split position for each thread that will
  // divide the final array equally between the threads.
  Vector<_DiffType> split_indx_A;
  split_indx_A.ReInit(p + 1);
  Vector<_DiffType> split_indx_B;
  split_indx_B.ReInit(p + 1);
  split_indx_A[0] = 0;
  split_indx_B[0] = 0;
  split_indx_A[p] = N1;
  split_indx_B[p] = N2;
#pragma omp parallel for schedule(static)
  for (int i = 1; i < p; i++) {
    _DiffType req_size = (i * (N1 + N2)) / p;

    int j = std::lower_bound(split_size.begin(), split_size.begin() + p * n, req_size, std::less<_DiffType>()) - split_size.begin();
    if (j >= p * n) j = p * n - 1;
    _ValType split1 = split[j];
    _DiffType split_size1 = split_size[j];

    j = (std::lower_bound(split_size.begin() + p * n, split_size.begin() + p * n * 2, req_size, std::less<_DiffType>()) - split_size.begin() + p * n) + p * n;
    if (j >= 2 * p * n) j = 2 * p * n - 1;
    if (abs(split_size[j] - req_size) < abs(split_size1 - req_size)) {
      split1 = split[j];
      split_size1 = split_size[j];
    }

    split_indx_A[i] = std::lower_bound(A_, A_last, split1, comp) - A_;
    split_indx_B[i] = std::lower_bound(B_, B_last, split1, comp) - B_;
  }

// Merge for each thread independently.
#pragma omp parallel for schedule(static)
  for (int i = 0; i < p; i++) {
    Iter C = C_ + split_indx_A[i] + split_indx_B[i];
    std::merge(A_ + split_indx_A[i], A_ + split_indx_A[i + 1], B_ + split_indx_B[i], B_ + split_indx_B[i + 1], C, comp);
  }
}

template <class T, class StrictWeakOrdering> inline void omp_par::merge_sort(T A, T A_last, StrictWeakOrdering comp) {
  typedef typename std::iterator_traits<T>::difference_type _DiffType;
  typedef typename std::iterator_traits<T>::value_type _ValType;

  int p = omp_get_max_threads();
  _DiffType N = A_last - A;
  if (N < 2 * p) {
    std::sort(A, A_last, comp);
    return;
  }
  SCTL_UNUSED(A[0]);
  SCTL_UNUSED(A[N - 1]);

  // Split the array A into p equal parts.
  Vector<_DiffType> split;
  split.ReInit(p + 1);
  split[p] = N;
#pragma omp parallel for schedule(static)
  for (int id = 0; id < p; id++) {
    split[id] = (id * N) / p;
  }

// Sort each part independently.
#pragma omp parallel for schedule(static)
  for (int id = 0; id < p; id++) {
    std::sort(A + split[id], A + split[id + 1], comp);
  }

  // Merge two parts at a time.
  Vector<_ValType> B;
  B.ReInit(N);
  Iterator<_ValType> A_ = Ptr2Itr<_ValType>(&A[0], N);
  Iterator<_ValType> B_ = B.begin();
  for (int j = 1; j < p; j = j * 2) {
    for (int i = 0; i < p; i = i + 2 * j) {
      if (i + j < p) {
        omp_par::merge(A_ + split[i], A_ + split[i + j], A_ + split[i + j], A_ + split[(i + 2 * j <= p ? i + 2 * j : p)], B_ + split[i], p, comp);
      } else {
#pragma omp parallel for
        for (int k = split[i]; k < split[p]; k++) B_[k] = A_[k];
      }
    }
    Iterator<_ValType> tmp_swap = A_;
    A_ = B_;
    B_ = tmp_swap;
  }

  // The final result should be in A.
  if (A_ != Ptr2Itr<_ValType>(&A[0], N)) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) A[i] = A_[i];
  }
}

template <class T> inline void omp_par::merge_sort(T A, T A_last) {
  typedef typename std::iterator_traits<T>::value_type _ValType;
  omp_par::merge_sort(A, A_last, std::less<_ValType>());
}

template <class ConstIter, class Int> typename std::iterator_traits<ConstIter>::value_type omp_par::reduce(ConstIter A, Int cnt) {
  typedef typename std::iterator_traits<ConstIter>::value_type ValueType;
  ValueType sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (Int i = 0; i < cnt; i++) sum += A[i];
  return sum;
}

template <class ConstIter, class Iter, class Int> void omp_par::scan(ConstIter A, Iter B, Int cnt) {
  typedef typename std::iterator_traits<Iter>::value_type ValueType;

  Integer p = omp_get_max_threads();
  if (cnt < (Int)100 * p) {
    for (Int i = 1; i < cnt; i++) B[i] = B[i - 1] + A[i - 1];
    return;
  }

  Int step_size = cnt / p;

#pragma omp parallel for schedule(static)
  for (Integer i = 0; i < p; i++) {
    Int start = i * step_size;
    Int end = start + step_size;
    if (i == p - 1) end = cnt;
    if (i != 0) B[start] = 0;
    for (Int j = (Int)start + 1; j < (Int)end; j++) B[j] = B[j - 1] + A[j - 1];
  }

  Vector<ValueType> sum(p);
  sum[0] = 0;
  for (Integer i = 1; i < p; i++) sum[i] = sum[i - 1] + B[i * step_size - 1] + A[i * step_size - 1];

#pragma omp parallel for schedule(static)
  for (Integer i = 1; i < p; i++) {
    Int start = i * step_size;
    Int end = start + step_size;
    if (i == p - 1) end = cnt;
    ValueType sum_ = sum[i];
    for (Int j = (Int)start; j < (Int)end; j++) B[j] += sum_;
  }
}

}  // end namespace

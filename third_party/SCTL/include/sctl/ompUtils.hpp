#ifndef _SCTL_OMP_UTILS_H_
#define _SCTL_OMP_UTILS_H_

#include <sctl/common.hpp>

#include <iterator>

namespace SCTL_NAMESPACE {
namespace omp_par {

template <class ConstIter, class Iter, class Int, class StrictWeakOrdering> void merge(ConstIter A_, ConstIter A_last, ConstIter B_, ConstIter B_last, Iter C_, Int p, StrictWeakOrdering comp);

template <class T, class StrictWeakOrdering> void merge_sort(T A, T A_last, StrictWeakOrdering comp);

template <class T> void merge_sort(T A, T A_last);

template <class ConstIter, class Int> typename std::iterator_traits<ConstIter>::value_type reduce(ConstIter A, Int cnt);

template <class ConstIter, class Iter, class Int> void scan(ConstIter A, Iter B, Int cnt);

}  // end namespace omp_par
}  // end namespace SCTL_NAMESPACE

#include SCTL_INCLUDE(ompUtils.txx)

#endif  //_SCTL_OMP_UTILS_H_

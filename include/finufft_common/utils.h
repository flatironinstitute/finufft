#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include "defines.h"

namespace finufft {
namespace common {

FINUFFT_EXPORT_TEST void gaussquad(int n, double *xgl, double *wgl);
std::tuple<double, double> leg_eval(int n, double x);

// Series implementation of the modified Bessel function of the first kind I_nu(x)
double cyl_bessel_i(double nu, double x) noexcept;

// helper to generate the integer sequence in range [Start, End]
template<int Offset, typename Seq> struct offset_seq;

template<int Offset, int... I>
struct offset_seq<Offset, std::integer_sequence<int, I...>> {
  using type = std::integer_sequence<int, (Offset + I)...>;
};

template<int Start, int End>
using make_range =
    typename offset_seq<Start, std::make_integer_sequence<int, End - Start + 1>>::type;

template<typename Seq> struct DispatchParam {
  int runtime_val;
  using seq_type = Seq;
};

} // namespace common
} // namespace finufft

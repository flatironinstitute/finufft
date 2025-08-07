#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include "defines.h"

namespace finufft {
namespace common {

FINUFFT_EXPORT void FINUFFT_CDECL gaussquad(int n, double *xgl, double *wgl);
std::tuple<double, double> leg_eval(int n, double x);

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

// Cartesian product over integer sequences.
// Invokes f.template operator()<...>() for each combination of values.
// The functor F must provide a templated call operator.
namespace detail {

template<typename F, typename... Seq> struct Product;

// Recursive case: at least two sequences remaining
template<typename F, int... I1, typename Seq2, typename... Rest>
struct Product<F, std::integer_sequence<int, I1...>, Seq2, Rest...> {
  template<int... Prefix> static void apply(F &f) {
    (Product<F, Seq2, Rest...>::template apply<Prefix..., I1>(f), ...);
  }
};

// Base case: single sequence left
template<typename F, int... I1> struct Product<F, std::integer_sequence<int, I1...>> {
  template<int... Prefix> static void apply(F &f) {
    (f.template operator()<Prefix..., I1>(), ...);
  }
};

template<typename F, typename... Seq> void product(F &f, Seq...) {
  Product<F, Seq...>::template apply<>(f);
}

// Helper functor invoked for each combination to check runtime values
template<typename Func, std::size_t N, typename ArgTuple, typename ResultType>
struct DispatcherCaller {
  Func &func;
  const std::array<int, N> &vals;
  ArgTuple &args;
  std::conditional_t<std::is_void_v<ResultType>, char, ResultType> result{};
  template<int... Params> void operator()() {
    static constexpr std::array<int, sizeof...(Params)> p{Params...};
    if (p == vals) {
      if constexpr (std::is_void_v<ResultType>) {
        std::apply(
            [&](auto &&...a) {
              func.template operator()<Params...>(std::forward<decltype(a)>(a)...);
            },
            args);
      } else {
        result = std::apply(
            [&](auto &&...a) {
              return func.template operator()<Params...>(std::forward<decltype(a)>(a)...);
            },
            args);
      }
    }
  }
};

template<typename Seq> struct seq_first;
template<int I0, int... I>
struct seq_first<std::integer_sequence<int, I0, I...>> : std::integral_constant<int, I0> {
};

template<typename Tuple, std::size_t... I>
auto extract_vals_impl(const Tuple &t, std::index_sequence<I...>) {
  return std::array<int, sizeof...(I)>{std::get<I>(t).runtime_val...};
}
template<typename Tuple> auto extract_vals(const Tuple &t) {
  using T = std::remove_reference_t<Tuple>;
  return extract_vals_impl(t, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template<typename Tuple, std::size_t... I>
auto extract_seqs_impl(const Tuple &t, std::index_sequence<I...>) {
  using T = std::remove_reference_t<Tuple>;
  return std::make_tuple(typename std::tuple_element_t<I, T>::seq_type{}...);
}
template<typename Tuple> auto extract_seqs(const Tuple &t) {
  using T = std::remove_reference_t<Tuple>;
  return extract_seqs_impl(t, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template<typename Func, typename ArgTuple, typename... Seq>
struct dispatch_result_helper {
  template<std::size_t... I>
  static auto test(std::index_sequence<I...>)
      -> decltype(std::declval<Func>().template operator()<seq_first<Seq>::value...>(
          std::get<I>(std::declval<ArgTuple>())...));
  using type = decltype(test(std::make_index_sequence<std::tuple_size_v<ArgTuple>>{}));
};
template<typename Func, typename ArgTuple, typename SeqTuple> struct dispatch_result;
template<typename Func, typename ArgTuple, typename... Seq>
struct dispatch_result<Func, ArgTuple, std::tuple<Seq...>> {
  using type = typename dispatch_result_helper<Func, ArgTuple, Seq...>::type;
};
template<typename Func, typename ArgTuple, typename SeqTuple>
using dispatch_result_t = typename dispatch_result<Func, ArgTuple, SeqTuple>::type;

} // namespace detail

// Generic dispatcher mapping runtime ints to template parameters.
// params is a tuple of DispatchParam holding runtime values and sequences.
// When a match is found, the functor is invoked with those template parameters
// and its result returned. Otherwise, the default-constructed result is returned.
template<typename Func, typename ParamTuple, typename... Args>
decltype(auto) dispatch(Func &&func, ParamTuple &&params, Args &&...args) {
  using tuple_t           = std::remove_reference_t<ParamTuple>;
  constexpr std::size_t N = std::tuple_size_v<tuple_t>;
  auto vals               = detail::extract_vals(params);
  auto seqs               = detail::extract_seqs(params);
  auto arg_tuple          = std::forward_as_tuple(std::forward<Args>(args)...);
  using result_t = detail::dispatch_result_t<Func, decltype(arg_tuple), decltype(seqs)>;
  detail::DispatcherCaller<Func, N, decltype(arg_tuple), result_t> caller{func, vals,
                                                                          arg_tuple};
  std::apply([&](auto &&...s) { detail::product(caller, s...); }, seqs);
  if constexpr (!std::is_void_v<result_t>) return caller.result;
}

} // namespace common
} // namespace finufft

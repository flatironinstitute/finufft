#pragma once

#include <array>
#include <cstdint>
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
// Adapted upon suggestion from Nils Wentzell: godbolt.org/z/GM94xb1j4
//
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
auto extract_seqs_impl(const Tuple &, std::index_sequence<I...>) {
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

// ----------------------------------------------
// compile-time range length for [Start, Stop) with step Inc
// ----------------------------------------------
template<std::int64_t Start, std::int64_t Stop, std::int64_t Inc>
inline constexpr std::int64_t compute_range_count = [] {
  static_assert(Inc != 0, "Inc must not be zero");
  if constexpr (Inc > 0) {
    const std::int64_t d = Stop - Start;
    return d > 0 ? ((d + Inc - 1) / Inc) : 0;
  } else {
    const std::int64_t d = Start - Stop;
    const std::int64_t s = -Inc;
    return d > 0 ? ((d + s - 1) / s) : 0;
  }
}();

// ----------------------------------------------
// low-level emitters (C++17, no templated lambdas)
// ----------------------------------------------
template<std::int64_t Start, std::int64_t Inc, class F, std::size_t... Is>
constexpr void static_loop_impl_block(F &&f, std::index_sequence<Is...>) {
  (f(std::integral_constant<std::int64_t, Start + static_cast<std::int64_t>(Is) * Inc>{}),
   ...);
}

template<std::int64_t Start, std::int64_t Inc, std::int64_t K, class F, std::size_t... Bs>
constexpr void static_loop_emit_all_blocks(F &&f, std::index_sequence<Bs...>) {
  static_assert(K > 0, "UNROLL K must be positive");
  (static_loop_impl_block<Start + static_cast<std::int64_t>(Bs) * K * Inc, Inc>(
       f, std::make_index_sequence<static_cast<std::size_t>(K)>{}),
   ...);
}

// ----------------------------------------------
// static_loop
// ----------------------------------------------
template<std::int64_t Start, std::int64_t Stop, std::int64_t Inc = 1,
         std::int64_t UNROLL = compute_range_count<Start, Stop, Inc>, class F>
constexpr void static_loop(F &&f) {
  static_assert(Inc != 0, "Inc must not be zero");

  constexpr std::int64_t Count = compute_range_count<Start, Stop, Inc>;
  if constexpr (Count == 0) {
    // do nothing
  } else {
    constexpr std::int64_t k = (UNROLL > 0 ? UNROLL : Count);
    static_assert(k > 0, "internal: k must be positive");
    constexpr std::int64_t Blocks = Count / k;
    constexpr std::int64_t Tail   = Count % k;

    if constexpr (Blocks > 0) {
      static_loop_emit_all_blocks<Start, Inc, k>(
          std::forward<F>(f),
          std::make_index_sequence<static_cast<std::size_t>(Blocks)>{});
    }
    if constexpr (Tail > 0) {
      constexpr std::int64_t TailStart = Start + Blocks * k * Inc;
      static_loop_impl_block<TailStart, Inc>(
          std::forward<F>(f), std::make_index_sequence<static_cast<std::size_t>(Tail)>{});
    }
  }
}

// convenience: Stop only => Start=0, Inc=1
template<std::int64_t Stop, class F> constexpr void static_loop(F &&f) {
  static_loop<0, Stop, 1, compute_range_count<0, Stop, 1>>(std::forward<F>(f));
}

// ----------------------------------------------
// static_for wrappers expecting f.template operator()<I>()
// keeps C++17 by adapting to integral_constant form above
// ----------------------------------------------
namespace detail {
template<class F> struct as_template_index {
  F *pf;
  constexpr explicit as_template_index(F &f) : pf(&f) {}
  template<std::int64_t I>
  constexpr void operator()(std::integral_constant<std::int64_t, I>) const {
    pf->template operator()<I>();
  }
};
template<class T>
using is_integral_cx = std::integral_constant<bool, std::is_integral_v<T>>;
} // namespace detail

template<auto Start, auto End, class F> constexpr void static_for(F &&f) {
  static_assert(detail::is_integral_cx<decltype(Start)>::value &&
                    detail::is_integral_cx<decltype(End)>::value,
                "Start/End must be integral constant expressions");
  static_assert(End >= Start, "End must be >= Start");

  constexpr auto S = static_cast<std::int64_t>(Start);
  constexpr auto E = static_cast<std::int64_t>(End);
  static_loop<S, E, 1>(detail::as_template_index<F>{f});
}

template<auto End, class F> constexpr void static_for(F &&f) {
  static_for<0, End>(std::forward<F>(f));
}
} // namespace common
} // namespace finufft

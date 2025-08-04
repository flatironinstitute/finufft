#pragma once
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace finufft {
namespace utils {

// helper to generate integer sequence in range [Start, End]
template<int Offset, typename Seq> struct offset_seq;

template<int Offset, int... I>
struct offset_seq<Offset, std::integer_sequence<int, I...>> {
  using type = std::integer_sequence<int, (Offset + I)...>;
};

template<int Start, int End>
using make_range =
    typename offset_seq<Start, std::make_integer_sequence<int, End - Start + 1>>::type;

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

template<typename F, typename... Seq> void product(F &f, Seq... seqs) {
  Product<F, Seq...>::template apply<>(f);
}

// Helper functor invoked for each combination to check runtime values
template<typename Func, std::size_t N, typename ArgTuple> struct DispatcherCaller {
  Func &func;
  const std::array<int, N> &vals;
  ArgTuple &args;
  bool matched = false;
  int result   = 0;
  template<int... Params> void operator()() {
    if (matched) return;
    std::array<int, sizeof...(Params)> p{Params...};
    if (p == vals) {
      result = std::apply(
          [&](auto &&...a) {
            return func.template operator()<Params...>(std::forward<decltype(a)>(a)...);
          },
          args);
      matched = true;
    }
  }
};

} // namespace detail

// Generic dispatcher mapping runtime ints to template parameters.
// vals contains the runtime values corresponding to each sequence.
// When a match is found, the functor is invoked with those template parameters
// and its result returned. Otherwise, zero is returned.
template<typename Func, std::size_t N, typename SeqTuple, typename... Args>
int dispatch(Func &&func, const std::array<int, N> &vals, SeqTuple &&seqs,
             Args &&...args) {
  using tuple_t = std::remove_reference_t<SeqTuple>;
  static_assert(N == std::tuple_size<tuple_t>::value, "vals size must match sequences");
  auto arg_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
  detail::DispatcherCaller<Func, N, decltype(arg_tuple)> caller{func, vals, arg_tuple};
  std::apply([&](auto &&...s) { detail::product(caller, s...); },
             std::forward<SeqTuple>(seqs));
  return caller.result;
}

} // namespace utils
} // namespace finufft

#pragma once

#include <array>
#include <optional>
#include <type_traits>
#include <utility>

#include <common/common.h>
#include <finufft_errors.h>

namespace finufft {
namespace dispatch {

template<typename Func, typename T, int ns, typename... Args>
int dispatch_ns(Func &&func, int target_ns, Args &&...args) {
  if constexpr (ns > ::finufft::common::MAX_NSPREAD) {
    return FINUFFT_ERR_METHOD_NOTVALID;
  } else {
    if (target_ns == ns) {
      return std::forward<Func>(func).template operator()<ns>(
          std::forward<Args>(args)...);
    }
    return dispatch_ns<Func, T, ns + 1>(std::forward<Func>(func), target_ns,
                                        std::forward<Args>(args)...);
  }
}

template<typename Func, typename T, typename... Args>
int launch_dispatch_ns(Func &&func, int target_ns, Args &&...args) {
  return dispatch_ns<Func, T, ::finufft::common::MIN_NSPREAD>(
      std::forward<Func>(func), target_ns, std::forward<Args>(args)...);
}

namespace detail {

template<int MIN, int MAX, typename Seq = std::make_integer_sequence<int, MAX - MIN + 1>>
struct integer_range;

template<int MIN, int MAX, int... I>
struct integer_range<MIN, MAX, std::integer_sequence<int, I...>> {
  using type = std::integer_sequence<int, MIN + I...>;
};

template<typename Seq> struct first;

template<int I0, int... I>
struct first<std::integer_sequence<int, I0, I...>> : std::integral_constant<int, I0> {};

template<typename Func, int... Is, int... Js, typename... IntSeqs>
void product(Func &&f, std::integer_sequence<int, Js...>, IntSeqs... isqs) {
  if constexpr (sizeof...(IntSeqs) == 0) {
    (f(std::integral_constant<int, Is>{}, std::integral_constant<int, Js>{}), ...);
  } else {
    (product<Func, Is..., Js>(std::forward<Func>(f), isqs...), ...);
  }
}

template<typename Func, typename Seq, typename... Rest>
void for_each_combination(Func &&f, Seq, Rest... rest) {
  product<Func>(std::forward<Func>(f), Seq{}, rest...);
}

} // namespace detail

template<typename Func, typename... Seqs, typename... Args>
auto dispatch_multi(Func &&func, const std::array<int, sizeof...(Seqs)> &targets,
                    Args &&...args) {
  using return_t =
      decltype(std::declval<Func>().template operator()<detail::first<Seqs>::value...>(
          std::forward<Args>(args)...));
  bool matched = false;
  if constexpr (!std::is_void_v<return_t>) {
    std::optional<return_t> result;
    detail::for_each_combination(
        [&](auto... Ns) {
          if (std::array<int, sizeof...(Seqs)>{Ns...} == targets) {
            matched = true;
            result  = std::forward<Func>(func).template operator()<Ns.value...>(
                std::forward<Args>(args)...);
          }
        },
        Seqs{}...);
    if (matched) return *result;
    if constexpr (std::is_same_v<return_t, int>)
      return FINUFFT_ERR_METHOD_NOTVALID;
    else
      return return_t{};
  } else {
    detail::for_each_combination(
        [&](auto... Ns) {
          if (std::array<int, sizeof...(Seqs)>{Ns...} == targets) {
            matched = true;
            std::forward<Func>(func).template operator()<Ns.value...>(
                std::forward<Args>(args)...);
          }
        },
        Seqs{}...);
    return;
  }
}

} // namespace dispatch
} // namespace finufft

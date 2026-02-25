#pragma once
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

#include "finufft_errors.h"

namespace finufft {
namespace common {

// Wrap a C++ callable for C APIs.
// - Perfect-forwards callable/args via std::invoke + std::forward.
//   This means that for temporaries (rvalues) T is deduced as T&& and moved. (no copies)
//   For lvalues, T is deduced as T& and passed by reference. (no copies again)
// - If return type is void, return 0; otherwise require int-convertible.
// - Map exceptions to FINUFFT error codes.
template<class F, class... Args> inline int safe_finufft_call(F &&f, Args &&...args) {
  try {
    using R = std::invoke_result_t<F, Args...>;

    if constexpr (std::is_void_v<R>) {
      std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
      return 0;
    } else {
      static_assert(
          std::is_convertible_v<R, int>,
          "safe_finufft_call: callable must return void or a type convertible to int");
      return static_cast<int>(
          std::invoke(std::forward<F>(f), std::forward<Args>(args)...));
    }
  } catch (int retcode) {
    return retcode;
  } catch (const std::bad_alloc &) {
    return FINUFFT_ERR_ALLOC;
  } catch (...) {
    return FINUFFT_ERR_UNKNOWN_EXCEPTION;
  }
}

} // namespace common
} // namespace finufft

#pragma once
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "finufft_errors.h"

namespace finufft {

// Exception type for FINUFFT errors. Carries an integer error code matching
// the FINUFFT_ERR_* constants in finufft_errors.h. Thrown by internal C++
// code; caught by safe_finufft_call at the C boundary and converted back to
// an integer return code.
class exception : public std::runtime_error {
  int code_;

public:
  explicit exception(int code) : std::runtime_error(error_string(code)), code_(code) {}
  exception(int code, const std::string &detail)
      : std::runtime_error(std::string(error_string(code)) + ": " + detail), code_(code) {}
  int code() const noexcept { return code_; }

  static const char *error_string(int code) {
    switch (code) {
    case FINUFFT_WARN_EPS_TOO_SMALL: return "requested tolerance too small";
    case FINUFFT_ERR_MAXNALLOC: return "fine grid size exceeds MAX_NF";
    case FINUFFT_ERR_SPREAD_BOX_SMALL: return "grid dims smaller than 2*nspread";
    case FINUFFT_ERR_SPREAD_DIR: return "spread_direction must be 1 or 2";
    case FINUFFT_ERR_UPSAMPFAC_TOO_SMALL: return "upsampfac not > 1.0";
    case FINUFFT_ERR_HORNER_WRONG_BETA: return "wrong beta for Horner coefficients";
    case FINUFFT_ERR_NTRANS_NOTVALID: return "ntrans not valid (must be >= 1)";
    case FINUFFT_ERR_TYPE_NOTVALID: return "type not valid (must be 1, 2, or 3)";
    case FINUFFT_ERR_ALLOC: return "memory allocation failed";
    case FINUFFT_ERR_DIM_NOTVALID: return "dimension not valid (must be 1, 2, or 3)";
    case FINUFFT_ERR_SPREAD_THREAD_NOTVALID: return "spread_thread option not valid";
    case FINUFFT_ERR_NUM_NU_PTS_INVALID: return "number of NU points invalid";
    case FINUFFT_ERR_LOCK_FUNS_INVALID: return "FFTW lock/unlock functions invalid";
    case FINUFFT_ERR_NTHREADS_NOTVALID: return "number of threads not valid";
    case FINUFFT_ERR_KERFORMULA_NOTVALID: return "kernel formula not valid";
    default: return "unknown FINUFFT error";
    }
  }
};

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
  } catch (const finufft::exception &e) {
    return e.code();
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

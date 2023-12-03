#pragma once

#include <cstdlib>

// Utilities for cpuid-based dispatching of functions.

namespace finufft {

// Summary of the capability of the current cpu.
// Note that we use a simplified approach categorizing the capabilities as:
// - Scalar: no SIMD instruction capability
// - SSE4: SSE4.1 instructions
// - AVX2: AVX2 + FMA instructions
// - AVX512: AVX512F + AVX512VL + AVX512DQ
// Note that it is assumed that these capabilities are inclusive, that is,
// AVX512 implies AVX2 and SSE4 etc.
struct Dispatch {
    enum Type : int { Scalar = 0, SSE4 = 1, AVX2 = 2, AVX512 = 3 };
};

// Query the current cpu for its capability.
Dispatch::Type get_current_capability() noexcept;

// Query the current dispatch target, can be different from the current capability
// on user request (e.g. by setting the environment variable FINUFFT_DISPATCH).
Dispatch::Type get_current_dispatch_target() noexcept;

// Utility class to assemble a lookup table for a kernel dispatch table.
template <typename Sig, typename FnScalar, typename FnSse, typename FnAvx2, typename FnAvx512>
class LazyDispatchedFunctor;

template <
    typename FnScalar, typename FnSse, typename FnAvx2, typename FnAvx512, typename R,
    typename... Args>
class LazyDispatchedFunctor<R(Args...), FnScalar, FnSse, FnAvx2, FnAvx512> {
    FnScalar fn_scalar_;
    FnSse fn_sse_;
    FnAvx2 fn_avx2_;
    FnAvx512 fn_avx512_;
    Dispatch::Type dispatch_target_;

  public:
    LazyDispatchedFunctor(FnScalar fn_scalar, FnSse fn_sse, FnAvx2 fn_avx2, FnAvx512 fn_avx512)
        : fn_scalar_(fn_scalar), fn_sse_(fn_sse), fn_avx2_(fn_avx2), fn_avx512_(fn_avx512),
          dispatch_target_(get_current_dispatch_target()) {}
    LazyDispatchedFunctor(const LazyDispatchedFunctor &) = delete;
    LazyDispatchedFunctor(LazyDispatchedFunctor &&) = default;

    R operator()(Args... args) const noexcept {
        switch (dispatch_target_) {
        case Dispatch::Scalar:
            return fn_scalar_(args...);
        case Dispatch::SSE4:
            return fn_sse_(args...);
        case Dispatch::AVX2:
            return fn_avx2_(args...);
        case Dispatch::AVX512:
            return fn_avx512_(args...);
        default:
            // invalid dispatch value
            std::abort();
        }
    }
};

// Template type deduction helper.
template <typename Sig, typename FnScalar, typename FnSse, typename FnAvx2, typename FnAvx512>
LazyDispatchedFunctor<Sig, FnScalar, FnSse, FnAvx2, FnAvx512>
make_dispatched_functor(FnScalar fn_scalar, FnSse fn_sse, FnAvx2 fn_avx2, FnAvx512 fn_avx512) {
    return LazyDispatchedFunctor<Sig, FnScalar, FnSse, FnAvx2, FnAvx512>(
        fn_scalar, fn_sse, fn_avx2, fn_avx512);
}

} // namespace finufft

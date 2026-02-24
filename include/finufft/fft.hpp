#pragma once
// Forward declaration only. Full definition in src/fft.cpp.
template<typename T> class Finufft_FFT_plan;

// Custom deleter for unique_ptr<Finufft_FFT_plan<T>> so that the complete
// Finufft_FFT_plan type is only required in fft.cpp (where operator() is defined),
// not in every TU that instantiates FINUFFT_PLAN_T's constructor or destructor.
template<typename T> struct Finufft_FFT_plan_deleter {
  void operator()(Finufft_FFT_plan<T> *p) const; // defined in fft.cpp
};

// FFTW global cleanup utilities (defined in fft.cpp).
// FINUFFT_EXPORT_TEST: exported only when FINUFFT_BUILD_TESTS is set.
#include <finufft_common/defines.h>
FINUFFT_EXPORT_TEST void finufft_fft_forget_wisdom();
FINUFFT_EXPORT_TEST void finufft_fft_cleanup();
FINUFFT_EXPORT_TEST void finufft_fft_cleanup_threads();

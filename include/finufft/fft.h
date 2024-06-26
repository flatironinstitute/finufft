#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#ifdef FINUFFT_USE_DUCC0
#include "ducc0/fft/fftnd_impl.h"
// temporary hack since some tests call these unconditionally
static inline void FFTW_FORGET_WISDOM() {}
static inline void FFTW_CLEANUP() {}
static inline void FFTW_CLEANUP_THREADS() {}
#else
#include "fftw_defs.h"
#endif

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H

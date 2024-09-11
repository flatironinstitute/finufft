#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#ifdef FINUFFT_USE_DUCC0
#include "ducc0/fft/fftnd_impl.h"
// temporary hacks to allow compilation of tests that assume FFTW is used
static inline void FFTW_FORGET_WISDOM() {}
static inline void FFTW_CLEANUP() {}
static inline void FFTW_CLEANUP_THREADS() {}
#else
#include "fftw_defs.h"
#endif
#include <finufft/defs.h>

int *gridsize_for_fft(FINUFFT_PLAN p);
void do_fft(FINUFFT_PLAN p);

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H

#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#include <vector>
#ifdef FINUFFT_USE_DUCC0
#include "ducc0/fft/fftnd_impl.h"
// temporary hack since some tests call these unconditionally
static inline void FFTW_FORGET_WISDOM() {}
static inline void FFTW_CLEANUP() {}
static inline void FFTW_CLEANUP_THREADS() {}
#else
#include "fftw_defs.h"
#endif
#include <finufft/defs.h>

std::vector<int> gridsize_for_fft(FINUFFT_PLAN p);
void do_fft(FINUFFT_PLAN p, CPX *fwBatch);

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H

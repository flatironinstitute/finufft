#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#ifdef FINUFFT_USE_DUCC0
#include "ducc0/fft/fftnd_impl.h"
#define FFTW_FORGET_WISDOM()   // temporary hack since some tests call this unconditionally
#define FFTW_CLEANUP()         // temporary hack since some tests call this unconditionally
#define FFTW_CLEANUP_THREADS() // temporary hack since some tests call this
                               // unconditionally
#else
#include "fftw_defs.h"
#endif
#include <finufft/defs.h>

int *gridsize_for_fft(FINUFFT_PLAN p);
void do_fft(FINUFFT_PLAN p);

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H

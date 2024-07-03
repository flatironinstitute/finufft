#ifndef FINUFFT_INCLUDE_FINUFFT_FFT_H
#define FINUFFT_INCLUDE_FINUFFT_FFT_H

#include <finufft/defs.h>
#ifdef FINUFFT_USE_DUCC0
#include "ducc0/fft/fftnd_impl.h"
#define FFTW_FORGET_WISDOM()   // temporary hack since some tests call this unconditionally
#define FFTW_CLEANUP()         // temporary hack since some tests call this unconditionally
#define FFTW_CLEANUP_THREADS() // temporary hack since some tests call this
                               // unconditionally
#else
#include "fftw_defs.h"
#endif

int *gridsize_for_fft(FINUFFT_PLAN p, CPX *fwBatch);
void do_fft (FINUFFT_PLAN p, CPX *fwBatch);

#endif // FINUFFT_INCLUDE_FINUFFT_FFT_H

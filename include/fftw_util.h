#ifndef FFTW_UTIL_H
#define FFTW_UTIL_H

#include <finufft_plan.h>

#ifdef SINGLE
  #define GRIDSIZE_FOR_FFTW gridsize_for_fftwf
#else
  #define GRIDSIZE_FOR_FFTW gridsize_for_fftw
#endif

int* GRIDSIZE_FOR_FFTW(finufft_plan* p);

#endif

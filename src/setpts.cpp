#include <finufft/setpts.hpp>

// Explicit instantiation, selected by FINUFFT_SINGLE define.

#ifdef FINUFFT_SINGLE
using FLT = float;
#else
using FLT = double;
#endif

extern template int FINUFFT_PLAN_T<FLT>::init_grid_kerFT_FFT(); // instantiated in fft.cpp

template int FINUFFT_PLAN_T<FLT>::setpts(BIGINT nj, const FLT *xj, const FLT *yj,
                                         const FLT *zj, BIGINT nk, const FLT *s,
                                         const FLT *t, const FLT *u);

#ifndef FFTW_DEFS_H
#define FFTW_DEFS_H

// ---------- Precision-indep complex types, macros to FFTW -------------------

#include <fftw3.h>          // (after complex.h) needed so can typedef FFTW_CPX

#ifdef T

//returns 1 if float, 0 otherwise (double)
typedef fftwf_complex FFTW_CPX_float;           //  single-prec has fftwf_*
typedef fftwf_plan FFTW_PLAN_float;

typedef fftw_complex FFTW_CPX_double;           // double-prec has fftw_*
typedef fftw_plan FFTW_PLAN_double;


#define FFTW_INIT_float fftwf_init_threads
#define FFTW_INIT_double fftw_init_threads
#define FFTW_PLAN_TH_float fftwf_plan_with_nthreads
#define FFTW_PLAN_TH_double fftw_plan_with_nthreads
#define FFTW_ALLOC_RE_float fftwf_alloc_real
#define FFTW_ALLOC_RE_double fftw_alloc_real
#define FFTW_ALLOC_CPX_float  fftwf_alloc_complex
#define FFTW_ALLOC_CPX_double fftw_alloc_complex
#define FFTW_PLAN_1D_float fftwf_plan_dft_1d
#define FFTW_PLAN_1D_double fftw_plan_dft_1d
#define FTW_PLAN_2D_float fftwf_plan_dft_2d
#define FFTW_PLAN_2D_double fftw_plan_dft_2d
#define FFTW_PLAN_3D_float fftwf_plan_dft_3d
#define FFTW_PLAN_3D_double fftw_plan_dft_3d
#define FFTW_PLAN_MANY_DFT_float fftwf_plan_many_dft
#define FFTW_PLAN_MANY_DFT_double fftw_plan_many_dft
#define FFTW_EX_float fftwf_execute
#define FFTW_EX_double fftw_execute
#define FFTW_DE_float fftwf_destroy_plan
#define FFTW_DE_double fftw_destroy_plan
#define FFTW_FR_float fftwf_free
#define FFTW_FR_double fftw_free
#define FFTW_FORGET_WISDOM_float fftwf_forget_wisdom
#define FFTW_FOEGET_WISDOM_double fftw_forget_wisdom


#endif //Def T


#endif

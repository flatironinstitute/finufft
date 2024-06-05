// all FFTW-related private FINUFFT headers

#ifndef FFTW_DEFS_H
#define FFTW_DEFS_H

// Here we define typedefs and MACROS to switch between single and double
// precision library compilation, which need different FFTW command symbols.
// Barnett simplified via FFTWIFY, 6/7/22.

#include <fftw3.h> // (after complex.h) needed so can typedef FFTW_CPX

// precision-switching names for interfaces to FFTW...
#ifdef SINGLE
// macro to prepend fftw_ (for double) or fftwf_ (for single) to a string
// without a space. The 2nd level of indirection is needed for safety, see:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-token-pasting
#define FFTWIFY_UNSAFE(x) fftwf_##x
#else
#define FFTWIFY_UNSAFE(x) fftw_##x
#endif
#define FFTWIFY(x)         FFTWIFY_UNSAFE(x)
// now use this tool (note we replaced typedefs v<=2.0.4, in favor of macros):
#define FFTW_CPX           FFTWIFY(complex)
#define FFTW_PLAN          FFTWIFY(plan)
#define FFTW_ALLOC_RE      FFTWIFY(alloc_real)
#define FFTW_ALLOC_CPX     FFTWIFY(alloc_complex)
#define FFTW_PLAN_1D       FFTWIFY(plan_dft_1d)
#define FFTW_PLAN_2D       FFTWIFY(plan_dft_2d)
#define FFTW_PLAN_3D       FFTWIFY(plan_dft_3d)
#define FFTW_PLAN_MANY_DFT FFTWIFY(plan_many_dft)
#define FFTW_EX            FFTWIFY(execute)
#define FFTW_DE            FFTWIFY(destroy_plan)
#define FFTW_FR            FFTWIFY(free)
#define FFTW_FORGET_WISDOM FFTWIFY(forget_wisdom)
#define FFTW_CLEANUP       FFTWIFY(cleanup)
// the following OMP switch could be done in the src code instead...
#ifdef _OPENMP
#define FFTW_INIT            FFTWIFY(init_threads)
#define FFTW_PLAN_TH         FFTWIFY(plan_with_nthreads)
#define FFTW_CLEANUP_THREADS FFTWIFY(cleanup_threads)
#else
// no OMP (no fftw{f}_threads or _omp), need dummy fftw threads calls...
#define FFTW_INIT()
#define FFTW_PLAN_TH(x)
#define FFTW_CLEANUP_THREADS()
#endif

#endif // FFTW_DEFS_H

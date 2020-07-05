// Defines C++/C user interface to the FINUFFT plan structs in both precisions:
// it provides finufft_plan (double-prec) and finufftf_plan (single-prec).
// Barnett 7/5/20

// save whether SINGLE defined or not...
#ifdef SINGLE
#define WAS_SINGLE
#endif

#undef SINGLE
#include <finufft_plan_eitherprec.h>
#define SINGLE
#include <finufft_plan_eitherprec.h>
#undef SINGLE

// ... and reconstruct it. (We still clobber the unlikely WAS_SINGLE symbol)
#ifdef WAS_SINGLE
#define SINGLE
#endif

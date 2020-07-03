// Defines the C++/C user interface to FINUFFT library.

// It simply combines single and double precision headers, by flipping a flag
// in the main macros which are in finufft_eitherprec.h
// No usual #ifndef testing is needed; it's done in finufft_eitherprec.h
// Internal FINUFFT routines that are compiled separately for
// each precision should include finufft_eitherprec.h and not finufft.h

// Barnett 7/1/20

#undef SINGLE
#include <finufft_eitherprec.h>
#define SINGLE
#include <finufft_eitherprec.h>
#undef SINGLE

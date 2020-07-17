// Defines the C++/C user interface to FINUFFT library.

// It simply combines single and double precision headers, by flipping a flag
// in the main macros which are in cufinufft_eitherprec.h
// No usual #ifndef testing is needed; it's done in cufinufft_eitherprec.h
// Internal cufinufft routines that are compiled separately for
// each precision should include cufinufft_eitherprec.h directly, and not cufinufft.h.

#undef SINGLE
#include <cufinufft_eitherprec.h>
#define SINGLE
#include <cufinufft_eitherprec.h>
#undef SINGLE

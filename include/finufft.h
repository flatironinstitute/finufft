// Defines the public C++ and C compatible user interface to FINUFFT library.

// This contains both single and double precision user-facing commands.
// "macro-safe" rewrite, including the plan object, Barnett 5/21/22-6/7/22.
// They will clobber any prior macros starting FINUFFT*, so in the lib/test
// sources finufft.h must be included before defs.h

/* Devnotes.
   A) Two precisions done by including the "either precision" headers twice.
   No use of the private headers for lib/test/example compilation is made.

   B) Good ways to debug this header ---
   1) preprocessor output (gets the general idea the macros worked):
   cpp include/finufft.h -Iinclude
   cpp -dD include/finufft.h -Iinclude
   then https://gcc.gnu.org/onlinedocs/cpp/Preprocessor-Output.html
   2) compile examples in both precs and C/C++, needed to catch typos:
   g++ examples/simple1d1.cpp -Iinclude -c
   g++ examples/simple1d1f.cpp -Iinclude -c
   gcc examples/simple1d1c.c -Iinclude -c
   gcc examples/simple1d1cf.c -Iinclude -c
*/

#ifndef FINUFFT_H
#define FINUFFT_H

// prec-indep stuff. both these are thus made public-facing
#include <finufft_opts.h>
#include <finufft_spread_opts.h>

// Public error numbers
#include <finufft_errors.h>

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>
#define FINUFFT_BIGINT int64_t

#ifndef __cplusplus
#include <stdbool.h> // for bool type in C (needed for item in plan struct)
#endif

// this macro name has to be safe since exposed to user
#define FINUFFT_SINGLE
#include <finufft_eitherprec.h>
#undef FINUFFT_SINGLE
// do it again for double-prec...
#include <finufft_eitherprec.h>

// clean up any purely local defs that are not in finufft_eitherprec.h...
#undef FINUFFT_BIGINT

#endif // FINUFFT_H

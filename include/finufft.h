// Defines the public C-compatible C++/C user interface to FINUFFT library.

// This contains both single and double precision user-facing commands.
// This is achieved by including the "either precision" headers twice.
// Barnett 5/21/22

#ifndef FINUFFT_H
#define FINUFFT_H

// prec-indep stuff. both these are thus made public-facing
#include <finufft_opts.h>
#include <finufft_spread_opts.h>

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>
#define FINUFFT_BIGINT int64_t

#ifndef __cplusplus
#include <stdbool.h>     // for bool type in C (needed for item in plan struct)
#endif

// this macro name has to be safe since exposed to user
#define FINUFFT_SINGLE
#include <finufft/finufft_eitherprec.h>
#undef FINUFFT_SINGLE
// do it again for double-prec...
#include <finufft/finufft_eitherprec.h>

// clean up any purely local defs that are not in finufft_eitherprec.h...
#undef FINUFFT_BIGINT

#endif  // FINUFFT_H

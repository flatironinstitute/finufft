#ifndef FINUFFT_J_H
#define FINUFFT_J_H

#include <dataTypes.h>
#include <finufft.h>

// This defines functions used in Julia interface

/* Note our typedefs:
   FLT = double (or float, depending on compilation precision)
   CPX = double complex (or float complex, depending on compilation precision)
   BIGINT = int64
   Make sure you call this library with matching Julia types
*/

// --------------- plan opaque pointer ------------
finufft_plan* finufft_plan_alloc();
void finufft_plan_free(finufft_plan* plan);

#endif  // FINUFFT_J_H

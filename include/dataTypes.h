#if (!defined(DATATYPES_H) && !defined(SINGLE)) || (!defined(DATATYPESF_H) && defined(SINGLE))
// Make sure we only include once per precision (see finufft_eitherprec.h).
#ifndef SINGLE
#define DATATYPES_H
#else
#define DATATYPESF_H
#endif


// ------------ FINUFFT data type definitions ----------------------------------

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday...
typedef int64_t BIGINT;

// decide which kind of complex numbers to use in interface...
#ifdef __cplusplus
#include <complex>          // C++ type
#define COMPLEXIFY(X) std::complex<X>
#else
#include <complex.h>        // C99 type
#define COMPLEXIFY(X) X complex
#endif

#undef FLT
#undef CPX

// Precision-independent real and complex types for interfacing...
#ifdef SINGLE
  #define FLT float
#else
  #define FLT double
#endif

#define CPX COMPLEXIFY(FLT)

#endif  // DATATYPE_H or DATATYPEF_H

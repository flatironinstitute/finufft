#ifndef DATATYPE_H
#define DATATYPE_H

// ------------ FINUFFT data type definitions ----------------------------------

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday...
typedef int64_t BIGINT;

// decide which kind of complex numbers to use in interface...
#ifdef __cplusplus
#include <complex>          // C++ type
#else
#include <complex.h>        // C99 type
#endif

// Precision-independent real and complex types for interfacing...
#ifdef SINGLE
  typedef float FLT;
  #ifdef __cplusplus
    typedef std::complex<float> CPX;
  #else
    typedef float complex CPX;
  #endif
#else
  typedef double FLT;
  #ifdef __cplusplus
    typedef std::complex<double> CPX;
  #else
    typedef double complex CPX;
  #endif
#endif

#endif  // DATATYPE_H

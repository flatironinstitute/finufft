#ifndef DATATYPE_H
#define DATATYPE_H

// ----------------- data type definitions ----------------------------------
// (note: non-interface precision- and omp-dependent defs are in defs.h)

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


#ifdef T
#ifdef __cplusplus
typedef std::complex<float> CPX_float;
typedef std::complex<double> CPX_double;
#else
typedef float complex CPX_float;
typedef double complex CPX_double;
#endif
#endif
//hhmmm what to do here? 
//#define EPSILON (float)6e-08
//#define FABS(x) fabsf(x)

#define EPSILON (double)1.1e-16
#define FABS(x) fabs(x)



#endif

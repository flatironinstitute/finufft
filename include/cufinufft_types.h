// ------------ FINUFFT data type definitions ----------------------------------

#if (!defined(DATATYPES_H) && !defined(CUFINUFFT_SINGLE)) || (!defined(DATATYPESF_H) && defined(CUFINUFFT_SINGLE))
// Make sure we only include once per precision (as in finufft_eitherprec.h).
#ifndef CUFINUFFT_SINGLE
#define DATATYPES_H
#else
#define DATATYPESF_H
#endif

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <complex>
#include <cstdint>

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday...
// Note: CUFINUFFT_BIGINT is modified to have ``int'' data type for cufinufft.
typedef int CUFINUFFT_BIGINT;

// decide which kind of complex numbers to use in interface...
#ifdef __cplusplus
#include <complex> // C++ type
#define COMPLEXIFY(X) std::complex<X>
#else
#include <complex.h> // C99 type
#define COMPLEXIFY(X) X complex
#endif

#undef CUFINUFFT_FLT
#undef CUFINUFFT_CPX

// Precision-independent real and complex types for interfacing...
// (note these cannot be typedefs since we want dual-precision library)
#ifdef CUFINUFFT_SINGLE
#define CUFINUFFT_FLT float
#else
#define CUFINUFFT_FLT double
#endif

#define CUFINUFFT_CPX COMPLEXIFY(CUFINUFFT_FLT)
typedef std::complex<double> dcomplex; // slightly sneaky since duplicated by mwrap

#undef EPSILON
#undef IMA
#undef FABS
#undef CUCPX
#undef CUFFT_TYPE
#undef CUFFT_EX
#undef SET_NF_TYPE12

// Compile-flag choice of single or double (default) precision:
// (Note in the other codes, CUFINUFFT_FLT is "double" or "float", CUFINUFFT_CPX same but complex)
#ifdef CUFINUFFT_SINGLE
// machine epsilon for rounding
#define EPSILON (float)6e-08
#define IMA std::complex<float>(0.0, 1.0)
#define FABS(x) fabs(x)
#define CUCPX cuFloatComplex
#define CUFFT_TYPE CUFFT_C2C
#define CUFFT_EX cufftExecC2C
#define SET_NF_TYPE12 set_nf_type12f
#else
// machine epsilon for rounding
#define EPSILON (double)1.1e-16
#define IMA std::complex<double>(0.0, 1.0)
#define FABS(x) fabsf(x)
#define CUCPX cuDoubleComplex
#define CUFFT_TYPE CUFFT_Z2Z
#define CUFFT_EX cufftExecZ2Z
#define SET_NF_TYPE12 set_nf_type12
#endif

#endif // DATATYPES_H or DATATYPESF_H

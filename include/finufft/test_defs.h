// test-wide definitions and headers for use in ../test/ and ../perftest/
// Private to library; not for user use.
// These switch precision based on if SINGLE is defined.

#ifndef TEST_DEFS_H
#define TEST_DEFS_H

// TESTER SETTINGS...
// how big a problem to check direct DFT for in 1D...
#define TEST_BIGPROB   1e8
// for omp rand filling
#define TEST_RANDCHUNK 1000000

// the public interface
#include <finufft.h>

// convenient private finufft internals that tests need
#include <finufft/finufft_core.h>
#include <finufft/finufft_utils.hpp>
#include <memory>

// --------------- Private data types for compilation in either prec ---------
// Devnote: must match those in relevant prec of public finufft.h interface!

// All indexing in library that potentially can exceed 2^31 uses 64-bit signed.
// This includes all calling arguments (eg M,N) that could be huge someday.

// Precision-independent real and complex types, for private lib/test compile
#ifdef SINGLE
using FLT = float;
#else
using FLT = double;
#endif
#include <complex> // we define C++ complex type only
using CPX = std::complex<FLT>;

// -------------- Math consts (not in math.h) and useful math macros ----------
#include <cmath>

// either-precision unit imaginary number...
#define IMA (CPX(0.0, 1.0))

// Random numbers: crappy unif random number generator in [0,1).
// These macros should probably be replaced by modern C++ std lib or random123.
// (RAND_MAX is in stdlib.h)
#include <cstdlib>
static inline FLT rand01 [[maybe_unused]] () { return FLT(rand()) / FLT(RAND_MAX); }
// unif[-1,1]:
static inline FLT randm11 [[maybe_unused]] () { return 2 * rand01() - FLT(1); }
// complex unif[-1,1] for Re and Im:
static inline CPX crandm11 [[maybe_unused]] () { return randm11() + IMA * randm11(); }

// Thread-safe seed-carrying versions of above (x is ptr to seed)...
// MR: we have to leave those as macros for now, as "rand_r" is deprecated
// and apparently no longer available on Windows.
#if 1
#define rand01r(x)   ((FLT)rand_r(x) / (FLT)RAND_MAX)
// unif[-1,1]:
#define randm11r(x)  (2 * rand01r(x) - (FLT)1.0)
// complex unif[-1,1] for Re and Im:
#define crandm11r(x) (randm11r(x) + IMA * randm11r(x))
#else
static inline FLT rand01r [[maybe_unused]] (unsigned int *x) {
  return FLT(rand_r(x)) / FLT(RAND_MAX);
}
// unif[-1,1]:
static inline FLT randm11r [[maybe_unused]] (unsigned int *x) {
  return 2 * rand01r(x) - FLT(1);
}
// complex unif[-1,1] for Re and Im:
static inline CPX crandm11r [[maybe_unused]] (unsigned int *x) {
  return randm11r(x) + IMA * randm11r(x);
}
#endif

// Prec-switching name macros (respond to SINGLE), used in lib & test sources
// and the plan object below.
// Note: crucially, these are now indep of macros used to gen public finufft.h!
#ifdef SINGLE
// a macro to prepend finufft or finufftf to a string without a space.
// The 2nd level of indirection is needed for safety, see:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-token-pasting
#define FINUFFTIFY_UNSAFE(x) finufftf##x
#else
#define FINUFFTIFY_UNSAFE(x) finufft##x
#endif
#define FINUFFTIFY(x)        FINUFFTIFY_UNSAFE(x)
// Now use the above tool to set up 2020-style macros used in tester source...
#define FINUFFT_PLAN         FINUFFTIFY(_plan)
#define FINUFFT_PLAN_S       FINUFFTIFY(_plan_s)
#define FINUFFT_DEFAULT_OPTS FINUFFTIFY(_default_opts)
#define FINUFFT_MAKEPLAN     FINUFFTIFY(_makeplan)
#define FINUFFT_SETPTS       FINUFFTIFY(_setpts)
#define FINUFFT_EXECUTE      FINUFFTIFY(_execute)
#define FINUFFT_DESTROY      FINUFFTIFY(_destroy)
#define FINUFFT1D1           FINUFFTIFY(1d1)
#define FINUFFT1D2           FINUFFTIFY(1d2)
#define FINUFFT1D3           FINUFFTIFY(1d3)
#define FINUFFT2D1           FINUFFTIFY(2d1)
#define FINUFFT2D2           FINUFFTIFY(2d2)
#define FINUFFT2D3           FINUFFTIFY(2d3)
#define FINUFFT3D1           FINUFFTIFY(3d1)
#define FINUFFT3D2           FINUFFTIFY(3d2)
#define FINUFFT3D3           FINUFFTIFY(3d3)
#define FINUFFT1D1MANY       FINUFFTIFY(1d1many)
#define FINUFFT1D2MANY       FINUFFTIFY(1d2many)
#define FINUFFT1D3MANY       FINUFFTIFY(1d3many)
#define FINUFFT2D1MANY       FINUFFTIFY(2d1many)
#define FINUFFT2D2MANY       FINUFFTIFY(2d2many)
#define FINUFFT2D3MANY       FINUFFTIFY(2d3many)
#define FINUFFT3D1MANY       FINUFFTIFY(3d1many)
#define FINUFFT3D2MANY       FINUFFTIFY(3d2many)
#define FINUFFT3D3MANY       FINUFFTIFY(3d3many)

// --------  FINUFFT's plan object, prec-switching version ------------------
// NB: now private (the public C++ or C etc user sees an opaque pointer to it)

#include <finufft/fft.h>
struct FINUFFT_PLAN_S : public FINUFFT_PLAN_T<FLT> {};

// std stuff for tester src
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#endif // TEST_DEFS_H

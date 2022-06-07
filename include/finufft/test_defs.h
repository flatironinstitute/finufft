// test-wide definitions and headers for use in ../test/ and ../perftest/
// Private to library; not for user use.
// These switch precision based on if SINGLE is defined.

#ifndef TEST_DEFS_H
#define TEST_DEFS_H

// TESTER SETTINGS...
// how big a problem to check direct DFT for in 1D...
#define TEST_BIGPROB 1e8
// for omp rand filling
#define TEST_RANDCHUNK 1000000


// the public interface
#include <finufft.h>

// convenient private finufft internals
#include <finufft/utils.h>
#include <finufft/utils_precindep.h>
// prec-switching (SINGLE) to set up FLT, CPX, BIGINT, etc...
#include <finufft/defs.h>

// tester prec-switching macros (responds to SINGLE), in test and perftest only
#ifdef SINGLE
// macro to prepend finufft or finufftf to a string without a space.
// The 2nd level of indirection is needed for safety, see:
// https://isocpp.org/wiki/faq/misc-technical-issues#macros-with-token-pasting
#define FINUFFTIFY_UNSAFE(x) finufftf##x
#else
#define FINUFFTIFY_UNSAFE(x) finufft##x
#endif
#define FINUFFTIFY(x) FINUFFTIFY_UNSAFE(x)
// The following set up 2020-style macros needed for testers...
#define FINUFFT_PLAN FINUFFTIFY(_plan)
#define FINUFFT_DEFAULT_OPTS FINUFFTIFY(_default_opts)
#define FINUFFT_MAKEPLAN FINUFFTIFY(_makeplan)
#define FINUFFT_SETPTS FINUFFTIFY(_setpts)
#define FINUFFT_EXEC FINUFFTIFY(_execute)
#define FINUFFT_DESTROY FINUFFTIFY(_destroy)
#define FINUFFT1D1 FINUFFTIFY(1d1)
#define FINUFFT1D2 FINUFFTIFY(1d2)
#define FINUFFT1D3 FINUFFTIFY(1d3)
#define FINUFFT2D1 FINUFFTIFY(2d1)
#define FINUFFT2D2 FINUFFTIFY(2d2)
#define FINUFFT2D3 FINUFFTIFY(2d3)
#define FINUFFT3D1 FINUFFTIFY(3d1)
#define FINUFFT3D2 FINUFFTIFY(3d2)
#define FINUFFT3D3 FINUFFTIFY(3d3)
#define FINUFFT1D1MANY FINUFFTIFY(1d1many)
#define FINUFFT1D2MANY FINUFFTIFY(1d2many)
#define FINUFFT1D3MANY FINUFFTIFY(1d3many)
#define FINUFFT2D1MANY FINUFFTIFY(2d1many)
#define FINUFFT2D2MANY FINUFFTIFY(2d2many)
#define FINUFFT2D3MANY FINUFFTIFY(2d3many)
#define FINUFFT3D1MANY FINUFFTIFY(3d1many)
#define FINUFFT3D2MANY FINUFFTIFY(3d2many)
#define FINUFFT3D3MANY FINUFFTIFY(3d3many)

// std stuff for tester src
#include <math.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>

#endif   // TEST_DEFS_H

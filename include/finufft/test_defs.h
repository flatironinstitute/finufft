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
// prec-switching (SINGLE) to set up FLT, CPX, BIGINT, FINUFFT1D1, etc...
#include <finufft/defs.h>
// since "many" tests need direct access to FFTW commands...
#include <finufft/fftw_defs.h>

// std stuff for tester src
#include <math.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>

#endif   // TEST_DEFS_H

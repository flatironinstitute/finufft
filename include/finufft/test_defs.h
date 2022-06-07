// test-wide definitions and headers for use in ../test/ and ../perftest/
// Private to library; not for user use.
// These switch precision based on if SINGLE is defined.

#ifndef TEST_DEFS_H
#define TEST_DEFS_H

// convenient finufft internals
#include <finufft/utils.h>
#include <finufft/utils_precindep.h>
#include <finufft/defs.h>

// responds to SINGLE, and defines FINUFFT?D? used in test/*.cpp
// *** to do


// std stuff
#include <math.h>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>

// how big a problem to check direct DFT for in 1D...
#define TEST_BIGPROB 1e8

// for omp rand filling
#define TEST_RANDCHUNK 1000000

#endif   // TEST_DEFS_H

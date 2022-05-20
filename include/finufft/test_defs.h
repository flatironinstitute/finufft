// test-wide definitions and headers for use in ../test/*.cpp and ../perftest

#ifndef TEST_DEFS_H
#define TEST_DEFS_H

// responds to SINGLE, and defines FINUFFT?D? used in test/*.cpp
#include <finufft_eitherprec.h>

// convenient finufft internals
#include <utils.h>
#include <utils_precindep.h>
#include <defs.h>

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

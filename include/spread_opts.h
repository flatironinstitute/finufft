#ifndef SPREAD_OPTS_H
#define SPREAD_OPTS_H

#define FINUFFT_FLT float
#define SPREAD_OPTS spreads_optsf
#include "spread_opts.inl"
#undef FINUFFT_FLT
#undef SPREAD_OPTS

#define FINUFFT_FLT double
#define SPREAD_OPTS spreads_optsd
#include "spread_opts.inl"
#undef FINUFFT_FLT
#undef SPREAD_OPTS

#include <dataTypes.h>

#ifdef SINGLE
#define SPREAD_OPTS spreads_optsf
#else
#define SPREAD_OPTS spreads_optsd
#endif

#endif   // SPREAD_OPTS_H

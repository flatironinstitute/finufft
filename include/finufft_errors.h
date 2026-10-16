#ifndef FINUFFT_ERRORS_H
#define FINUFFT_ERRORS_H

#include <finufft_common/defines.h>

// ---------- Global error/warning output codes for the library ---------------
// Documentation for these codes is also maintained here (between sphinx tags)
// clang-format off
enum {
  // sphinx tag (don't remove): @doc_start
  FINUFFT_ERR_MAXNALLOC              = 2,
      // attempt to internal array size >MAX_NF (defined in plan.hpp)
  FINUFFT_ERR_SPREAD_BOX_SMALL       = 3,
      // spreader: fine grid too small compared to spread (kernel) width
  FINUFFT_ERR_SPREAD_DIR             = 6,
      // spreader: illegal direction (should be 1 or 2)
  FINUFFT_ERR_UPSAMPFAC_TOO_SMALL    = 7,
      // upsampling factor sigma too small (should be 0.0 for auto, else >1.0)
  FINUFFT_ERR_HORNER_WRONG_BETA      = 8,
      // upsampling factor not a value with known Horner poly eval rule,
      // raised only by the GPU library only with opts.gpu_kerevalmeth=1.
  FINUFFT_ERR_NTRANS_NOTVALID        = 9,
      // ntrans invalid in "many" (vector) or guru interface (should be >= 1)
  FINUFFT_ERR_TYPE_NOTVALID          = 10,
      // transform type invalid (must be 1, 2, or 3)
  FINUFFT_ERR_ALLOC                  = 11,
      // general internal memory allocation failure
  FINUFFT_ERR_DIM_NOTVALID           = 12,
      // dimension invalid (must be 1, 2, or 3)
  FINUFFT_ERR_NDATA_NOTVALID         = 14,
      // invalid M or mode array size (>2^31 modes, a dimension has 0 modes, etc).
      // Used by the GPU library only.
  FINUFFT_ERR_CUDA_FAILURE           = 15,
      // CUDA failure (failure to call any cuda function/kernel, malloc/memset, etc)
  FINUFFT_ERR_PLAN_NOTVALID          = 16,
      // attempt to destroy an uninitialized plan
  FINUFFT_ERR_METHOD_NOTVALID        = 17,
      // invalid spread/interp method for dim (attempt to blockgather in 1D, e.g.).
      // Used by GPU library only.
  FINUFFT_ERR_BINSIZE_NOTVALID       = 18,
      // size of bins for subproblem/blockgather invalid, in GPU libary
  FINUFFT_ERR_INSUFFICIENT_SHMEM     = 19,
      // GPU shmem too small for subprob/blockgather parameters
  FINUFFT_ERR_NUM_NU_PTS_INVALID     = 20,
      // invalid number of nonuniform points: nj or nk <0, or too big (see plan.hpp)
  FINUFFT_ERR_INVALID_ARGUMENT       = 21,
      // invalid input argument not covered by other errors
  FINUFFT_ERR_LOCK_FUNS_INVALID      = 22,
      // invalid FFTW lock function
  FINUFFT_ERR_NTHREADS_NOTVALID      = 23,
      // nthreads option invalid
  FINUFFT_ERR_KERFORMULA_NOTVALID    = 24,
      // spread kernel formula type invalid
  FINUFFT_ERR_UNKNOWN_EXCEPTION      = 25,
      // unknown exception caught
  FINUFFT_ERR_EPS_TOO_SMALL          = 26,
      // requested tolerance (eps) deemed too small to achieve given problem parameters
      // in working precision, due to rounding errors or condition number of problem.
      // See Troubleshooting section of docs. If you know what you are doing, disable
      // the raising of this by setting opts.allow_eps_too_small=1. A warning is still
      // given.
  FINUFFT_ERR_PSWF_SETUP             = 27,
      // iteration inside the setup code for the PSWF function evaluator failed
      // to converge.
  // sphinx tag (don't remove): @doc_end

  // Deprecated error codes (no longer returned by CPU or GPU) go here, kept for ABI:

  FINUFFT_WARN_EPS_TOO_SMALL FINUFFT_DEPRECATED_ENUM(
      "retired: use FINUFFT_ERR_EPS_TOO_SMALL instead") = 1,
  // Note: a tol below eps_mach or estimated achievable error for the problem size
  // now warns on stderr or raises FINUFFT_ERR_EPS_TOO_SMALL.

  FINUFFT_ERR_SPREAD_PTS_OUT_RANGE FINUFFT_DEPRECATED_ENUM(
      "retired: bounds checking was removed in v2.3.0") = 4,

  FINUFFT_ERR_SPREAD_ALLOC FINUFFT_DEPRECATED_ENUM(
      "retired: allocation error, but was never returned by the library anyway") = 5,

  FINUFFT_ERR_SPREAD_THREAD_NOTVALID FINUFFT_DEPRECATED_ENUM(
      "retired: opts.spread_thread was deprecated in v2.6.0") = 13,

};
// clang-format on
#endif

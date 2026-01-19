#ifndef FINUFFT_SPREAD_OPTS_H
#define FINUFFT_SPREAD_OPTS_H

/* C-compatible options struct for spread/interpolation within FINUFFT

  Notes: 1) as of v2.5, no longer part of public-facing API (it was already a
  private member of the plan class.
  2) Deliberately uses fixed types (no templated precision-switching), although
  now the plan is a class, this would be fine.
  3) `flags` (TF_OMIT_*) timing-only flags have been purged.
  4) ES kernel fields replaced by generic shape param beta, and kerformula type
*/

/* clang-format off */
typedef struct finufft_spread_opts {
  // See finufft_core:setup_spreadinterp() for where most of these are set:
  int nspread;             // w, the kernel width in grid pts
  int spread_direction;    // 1 means spread NU->U, 2 means interpolate U->NU
  int sort;                // 0: don't sort NU pts, 1: do, 2: heuristic choice
  int kerevalmeth;         // kept for ABI compatibility, ignored (Horner is always used)
  int kerpad;              // kept for ABI compatibility, ignored (direct eval removed)
  int nthreads;            // # threads for spreadinterp (0: use max avail)
  int sort_threads;        // # threads for sort (0: auto-choice up to nthreads)
  int max_subproblem_size; // # pts per t1 subprob; sets extra RAM per thread
  int debug;               // 0: silent, 1: small text output, 2: verbose
  int atomic_threshold;    // num threads before switching spreadSorted to using atomic ops
  double upsampfac;        // sigma, upsampling factor, >1.
  double beta;             // main kernel shape parameter (for prolate-like kernels)
  int kerformula;          // kernel function type; see finufft_common/kernel.h
} finufft_spread_opts;
/* clang-format on */

#endif // FINUFFT_SPREAD_OPTS_H

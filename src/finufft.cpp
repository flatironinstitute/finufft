// public header
#include <finufft.h>

// private headers for lib build
// (must come after finufft.h which clobbers FINUFFT* macros)
#include <finufft/defs.h>

void FINUFFT_DEFAULT_OPTS(finufft_opts *o) { finufft_default_opts_t(o); }

int FINUFFT_MAKEPLAN(int type, int dim, const BIGINT *n_modes, int iflag, int ntrans,
                     FLT tol, FINUFFT_PLAN *pp, finufft_opts *opts) {
  return finufft_makeplan_t<FLT>(type, dim, n_modes, iflag, ntrans, tol,
                                 reinterpret_cast<FINUFFT_PLAN_T<FLT> **>(pp), opts);
}

int FINUFFT_SETPTS(FINUFFT_PLAN p, BIGINT nj, FLT *xj, FLT *yj, FLT *zj, BIGINT nk,
                   FLT *s, FLT *t, FLT *u) {
  return finufft_setpts_t<FLT>(reinterpret_cast<FINUFFT_PLAN_T<FLT> *>(p), nj, xj, yj, zj,
                               nk, s, t, u);
}

int FINUFFT_EXECUTE(FINUFFT_PLAN p, CPX *cj, CPX *fk) {
  return finufft_execute_t<FLT>(reinterpret_cast<FINUFFT_PLAN_T<FLT> *>(p), cj, fk);
}

int FINUFFT_DESTROY(FINUFFT_PLAN p)
// Free everything we allocated inside of finufft_plan pointed to by p.
// Also must not crash if called immediately after finufft_makeplan.
// Thus either each thing free'd here is guaranteed to be NULL or correctly
// allocated.
{
  if (!p) // NULL ptr, so not a ptr to a plan, report error
    return 1;

  delete reinterpret_cast<FINUFFT_PLAN_T<FLT> *>(p);
  p = nullptr;
  return 0; // success
}

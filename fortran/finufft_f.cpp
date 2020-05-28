#include <utils.h>
#include <finufft_f.h>

// C++ layer for calling FINUFFT from fortran, in f77 style.
// Note the trailing underscore which is not present in the fortran name.
// Barnett 2/17/17. Single prec 4/5/17. Libin Lu & Alex Barnett 2020.

#ifdef __cplusplus
extern "C" {
#endif

// -------- helpers for guru ------------
void mkopts_mem(nufft_opts *o)
{
  o = (nufft_opts *)malloc(sizeof(nufft_opts));
}

void mkplan_mem(finufft_plan *plan)
{
  plan = (finufft_plan *)malloc(sizeof(finufft_plan));
}
  
// --------------------- guru interface ------------------------
void finufft_makeplan_(int *type, int *n_dims, BIGINT *n_modes, int *iflag, int *n_transf, FLT *tol, finufft_plan *plan, nufft_opts *o, int *ier)
{
  mkplan_mem(plan);
  *ier = finufft_makeplan(*type, *n_dims, n_modes, *iflag, *n_transf, *tol, plan, o);
}

void finufft_setpts_(finufft_plan *plan, BIGINT *M, FLT *xj, FLT *yj, FLT *zj, BIGINT *N, FLT *s, FLT *t, FLT *u, int *ier)
{
  *ier = finufft_setpts(plan, *M, xj, yj, zj, *N, s, t, u);
}

void finufft_exec_(finufft_plan *plan, CPX *weights, CPX *result, int *ier)
{
  *ier = finufft_exec(plan, weights, result);
}

void finufft_destroy_(finufft_plan *plan, int *ier)
{
  *ier = finufft_destroy(plan);
}

// --------------------- create a nufft_opts and set attributes ----------
void finufft_default_opts_(nufft_opts* o)
{
  mkopts_mem(o);
  finufft_default_opts(o);
}

void set_debug_(nufft_opts *o, int *debug)
{
  o->debug = *debug;
}

void set_spread_debug_(nufft_opts *o, int *spread_debug)
{
  o->spread_debug = *spread_debug;
}

void set_spread_kerevalmeth_(nufft_opts *o, int *spread_kerevalmeth)
{
  o->spread_kerevalmeth = *spread_kerevalmeth;
}

void set_spread_kerpad_(nufft_opts *o, int *spread_kerpad)
{
  o->spread_kerpad = *spread_kerpad;
}

void set_chkbnds_(nufft_opts *o, int *chkbnds)
{
  o->chkbnds = *chkbnds;
}

void set_fftw_(nufft_opts *o, int *fftw)
{
  o->fftw = *fftw;
}

void set_modeord_(nufft_opts *o, int *modeord)
{
  o->modeord = *modeord;
}

void set_upsampfac_(nufft_opts *o, FLT *upsampfac)
{
  o->upsampfac = *upsampfac;
}

void set_spread_thread_(nufft_opts *o, int *spread_thread)
{
  o->spread_thread = *spread_thread;
}

void set_maxbatchsize_(nufft_opts *o, int *maxbatchsize)
{
  o->maxbatchsize = *maxbatchsize;
}

// -------------- simple and many-vector interfaces --------------------
void finufft1d1_(BIGINT* nj,FLT* xj,CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = finufft1d1(*nj,xj,cj,*iflag,*eps,*ms,fk,o);
}

// *** 17 others to add...  :O





  
  
  
#ifdef __cplusplus
}
#endif

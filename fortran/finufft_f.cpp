#include <utils.h>
#include <finufft_f.h>

// wrappers for calling FINUFFT from fortran.
// Note the trailing underscore which is not present in the fortran name.
// Barnett 2/17/17. Single prec 4/5/17

// integer*4 for the sizes for now.

// All nufft_opts are default settings; this interface would need to change
// to allow control of them.
#ifdef __cplusplus
extern "C" {
#endif

void finufft1d1_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft1d1((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*ms,fk,&opts);
}

void finufft1d2_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft1d2((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*ms,fk,&opts);
}

void finufft1d3_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *nk, FLT* s, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  //opts.upsampfac=1.25;      // is recommended, for up to 9-digit prec
  *ier = finufft1d3((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*nk,s,fk,&opts);
}

void finufft2d1_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft2d1((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,&opts);
}

void finufft2d2_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft2d2((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,&opts);
}

void finufft2d3_f_(int *nj,FLT* xj,FLT* yj, CPX* cj,int *iflag,
		   FLT *eps, int *nk, FLT* s, FLT* t, CPX* fk,
		   int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  //opts.upsampfac=1.25;      // is recommended, for up to 9-digit prec
  *ier = finufft2d3((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*nk,s,t,fk,&opts);
}

void finufft3d1_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft3d1((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,(BIGINT)*mu,fk,&opts);
}

void finufft3d2_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft3d2((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,(BIGINT)*mu,fk,&opts);
}

void finufft3d3_f_(int *nj,FLT* xj,FLT* yj, FLT*zj, CPX* cj,
		   int *iflag, FLT *eps, int *nk, FLT* s, FLT* t,
		   FLT* u, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  //opts.upsampfac=1.25;      // is recommended, for up to 9-digit prec
  *ier = finufft3d3((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*nk,s,t,u,fk,&opts);
}

void finufft2d1many_f_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft2d1many(*ndata,(BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,&opts);
}

void finufft2d2many_f_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(&opts);
  *ier = finufft2d2many(*ndata,(BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,&opts);
}


  // -------- helpers ------------
void mkopts_mem(nufft_opts *o)
{
  o = (nufft_opts *)malloc(sizeof(nufft_opts));
}

void mkplan_mem(finufft_plan *plan)
{
  plan = (finufft_plan *)malloc(sizeof(finufft_plan));
}


  
// --------------------- guru interface ------------------------
void finufft_default_opts_(nufft_opts* o)
{
  mkopts_mem(o);
  finufft_default_opts(o);
}

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

// --------------------- set nufft_opts attributes -------------------
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

#ifdef __cplusplus
}
#endif

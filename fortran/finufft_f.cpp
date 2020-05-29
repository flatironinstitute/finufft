#include <utils.h>
#include <finufft_f.h>

// C++ layer for calling FINUFFT from fortran, in f77 style.
// Note the trailing underscore which is not present in the fortran name.
// Barnett 2/17/17. Single prec 4/5/17. Libin Lu & Alex Barnett 2020.

#ifdef __cplusplus
extern "C" {
#endif

// -------- helpers for guru ------------
/*
void mkopts_mem(nufft_opts* o)
{
  o = (nufft_opts *)malloc(sizeof(nufft_opts));
}

void mkplan_mem(finufft_plan* plan)
{
  plan = (finufft_plan *)malloc(sizeof(finufft_plan));
}
*/
  
// --------------------- guru interface ------------------------
void finufft_makeplan_(int *type, int *n_dims, BIGINT *n_modes, int *iflag, int *n_transf, FLT *tol, finufft_plan **plan, nufft_opts **o, int *ier)
{
  //mkplan_mem(plan);
  *plan = (finufft_plan *)malloc(sizeof(finufft_plan));
  *ier = finufft_makeplan(*type, *n_dims, n_modes, *iflag, *n_transf, *tol, *plan, (o?*o:NULL));
}

void finufft_setpts_(finufft_plan **plan, BIGINT *M, FLT *xj, FLT *yj, FLT *zj, BIGINT *N, FLT *s, FLT *t, FLT *u, int *ier)
{
  *ier = finufft_setpts(*plan, *M, xj, yj, zj, *N, s, t, u);
}

void finufft_exec_(finufft_plan **plan, CPX *weights, CPX *result, int *ier)
{
  *ier = finufft_exec(*plan, weights, result);
}

void finufft_destroy_(finufft_plan **plan, int *ier)
{
  *ier = finufft_destroy(*plan);
}

// --------------------- create a nufft_opts and set attributes ----------
void finufft_default_opts_(nufft_opts** o)
{
  //mkopts_mem(o);
  *o = (nufft_opts *)malloc(sizeof(nufft_opts));
  finufft_default_opts(*o);
}

void set_debug_(nufft_opts **o, int *debug)
{
  (*o)->debug = *debug;
}

void set_spread_debug_(nufft_opts **o, int *spread_debug)
{
  (*o)->spread_debug = *spread_debug;
}

void set_spread_kerevalmeth_(nufft_opts **o, int *spread_kerevalmeth)
{
  (*o)->spread_kerevalmeth = *spread_kerevalmeth;
}

void set_spread_kerpad_(nufft_opts **o, int *spread_kerpad)
{
  (*o)->spread_kerpad = *spread_kerpad;
}

void set_chkbnds_(nufft_opts **o, int *chkbnds)
{
  (*o)->chkbnds = *chkbnds;
}

void set_fftw_(nufft_opts **o, int *fftw)
{
  (*o)->fftw = *fftw;
}

void set_modeord_(nufft_opts **o, int *modeord)
{
  (*o)->modeord = *modeord;
}

void set_upsampfac_(nufft_opts **o, FLT *upsampfac)
{
  (*o)->upsampfac = *upsampfac;
}

void set_spread_thread_(nufft_opts **o, int *spread_thread)
{
  (*o)->spread_thread = *spread_thread;
}

void set_maxbatchsize_(nufft_opts **o, int *maxbatchsize)
{
  (*o)->maxbatchsize = *maxbatchsize;
}

// -------------- simple and many-vector interfaces --------------------
// --- 1D ---
void finufft1d1_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft1d1(*nj,xj,cj,*iflag,*eps,*ms,fk,(o?*o:NULL));
}

void finufft1d1many_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft1d1many(*ntransf,*nj,xj,cj,*iflag,*eps,*ms,fk,(o?*o:NULL));
}

void finufft1d2_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft1d2(*nj,xj,cj,*iflag,*eps,*ms,fk,(o?*o:NULL));
}

void finufft1d2many_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft1d2many(*ntransf,*nj,xj,cj,*iflag,*eps,*ms,fk,(o?*o:NULL));
}

void finufft1d3_(BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts** o, int* ier)
{
  *ier = finufft1d3(*nj,x,c,*iflag,*eps,*nk,s,f,(o?*o:NULL));
}

void finufft1d3many_(int* ntransf,
                 BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts** o, int* ier)
{
  *ier = finufft1d3many(*ntransf,*nj,x,c,*iflag,*eps,*nk,s,f,(o?*o:NULL));
}

// --- 2D ---
void finufft2d1_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft2d1(*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,(o?*o:NULL));
}
void finufft2d1many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft2d1many(*ntransf,*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,(o?*o:NULL));
}

void finufft2d2_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft2d2(*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,(o?*o:NULL));
}
void finufft2d2many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft2d2many(*ntransf,*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,(o?*o:NULL));
}

void finufft2d3_(BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts** o, int* ier)
{
  *ier = finufft2d3(*nj,x,y,c,*iflag,*eps,*nk,s,t,f,(o?*o:NULL));
}

void finufft2d3many_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts** o, int* ier)
{
  *ier = finufft2d3many(*ntransf,*nj,x,y,c,*iflag,*eps,*nk,s,t,f,(o?*o:NULL));
}

// --- 3D ---
void finufft3d1_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft3d1(*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,(o?*o:NULL));
}

void finufft3d1many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft3d1many(*ntransf,*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,(o?*o:NULL));
}

void finufft3d2_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft3d2(*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,(o?*o:NULL));
}

void finufft3d2many_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts** o, int* ier)
{
  *ier = finufft3d2many(*ntransf,*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,(o?*o:NULL));
}

void finufft3d3_(BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts** o, int* ier)
{
  *ier = finufft3d3(*nj,x,y,z,c,*iflag,*eps,*nk,s,t,u,f,(o?*o:NULL));
}

void finufft3d3many_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts** o, int* ier)
{
  *ier = finufft3d3many(*ntransf,*nj,x,y,z,c,*iflag,*eps,*nk,s,t,u,f,(o?*o:NULL));
}

  
#ifdef __cplusplus
}
#endif

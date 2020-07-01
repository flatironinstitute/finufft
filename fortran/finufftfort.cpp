#include <finufftfort.h>
#include <dataTypes.h>

/* C++ layer for calling FINUFFT from fortran, in f77 style + derived type
   for the nufft_opts C-struct. The ptr to finufft_plan is passed as an "opaque"
   pointer (as in FFTW3 legacy fortran interface).
   Note the trailing underscore name-mangle which is not needed from fortran.

   Note our typedefs:
   FLT = double (or float, depending on compilation precision)
   CPX = double complex (or float complex, depending on compilation precision)
   BIGINT = int64 (integer*8 in fortran)

   Make sure you call this library with matching fortran types

   For a demo see: examples/simple1d1.f

   Barnett 2/17/17. Single prec 4/5/17. Libin Lu & Alex Barnett, May 2020.
*/

#ifdef __cplusplus
extern "C" {
#endif
  
// --------------------- guru interface from fortran ------------------------
void FINUFFT_MAKEPLAN_(int *type, int *n_dims, BIGINT *n_modes, int *iflag, int *n_transf, FLT *tol, finufft_plan **plan, nufft_opts *o, int *ier)
{
  if (!plan)
    fprintf(stderr,"finufft_makeplan_ fortran interface: plan must be allocated as at least the size of a C pointer (usually 8 bytes)!\n");
  else {
    // note plan will be ptr to ptr to a nufft_opts. Must allocate the latter:
    *plan = (finufft_plan *)malloc(sizeof(finufft_plan));
    // pass o whether it's a NULL or pointer to a fortran-allocated nufft_opts:
    *ier = FINUFFT_MAKEPLAN(*type, *n_dims, n_modes, *iflag, *n_transf, *tol, *plan, o);
  }
}

void FINUFFT_SETPTS_(finufft_plan **plan, BIGINT *M, FLT *xj, FLT *yj, FLT *zj, BIGINT *nk, FLT *s, FLT *t, FLT *u, int *ier)
{
  if (!*plan) {
    fprintf(stderr,"finufft_setpts_ fortran: finufft_plan unallocated!");
    return;
  }
  int nk_safe = 0;           // catches the case where user passes NULL in
  if (nk)
    nk_safe = *nk;
  *ier = FINUFFT_SETPTS(*plan, *M, xj, yj, zj, nk_safe, s, t, u);
}

void FINUFFT_EXEC_(finufft_plan **plan, CPX *weights, CPX *result, int *ier)
{
  if (!*plan)
    fprintf(stderr,"finufft_exec_ fortran: finufft_plan unallocated!");
  else
    *ier = FINUFFT_EXEC(*plan, weights, result);
}

void FINUFFT_DESTROY_(finufft_plan **plan, int *ier)
{
  if (!*plan)
    fprintf(stderr,"finufft_destroy_ fortran: finufft_plan unallocated!");
  else
    *ier = FINUFFT_DESTROY(*plan);
}

  
// ------------ use FINUFFT to set the default options ---------------------
// (Note the nufft_opts is created in f90-style derived types, not here)
void FINUFFT_DEFAULT_OPTS_(nufft_opts* o)
{
  if (!o)
    fprintf(stderr,"finufft_default_opts_ fortran: opts must be allocated!\n");
  else
    // o is a ptr to already-allocated fortran nufft_opts derived type...
    FINUFFT_DEFAULT_OPTS(o);
}


// -------------- simple and many-vector interfaces --------------------
// --- 1D ---
void FINUFFT1D1_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT1D1(*nj,xj,cj,*iflag,*eps,*ms,fk,o);
}

void FINUFFT1D1MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT1D1MANY(*ntransf,*nj,xj,cj,*iflag,*eps,*ms,fk,o);
}

void FINUFFT1D2_(BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT1D2(*nj,xj,cj,*iflag,*eps,*ms,fk,o);
}

void FINUFFT1D2MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT1D2MANY(*ntransf,*nj,xj,cj,*iflag,*eps,*ms,fk,o);
}

void FINUFFT1D3_(BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts* o, int* ier)
{
  *ier = FINUFFT1D3(*nj,x,c,*iflag,*eps,*nk,s,f,o);
}

void FINUFFT1D3MANY_(int* ntransf,
                 BIGINT* nj, FLT* x, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, CPX* f, nufft_opts* o, int* ier)
{
  *ier = FINUFFT1D3MANY(*ntransf,*nj,x,c,*iflag,*eps,*nk,s,f,o);
}

// --- 2D ---
void FINUFFT2D1_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT2D1(*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,o);
}
void FINUFFT2D1MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT2D1MANY(*ntransf,*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,o);
}

void FINUFFT2D2_(BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT2D2(*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,o);
}
void FINUFFT2D2MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT2D2MANY(*ntransf,*nj,xj,yj,cj,*iflag,*eps,*ms,*mt,fk,o);
}

void FINUFFT2D3_(BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts* o, int* ier)
{
  *ier = FINUFFT2D3(*nj,x,y,c,*iflag,*eps,*nk,s,t,f,o);
}

void FINUFFT2D3MANY_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, CPX* f, nufft_opts* o, int* ier)
{
  *ier = FINUFFT2D3MANY(*ntransf,*nj,x,y,c,*iflag,*eps,*nk,s,t,f,o);
}

// --- 3D ---
void FINUFFT3D1_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT3D1(*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,o);
}

void FINUFFT3D1MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT3D1MANY(*ntransf,*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,o);
}

void FINUFFT3D2_(BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT3D2(*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,o);
}

void FINUFFT3D2MANY_(int* ntransf,
                 BIGINT* nj, FLT* xj, FLT* yj, FLT* zj, CPX* cj, int* iflag, FLT* eps,
                 BIGINT* ms, BIGINT* mt, BIGINT* mu, CPX* fk, nufft_opts* o, int* ier)
{
  *ier = FINUFFT3D2MANY(*ntransf,*nj,xj,yj,zj,cj,*iflag,*eps,*ms,*mt,*mu,fk,o);
}

void FINUFFT3D3_(BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts* o, int* ier)
{
  *ier = FINUFFT3D3(*nj,x,y,z,c,*iflag,*eps,*nk,s,t,u,f,o);
}

void FINUFFT3D3MANY_(int* ntransf,
                 BIGINT* nj, FLT* x, FLT* y, FLT* z, CPX* c, int* iflag, FLT* eps,
                 BIGINT* nk, FLT* s, FLT* t, FLT* u, CPX* f, nufft_opts* o, int* ier)
{
  *ier = FINUFFT3D3MANY(*ntransf,*nj,x,y,z,c,*iflag,*eps,*nk,s,t,u,f,o);
}

  
#ifdef __cplusplus
}
#endif

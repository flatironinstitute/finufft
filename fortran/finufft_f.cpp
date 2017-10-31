#include "../src/utils.h"
#include "finufft_f.h"
#include "../src/finufft.h"

// wrappers for calling FINUFFT from fortran.
// Note the trailing underscore which is not present in the fortran name.
// Barnett 2/17/17. Single prec 4/5/17

// integer*4 for the sizes for now.

// All nufft_opts are default settings; this interface would need to change
// to allow control of them.

void finufft1d1_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft1d1((INT)*nj,xj,cj,*iflag,*eps,(INT)*ms,fk,opts);
}

void finufft1d2_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft1d2((INT)*nj,xj,cj,*iflag,*eps,(INT)*ms,fk,opts);
}

void finufft1d3_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *nk, FLT* s, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft1d3((INT)*nj,xj,cj,*iflag,*eps,(INT)*nk,s,fk,opts);
}

void finufft2d1_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft2d1((INT)*nj,xj,yj,cj,*iflag,*eps,(INT)*ms,(INT)*mt,fk,opts);
}

void finufft2d2_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft2d2((INT)*nj,xj,yj,cj,*iflag,*eps,(INT)*ms,(INT)*mt,fk,opts);
}

void finufft2d3_f_(int *nj,FLT* xj,FLT* yj, CPX* cj,int *iflag,
		   FLT *eps, int *nk, FLT* s, FLT* t, CPX* fk,
		   int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft2d3((INT)*nj,xj,yj,cj,*iflag,*eps,(INT)*nk,s,t,fk,opts);
}

void finufft3d1_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft3d1((INT)*nj,xj,yj,zj,cj,*iflag,*eps,(INT)*ms,(INT)*mt,(INT)*mu,fk,opts);
}

void finufft3d2_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft3d2((INT)*nj,xj,yj,zj,cj,*iflag,*eps,(INT)*ms,(INT)*mt,(INT)*mu,fk,opts);
}

void finufft3d3_f_(int *nj,FLT* xj,FLT* yj, FLT*zj, CPX* cj,
		   int *iflag, FLT *eps, int *nk, FLT* s, FLT* t,
		   FLT* u, CPX* fk, int *ier)
{
  nufft_opts opts; finufft_default_opts(opts);
  *ier = finufft3d3((INT)*nj,xj,yj,zj,cj,*iflag,*eps,(INT)*nk,s,t,u,fk,opts);
}

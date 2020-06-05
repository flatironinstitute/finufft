#include <finufft.h>
#include <stdio.h>

/* C++ layer for legacy interfaces calling FINUFFT from fortran
   without options control.
   These will become obsolete, but are needed for a transition period.
   Barnett 6/5/20.
*/



#ifdef __cplusplus
extern "C" {
#endif

void legacymsg()
{
  fprintf(stderr,"FINUFFT warning: this obsolete Fortran interface will be deprecated; see docs for current interface.\n");
}  
  
void finufft1d1_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft1d1((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*ms,fk,NULL);
}

void finufft1d2_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft1d2((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*ms,fk,NULL);
}

void finufft1d3_f_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *nk, FLT* s, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft1d3((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*nk,s,fk,NULL);
}

void finufft2d1_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft2d1((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}

void finufft2d2_f_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft2d2((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}

void finufft2d3_f_(int *nj,FLT* xj,FLT* yj, CPX* cj,int *iflag,
		   FLT *eps, int *nk, FLT* s, FLT* t, CPX* fk,
		   int *ier)
{
  legacymsg();
  *ier = finufft2d3((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*nk,s,t,fk,NULL);
}

void finufft3d1_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft3d1((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,(BIGINT)*mu,fk,NULL);
}

void finufft3d2_f_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft3d2((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,(BIGINT)*mu,fk,NULL);
}

void finufft3d3_f_(int *nj,FLT* xj,FLT* yj, FLT*zj, CPX* cj,
		   int *iflag, FLT *eps, int *nk, FLT* s, FLT* t,
		   FLT* u, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft3d3((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*nk,s,t,u,fk,NULL);
}

void finufft2d1many_f_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft2d1many(*ndata,(BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}

void finufft2d2many_f_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = finufft2d2many(*ndata,(BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}

  
#ifdef __cplusplus
}
#endif

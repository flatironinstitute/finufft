#include <finufft.h>
#include <finufft_f.h>
#include <cstdio>
using namespace std;

/* C++ layer for legacy interfaces calling FINUFFT from fortran
   without options control.
   These will become obsolete, but are needed for a transition period.
   Barnett 6/5/20.
*/

static void legacymsg()
{
  fprintf(stderr,"FINUFFT warning: this obsolete Fortran interface will be deprecated; see docs for current interface.\n");
}


#ifdef __cplusplus
extern "C" {
#endif

void FINUFFT1D1_F_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT1D1((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*ms,fk,NULL);
}

void FINUFFT1D2_F_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *ms, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT1D2((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*ms,fk,NULL);
}

void FINUFFT1D3_F_(int *nj,FLT* xj,CPX* cj,int *iflag, FLT *eps,
		  int *nk, FLT* s, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT1D3((BIGINT)*nj,xj,cj,*iflag,*eps,(BIGINT)*nk,s,fk,NULL);
}

void FINUFFT2D1_F_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT2D1((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}

void FINUFFT2D2_F_(int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT2D2((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}

void FINUFFT2D3_F_(int *nj,FLT* xj,FLT* yj, CPX* cj,int *iflag,
		   FLT *eps, int *nk, FLT* s, FLT* t, CPX* fk,
		   int *ier)
{
  legacymsg();
  *ier = FINUFFT2D3((BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*nk,s,t,fk,NULL);
}

void FINUFFT3D1_F_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT3D1((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,(BIGINT)*mu,fk,NULL);
}

void FINUFFT3D2_F_(int *nj,FLT* xj,FLT *yj,FLT* zj,CPX* cj,
		   int *iflag, FLT *eps, int *ms, int* mt, int* mu,
		   CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT3D2((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,(BIGINT)*mu,fk,NULL);
}

void FINUFFT3D3_F_(int *nj,FLT* xj,FLT* yj, FLT*zj, CPX* cj,
		   int *iflag, FLT *eps, int *nk, FLT* s, FLT* t,
		   FLT* u, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT3D3((BIGINT)*nj,xj,yj,zj,cj,*iflag,*eps,(BIGINT)*nk,s,t,u,fk,NULL);
}

void FINUFFT2D1MANY_F_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT2D1MANY(*ndata,(BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}

void FINUFFT2D2MANY_F_(int *ndata, int *nj,FLT* xj,FLT *yj,CPX* cj,int *iflag,
		   FLT *eps, int *ms, int *mt, CPX* fk, int *ier)
{
  legacymsg();
  *ier = FINUFFT2D2MANY(*ndata,(BIGINT)*nj,xj,yj,cj,*iflag,*eps,(BIGINT)*ms,(BIGINT)*mt,fk,NULL);
}


#ifdef __cplusplus
}
#endif

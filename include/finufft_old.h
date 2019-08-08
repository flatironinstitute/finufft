#ifndef FINUFFT_OLD_H
#define FINUFFT_OLD_H

#include <dataTypes_legacy.h>
#include <nufft_opts.h>
// ------------------ library provides ------------------------------------


int finufft1d1_old(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
		   CPX* fk, nufft_opts opts);
int finufft1d2_old(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
		   CPX* fk, nufft_opts opts);
int finufft1d3_old(BIGINT nj,FLT* xj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, CPX* fk, nufft_opts opts);

int finufft2d1_old(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d1many_old(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d2_old(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
		   BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);

int finufft2d2many_old(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
		       FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d3_old(BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts opts);

int finufft3d1_old(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);

int finufft3d2_old(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);
int finufft3d3_old(BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
	       FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
	       CPX* fk, nufft_opts opts);




#endif

#ifndef FINUFFT_OLD_H
#define FINUFFT_OLD_H

#include <dataTypes.h>
#include <nufft_opts.h>
// ------------------ library provides ------------------------------------


int finufft1d1_old(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
		   CPX* fk, nufft_opts opts);
int finufft1d2_old(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
		   CPX* fk, nufft_opts opts);

int finufft2d1_old(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d1many_old(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
                   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
int finufft2d2_old(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
		   BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);

int finufft2d2many_old(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
		       FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);

int finufft1d3_old(BIGINT nj,FLT* xj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, CPX* fk, nufft_opts opts);

#endif

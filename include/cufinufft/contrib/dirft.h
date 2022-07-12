#ifndef DIRFT_H
#define DIRFT_H

#include "utils.h"
#include "utils_fp.h"

void dirft1d1(BIGINT nj,CUFINUFFT_FLT* x,CPX* c,int isign,BIGINT ms, CPX* f);
void dirft1d2(BIGINT nj,CUFINUFFT_FLT* x,CPX* c,int iflag,BIGINT ms, CPX* f);
void dirft1d3(BIGINT nj,CUFINUFFT_FLT* x,CPX* c,int iflag,BIGINT nk, CUFINUFFT_FLT* s, CPX* f);

void dirft2d1(BIGINT nj,CUFINUFFT_FLT* x,CUFINUFFT_FLT *y,CPX* c,int iflag,BIGINT ms, BIGINT mt, CPX* f);
void dirft2d2(BIGINT nj,CUFINUFFT_FLT* x,CUFINUFFT_FLT *y,CPX* c,int iflag,BIGINT ms, BIGINT mt, CPX* f);
void dirft2d3(BIGINT nj,CUFINUFFT_FLT* x,CUFINUFFT_FLT *y,CPX* c,int iflag,BIGINT nk, CUFINUFFT_FLT* s, CUFINUFFT_FLT* t, CPX* f);

void dirft3d1(BIGINT nj,CUFINUFFT_FLT* x,CUFINUFFT_FLT *y,CUFINUFFT_FLT *z,CPX* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, CPX* f);
void dirft3d2(BIGINT nj,CUFINUFFT_FLT* x,CUFINUFFT_FLT *y,CUFINUFFT_FLT *z,CPX* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, CPX* f);
void dirft3d3(BIGINT nj,CUFINUFFT_FLT* x,CUFINUFFT_FLT *y,CUFINUFFT_FLT *z,CPX* c,int iflag,BIGINT nk, CUFINUFFT_FLT* s, CUFINUFFT_FLT* t, CUFINUFFT_FLT *u, CPX* f);

#endif

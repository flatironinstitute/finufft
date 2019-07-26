#ifndef DIRFT_H
#define DIRFT_H

#include "utils.h"

#ifdef T

void TEMPLATE(dirft1d1,T)(BIGINT nj,T* x,CPX* c,int isign,BIGINT ms, CPX* f);
void TEMPLATE(dirft1d2,T)(BIGINT nj,T* x,CPX* c,int iflag,BIGINT ms, CPX* f);
void TEMPLATE(dirft1d3,T)(BIGINT nj,T* x,CPX* c,int iflag,BIGINT nk, T* s, CPX* f);

void TEMPLATE(dirft2d1,T)(BIGINT nj,T* x,T *y,CPX* c,int iflag,BIGINT ms, BIGINT mt, CPX* f);
void TEMPLATE(dirft2d2,T)(BIGINT nj,T* x,T *y,CPX* c,int iflag,BIGINT ms, BIGINT mt, CPX* f);
void TEMPLATE(dirft2d3,T)(BIGINT nj,T* x,T *y,CPX* c,int iflag,BIGINT nk, T* s, T* t, CPX* f);

void TEMPLATE(dirft3d1,T)(BIGINT nj,T* x,T *y,T *z,CPX* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, CPX* f);
void TEMPLATE(dirft3d2,T)(BIGINT nj,T* x,T *y,T *z,CPX* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, CPX* f);
void TEMPLATE(dirft3d3,T)(BIGINT nj,T* x,T *y,T *z,CPX* c,int iflag,BIGINT nk, T* s, T* t, T *u, CPX* f);

#endif

#endif

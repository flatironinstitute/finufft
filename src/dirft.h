#ifndef DIRFT_H
#define DIRFT_H

#include "utils.h"

void dirft1d1(INT nj,FLT* x,CPX* c,int isign,INT ms, CPX* f);
void dirft1d2(INT nj,FLT* x,CPX* c,int iflag,INT ms, CPX* f);
void dirft1d3(INT nj,FLT* x,CPX* c,int iflag,INT nk, FLT* s, CPX* f);

void dirft2d1(INT nj,FLT* x,FLT *y,CPX* c,int iflag,INT ms, INT mt, CPX* f);
void dirft2d2(INT nj,FLT* x,FLT *y,CPX* c,int iflag,INT ms, INT mt, CPX* f);
void dirft2d3(INT nj,FLT* x,FLT *y,CPX* c,int iflag,INT nk, FLT* s, FLT* t, CPX* f);

void dirft3d1(INT nj,FLT* x,FLT *y,FLT *z,CPX* c,int iflag,INT ms, INT mt, INT mu, CPX* f);
void dirft3d2(INT nj,FLT* x,FLT *y,FLT *z,CPX* c,int iflag,INT ms, INT mt, INT mu, CPX* f);
void dirft3d3(INT nj,FLT* x,FLT *y,FLT *z,CPX* c,int iflag,INT nk, FLT* s, FLT* t, FLT *u, CPX* f);

#endif

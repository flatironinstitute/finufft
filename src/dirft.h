#ifndef DIRFT_H
#define DIRFT_H

#include "utils.h"

void dirft1d1(BIGINT nj,double* x,dcomplex* c,int isign,BIGINT ms, dcomplex* f);
void dirft1d2(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT ms, dcomplex* f);
void dirft1d3(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT nk, double* s, dcomplex* f);

void dirft2d1(BIGINT nj,double* x,double *y,dcomplex* c,int iflag,BIGINT ms, BIGINT mt, dcomplex* f);
void dirft2d2(BIGINT nj,double* x,double *y,dcomplex* c,int iflag,BIGINT ms, BIGINT mt, dcomplex* f);
void dirft2d3(BIGINT nj,double* x,double *y,dcomplex* c,int iflag,BIGINT nk, double* s, double* t, dcomplex* f);

void dirft3d1(BIGINT nj,double* x,double *y,double *z,dcomplex* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, dcomplex* f);
void dirft3d2(BIGINT nj,double* x,double *y,double *z,dcomplex* c,int iflag,BIGINT ms, BIGINT mt, BIGINT mu, dcomplex* f);
void dirft3d3(BIGINT nj,double* x,double *y,double *z,dcomplex* c,int iflag,BIGINT nk, double* s, double* t, double *u, dcomplex* f);
#endif

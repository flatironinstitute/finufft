#ifndef DIRFT_H
#define DIRFT_H

#include "utils.h"

void dirft1d1(INT nj,double* x,dcomplex* c,int isign,INT ms, dcomplex* f);
void dirft1d2(INT nj,double* x,dcomplex* c,int iflag,INT ms, dcomplex* f);
void dirft1d3(INT nj,double* x,dcomplex* c,int iflag,INT nk, double* s, dcomplex* f);

void dirft2d1(INT nj,double* x,double *y,dcomplex* c,int iflag,INT ms, INT mt, dcomplex* f);
void dirft2d2(INT nj,double* x,double *y,dcomplex* c,int iflag,INT ms, INT mt, dcomplex* f);
void dirft2d3(INT nj,double* x,double *y,dcomplex* c,int iflag,INT nk, double* s, double* t, dcomplex* f);

void dirft3d1(INT nj,double* x,double *y,double *z,dcomplex* c,int iflag,INT ms, INT mt, INT mu, dcomplex* f);
void dirft3d2(INT nj,double* x,double *y,double *z,dcomplex* c,int iflag,INT ms, INT mt, INT mu, dcomplex* f);
void dirft3d3(INT nj,double* x,double *y,double *z,dcomplex* c,int iflag,INT nk, double* s, double* t, double *u, dcomplex* f);

#endif

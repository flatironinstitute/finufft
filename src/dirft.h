#ifndef DIRFT_H
#define DIRFT_H

#include "utils.h"

void dirft1d1(BIGINT nj,double* x,dcomplex* c,int isign,BIGINT ms, dcomplex* f);
void dirft1d2(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT ms, dcomplex* f);
void dirft1d3(BIGINT nj,double* x,dcomplex* c,int iflag,BIGINT nk, double* s, dcomplex* f);

#endif

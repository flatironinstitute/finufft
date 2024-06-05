#ifndef DIRFT_H
#define DIRFT_H

#include <finufft/defs.h>

void dirft1d1(BIGINT nj, FLT *x, CPX *c, int isign, BIGINT ms, CPX *f);
void dirft1d2(BIGINT nj, FLT *x, CPX *c, int iflag, BIGINT ms, CPX *f);
void dirft1d3(BIGINT nj, FLT *x, CPX *c, int iflag, BIGINT nk, FLT *s, CPX *f);

void dirft2d1(BIGINT nj, FLT *x, FLT *y, CPX *c, int iflag, BIGINT ms, BIGINT mt, CPX *f);
void dirft2d2(BIGINT nj, FLT *x, FLT *y, CPX *c, int iflag, BIGINT ms, BIGINT mt, CPX *f);
void dirft2d3(BIGINT nj, FLT *x, FLT *y, CPX *c, int iflag, BIGINT nk, FLT *s, FLT *t,
              CPX *f);

void dirft3d1(BIGINT nj, FLT *x, FLT *y, FLT *z, CPX *c, int iflag, BIGINT ms, BIGINT mt,
              BIGINT mu, CPX *f);
void dirft3d2(BIGINT nj, FLT *x, FLT *y, FLT *z, CPX *c, int iflag, BIGINT ms, BIGINT mt,
              BIGINT mu, CPX *f);
void dirft3d3(BIGINT nj, FLT *x, FLT *y, FLT *z, CPX *c, int iflag, BIGINT nk, FLT *s,
              FLT *t, FLT *u, CPX *f);

#endif

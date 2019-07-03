#ifndef INVOKE_GURU_H
#define INVOKE_GURU_H

#include <finufft.h>



int invokeGuruInterface(int n_dims, finufft_type type, int n_transf, BIGINT nj, FLT* xj,FLT *yj, FLT *zj, CPX* cj,int iflag,
			FLT eps, BIGINT *n_modes, FLT *s, FLT *t,  FLT *u, CPX* fk, nufft_opts opts);

#endif 

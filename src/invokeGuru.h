#ifndef INVOKE_GURU_H
#define INVOKE_GURU_H

#include <finufft.h>



int invokeGuruInterface(int n_dims, finufft_type type, int n_vecs, BIGINT nj, FLT* xj,FLT *yj,CPX* cj,int iflag,
			FLT eps, BIGINT *n_modes, CPX* fk, nufft_opts opts);

#endif 

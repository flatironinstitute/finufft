#ifndef INVOKE_GURU_H
#define INVOKE_GURU_H


#include <dataTypes_legacy.h>
#include <finufft_type.h>
#include <nufft_opts.h>

int invokeGuruInterface(int n_dims, finufft_type type, int n_transf, BIGINT nj, void * xj, void *yj, void *zj,
			void* cj,int iflag, FLT eps, BIGINT *n_modes, BIGINT nk, void *s, void *t,  void *u,
			void* fk, nufft_opts opts);

#endif 

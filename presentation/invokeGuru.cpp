#include <invokeGuru.h>
#include <defs.h>


int invokeGuruInterface(int n_dims, finufft_type type, int n_transf, BIGINT nj, FLT* xj,FLT *yj, FLT *zj, CPX* cj,int iflag,
			FLT eps, BIGINT *n_modes, BIGINT nk, FLT *s, FLT *t,  FLT *u,  CPX* fk, nufft_opts opts){


  finufft_plan plan;
  
  plan.opts = opts;   /*Copy out user defined options in nufft_opts into the plan*/

  int blksize = MY_OMP_GET_MAX_THREADS(); //default - can only specify through guru interface 
  
  int ier = finufft_makeplan(type, n_dims, n_modes, iflag, n_transf, eps, blksize, &plan);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  ier = finufft_setpts(&plan, nj, xj, yj, zj, nk, s, t, u);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  ier = finufft_exec(&plan, cj, fk);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  finufft_destroy(&plan);
  
  return 0;
}

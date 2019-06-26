#include <invokeGuru.h>


int invokeGuruInterface(int n_dims, finufft_type type, int n_vecs, BIGINT nj, FLT* xj,FLT *yj,CPX* cj,int iflag,
			FLT eps, BIGINT *n_modes, CPX* fk, nufft_opts opts){


  finufft_plan plan;
  
  plan.opts = opts;   /*Copy out user defined options in nufft_opts into the plan*/
  
  int ier = make_finufft_plan(type, n_dims, n_modes, iflag, n_vecs, eps, &plan);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  ier = setNUpoints(&plan, nj, xj, yj, NULL, NULL);
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

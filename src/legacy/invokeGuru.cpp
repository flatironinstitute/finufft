#include <finufft_tempinstant.h>
#include <invokeGuru.h>
#include <defs.h>
#include <templates.h>

int invokeGuruInterface(int n_dims, finufft_type type, int n_transf, BIGINT nj, void * xj, void *yj, void *zj, void * cj,int iflag,
			FLT eps, BIGINT *n_modes, BIGINT nk, void *s, void *t,  void *u,  void* fk, nufft_opts opts){

  /*
  if(sizeof(FLT) == sizeof(float))
    typedef float T;
  else
    typedef double T;
  */ //this does not work, even though it is so much neater 

  if(sizeof(FLT) == sizeof(float)){

  TEMPLATE(finufft_plan,float) plan;
  
  plan.opts = opts;   /*Copy out user defined options in nufft_opts into the plan*/

  int blksize = MY_OMP_GET_MAX_THREADS(); //default - can only specify through guru interface 

  
  int ier = TEMPLATE(make_finufft_plan,float)(type, n_dims, n_modes, iflag, n_transf, eps, blksize, &plan);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  ier = TEMPLATE(setNUpoints,float)(&plan, nj, (float *)xj, (float * )yj, (float *)zj, nk, (float *)s, (float *)t, (float *)u);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  ier = TEMPLATE(finufft_exec,float)(&plan, (CPX_float *)cj, (CPX_float *)fk);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  TEMPLATE(finufft_destroy,float)(&plan);

  }
  else{

  TEMPLATE(finufft_plan,double) plan;
  
  plan.opts = opts;   /*Copy out user defined options in nufft_opts into the plan*/

  int blksize = MY_OMP_GET_MAX_THREADS(); //default - can only specify through guru interface 

  
  int ier = TEMPLATE(make_finufft_plan,double)(type, n_dims, n_modes, iflag, n_transf, eps, blksize, &plan);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  ier = TEMPLATE(setNUpoints,double)(&plan, nj, (double *)xj, (double *)yj, (double *)zj, nk, (double *)s, (double *)t, (double *)u);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  ier = TEMPLATE(finufft_exec,double)(&plan, (CPX_double *)cj, (CPX_double *)fk);
  if(ier){
    if(plan.opts.debug)
      printf("error (ier=%d)!\n", ier);
    return ier;
  }

  TEMPLATE(finufft_destroy,double)(&plan);

  }

  return 0;
}

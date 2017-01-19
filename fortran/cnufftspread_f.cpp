// Wrappers for spreading to be called from fortran.
// (like C wrappers except everything is passed by reference.)

// See cnufftspread_f.h for documentation.

// Magland Dec 2016. Barnett added ier, changed to _f, twopi*, 1/18/17

#include "cnufftspread_f.h"
#include "../src/cnufftspread.h"
#include <stdio.h>

void cnufftspread_f_(int *N1,int *N2,int *N3,double *d_uniform,int *M,double *kx,double *ky,double *kz,double *d_nonuniform,int *spread_direction,double *kernel_params, int *ier)
{
  if (kernel_params[0]!=1) {
    printf("Error: Unexpected kernel type.\n");
    return;
  }
  spread_opts opts;
  set_KB_opts_from_kernel_params(opts,kernel_params);
  opts.spread_direction = *spread_direction;
  
  *ier = cnufftspread(*N1,*N2,*N3,d_uniform,*M,kx,ky,kz,d_nonuniform,opts);
}

void twopispread1d_f_(int *N1,double *d_uniform,int *M,double *kx,double *d_nonuniform,int *spread_direction,double *kernel_params, int *ier)
{
  if (kernel_params[0]!=1) {
    printf("Error: Unexpected kernel type.\n");
    return;
  }
  spread_opts opts;
  *ier = set_KB_opts_from_kernel_params(opts,kernel_params);
  opts.spread_direction = *spread_direction;
  
  // todo: combine the ier outputs
  double *dummy;
  *ier = cnufftspread(*N1,1,1,d_uniform,*M,kx,dummy,dummy,d_nonuniform,opts);
}

void twopispread2d_f_(int *N1,int *N2,double *d_uniform,int *M,double *kx,double *ky,double *d_nonuniform,int *spread_direction,double *kernel_params, int *ier)
{
  if (kernel_params[0]!=1) {
    printf("Error: Unexpected kernel type.\n");
    return;
  }
  spread_opts opts;
  *ier = set_KB_opts_from_kernel_params(opts,kernel_params);
  opts.spread_direction = *spread_direction;
  
  // todo: combine the ier outputs
  double *dummy;
  *ier = cnufftspread(*N1,*N2,1,d_uniform,*M,kx,ky,dummy,d_nonuniform,opts);
}

void twopispread3d_f_(int *N1,int *N2,int *N3,double *d_uniform,int *M,double *kx,double *ky,double *kz,double *d_nonuniform,int *spread_direction,double *kernel_params, int *ier)
{
  if (kernel_params[0]!=1) {
    printf("Error: Unexpected kernel type.\n");
    return;
  }
  spread_opts opts;
  *ier = set_KB_opts_from_kernel_params(opts,kernel_params);
  opts.spread_direction = *spread_direction;
  
  // todo: combine the ier outputs
  *ier = cnufftspread(*N1,*N2,*N3,d_uniform,*M,kx,ky,kz,d_nonuniform,opts);
}

void get_kernel_params_for_eps_f_(double *kernel_params,double *eps,int *ier)
{
  spread_opts opts;
  *ier = set_KB_opts_from_eps(opts,*eps);
  kernel_params[0]=1;
  kernel_params[1]=opts.nspread;
  kernel_params[2]=opts.KB_fac1;
  kernel_params[3]=opts.KB_fac2;
}

void evaluate_kernel_f_(int *len,double *x,double *values,double *kernel_params)
// has no error status reporting
{
  spread_opts opts;
  int ier = set_KB_opts_from_kernel_params(opts,kernel_params);
  evaluate_kernel(*len,x,values,opts);
}

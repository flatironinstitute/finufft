/******************************************************
** See the accompanying README and LICENSE files
** Author(s): Jeremy Magland
** Created: 4/1/2016
*******************************************************/

#include "cnufftspread_c.h"
#include "cnufftspread.h"
#include <stdio.h>

void cnufftspread_f_(int *N1,int *N2,int *N3,double *d_uniform,int *M,double *kx,double *ky,double *kz,double *d_nonuniform,int *spread_direction,double *kernel_params) {

    if (kernel_params[0]!=1) {
        printf("Error: Unexpected kernel type.\n");
        return;
    }

    cnufftspread_opts opts;
    set_kb_opts_from_kernel_params(opts,kernel_params);

    opts.spread_direction=*spread_direction;

    int ier = cnufftspread(*N1,*N2,*N3,d_uniform,*M,kx,ky,kz,d_nonuniform,opts);
}

void get_kernel_params_for_eps_f_(double *kernel_params,double *eps) {
    cnufftspread_opts opts;
    set_kb_opts_from_eps(opts,*eps);
    kernel_params[0]=1;
    kernel_params[1]=opts.nspread;
    kernel_params[2]=opts.KB_fac1;
    kernel_params[3]=opts.KB_fac2;
}

void evaluate_kernel_f_(int *len,double *x,double *values,double *kernel_params) {
    cnufftspread_opts opts;
    set_kb_opts_from_kernel_params(opts,kernel_params);
    evaluate_kernel(*len,x,values,opts);
}

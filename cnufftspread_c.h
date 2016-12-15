/******************************************************
** See the accompanying README and LICENSE files
** Author(s): Jeremy Magland
** Created: 4/1/2016
*******************************************************/


#ifndef CNUFFTSPREAD_C_H
#define CNUFFTSPREAD_C_H

extern "C" {
    // spread direction: 1 means nonuniform->uniform, 2 means uniform->nonuniform
    // see below for info on kernel_params
    void cnufftspread_f_(
            int *N1,int *N2,int *N3,double *d_uniform,
            int *M,double *kx,double *ky,double *kz,double *d_nonuniform,
            int *spread_direction,double *kernel_params);

    // Use the following to get the kernel parameters associated with eps (eg 1e-6) to pass into the spreader.
    // See cnufftspread.h for information on the kernel_params array
    void get_kernel_params_for_eps_f_(double *kernel_params,double *eps);
}

#endif // BLOCKNUFFT3D_C_H


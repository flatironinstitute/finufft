/******************************************************
** See the accompanying README and LICENSE files
** Author(s): Jeremy Magland
** Created: 4/1/2016
*******************************************************/


#ifndef CNUFFTSPREAD_C_H
#define CNUFFTSPREAD_C_H

extern "C" {
    // See cnufftspread.h for information on the kernel_params array and spread_direction

    void cnufftspread_f_(int *N1,int *N2,int *N3,double *d_uniform,int *M,double *kx,double *ky,double *kz,double *d_nonuniform,int *spread_direction,double *kernel_params);
}

#endif // BLOCKNUFFT3D_C_H


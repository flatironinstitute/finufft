/******************************************************
** See the accompanying README and LICENSE files
** Author(s): Jeremy Magland
** Created: 4/1/2016
*******************************************************/


#ifndef CNUFFTSPREAD_C_H
#define CNUFFTSPREAD_C_H

extern "C" {
    void cnufftspread_f_(int *N1,int *N2,int *N3,double *d_uniform,int *M,double *kx,double *ky,double *kz,double *d_nonuniform,double *eps);
}

#endif // BLOCKNUFFT3D_C_H


/******************************************************
** See the accompanying README and LICENSE files
** Author(s): Jeremy Magland
** Created: 4/1/2016
*******************************************************/


#ifndef CNUFFTSPREAD_C_H
#define CNUFFTSPREAD_C_H

/*

cnufftspread_f_:: spread nonuniform points onto uniform grid, or vice versa
    N1,N2,N3 -- dimensions of the uniform grid
    d_uniform -- pointer to data on uniform grid (complex N1*N2*N3)
    M -- number of nonuniform points
    kx,ky,kz -- arrays of size M -- the coordinates of the nonuniform points.
        0 <= kx < N1
        0 <= ky < N2
        0 <= kz < N3
    spread direction -- 1 means nonuniform->uniform, 2 means uniform->nonuniform
    kernel_params -- a vector of length 4 containing the kernel params.
        This vector can be obtained using a call to get_kernel_params_for_eps_f_ (see below)
    Use N1>1, N2=1, N3=1 for 1d transform
    Use N1>1, N2>1, N3=1 for 2d transform
    Use N1>1, N2>1, N3>1 for 3d transform

get_kernel_params_for_eps_f_::
    get the kernel parameters associated with eps (eg 1e-6) to pass into the spreader.
    See cnufftspread.h for information on the kernel_params array
    kernel_params -- vector of length 4 containing the parameters for the kernel
    eps -- the desired precision, eg 1e-6

evaluate_kernel_f_:: evaluate the kernel on a 1d grid
    len -- the length of the 1d grid
    x -- a vector of positions (probably integers) of length len
    values -- the output of length len
    kernel_params -- same as above


*/

extern "C" {
  // spread direction: 1 means nonuniform->uniform, 2 means uniform->nonuniform
  // see below for info on kernel_params
  void cnufftspread_f_(int *N1,int *N2,int *N3,double *d_uniform,
		       int *M,double *kx,double *ky,double *kz,
		       double *d_nonuniform,
		       int *spread_direction,double *kernel_params);

    // Use the following to
  
  void get_kernel_params_for_eps_f_(double *kernel_params,double *eps);
  
  void evaluate_kernel_f_(int *len,double *x,double *values,double *kernel_params);
}

#endif // BLOCKNUFFT3D_C_H


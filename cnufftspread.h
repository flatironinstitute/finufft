#ifndef CNUFFTSPREAD_H
#define CNUFFTSPREAD_H

struct cnufftspread_opts {
    int nspread=6;
    double KB_W=0;
    double KB_beta=0;
    int spread_type=1;
};

bool cnufftspread(
            long N1, long N2, long N3, double *data_uniform,
            long M, double *kx, double *ky, double *kz, double *data_nonuniform,
            const cnufftspread_opts &opts
        );

void set_kb_opts(cnufftspread_opts &opts,int kernel_radius,double fac1=1,double fac2=1);
void set_kb_opts_from_eps(cnufftspread_opts &opts,double eps);


// MATLAB interface

/*
 * MCWRAP [ COMPLEX Y[N,N,N] ] = cnufftspread_type1(N,kx[M,1],ky[M,1],kz[M,1],COMPLEX X[M,1],eps)
 * SET_INPUT M = size(kx,1)
 * SOURCES qute.cpp cnufftspread.cpp besseli.cpp
 * HEADERS qute.h cnufftspread.h besseli.h
 */

void cnufftspread_type1(int N,double *Y,int M,double *kx,double *ky,double *kz,double *X,double eps);

#endif // CNUFFTSPREAD_H

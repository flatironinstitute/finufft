#ifndef CNUFFTSPREAD_H
#define CNUFFTSPREAD_H

#include <stdlib.h>
#include <stdio.h>

struct cnufftspread_opts {
    int nspread=6;
    double KB_fac1=1;
    double KB_fac2=1;
    int spread_direction=1; // --> 1 means Type I transform (nonuni->uni), 2 means Type II transform (uni->nonuni)
    bool sort_data=true;

    double private_KB_W=0;
    double private_KB_beta=0;
};

bool cnufftspread(
            long N1, long N2, long N3, double *data_uniform,
            long M, double *kx, double *ky, double *kz, double *data_nonuniform,
            cnufftspread_opts opts
        );

void evaluate_kernel(int len, double *x, double *values, cnufftspread_opts opts);

/*
 * kernel_params is an array with the following information
 *  0: kernel type (1 for kaiser-bessel)
 *  1: nspread
 *  2: KB_fac1 (eg 1)
 *  2: KB_fac2 (eg 1)
 */

void set_kb_opts_from_kernel_params(cnufftspread_opts &opts,double *kernel_params);
void set_kb_opts_from_eps(cnufftspread_opts &opts,double eps);


// MATLAB interface

/*
 * MCWRAP [ COMPLEX Y[N,N,N] ] = cnufftspread_type1(N,kx[M,1],ky[M,1],kz[M,1],COMPLEX X[M,1],kernel_params[4,1])
 * SET_INPUT M = size(kx,1)
 * SOURCES cnufftspread.cpp besseli.cpp
 * HEADERS cnufftspread.h besseli.h
 */

void cnufftspread_type1(int N,double *Y,int M,double *kx,double *ky,double *kz,double *X,double *kernel_params);

class CNTime {
public:
    void start();
    int restart();
    int elapsed();
private:
    struct timeval initial;
};

#endif // CNUFFTSPREAD_H

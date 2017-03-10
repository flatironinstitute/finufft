#ifndef FINUFFT_MEX_H
#define FINUFFT_MEX_H

#include "../src/finufft.h"

/* MATLAB interface using MCWRAP

MCWRAP [ COMPLEX uniform_data[N,1] ] = finufft1d1_mex( N, nonuniform_locations[M,1], COMPLEX nonuniform_data[M,1], isign, tol, num_threads)
    SET_INPUT M = size(nonuniform_locations,1)
    HEADERS finufft_mex.h
    SOURCES finufft_mex.cpp
    MEXARGS ../src/finufft.a -lm -lgomp -lfftw3 -lfftw3_threads -lrt

*/
void finufft1d1_mex(int M, int N, double *uniform_data, double *nonuniform_locations, double *nonuniform_data, int isign, double tol, int num_threads);

/*
MCWRAP [ COMPLEX uniform_data[N1,N2] ] = finufft2d1_mex( N1, N2, nonuniform_locations[M,2], COMPLEX nonuniform_data[M,1], isign, tol, num_threads)
    SET_INPUT M = size(nonuniform_locations,1)
    HEADERS finufft_mex.h
    SOURCES finufft_mex.cpp
    MEXARGS ../src/finufft.a -lm -lgomp -lfftw3 -lfftw3_threads -lrt

*/
void finufft2d1_mex(int M, int N1, int N2, double *uniform_data, double *nonuniform_locations, double *nonuniform_data, int isign, double tol, int num_threads);

/*
MCWRAP [ COMPLEX uniform_data[N1,N2,N3] ] = finufft3d1_mex( N1, N2, N3, nonuniform_locations[M,3], COMPLEX nonuniform_data[M,1], isign, tol, num_threads)
    SET_INPUT M = size(nonuniform_locations,1)
    HEADERS finufft_mex.h
    SOURCES finufft_mex.cpp
    MEXARGS ../src/finufft.a -lm -lgomp -lfftw3 -lfftw3_threads -lrt

*/
void finufft3d1_mex(int M, int N1, int N2, int N3, double *uniform_data, double *nonuniform_locations, double *nonuniform_data, int isign, double tol, int num_threads);



#endif // FINUFFT_MEX_H

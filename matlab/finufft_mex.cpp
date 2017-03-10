#include "finufft_mex.h"
#include "omp.h"

void finufft1d1_mex(int M, int N, double *uniform_data, double *nonuniform_locations, double *nonuniform_data, int isign, double tol, int num_threads) {
    nufft_opts opts;
    opts.debug = 1;            // to see some timings

    if (!num_threads) num_threads=omp_get_max_threads();
    omp_set_num_threads(num_threads);
    finufft1d1(M,nonuniform_locations,(dcomplex*)nonuniform_data,isign,tol,N,(dcomplex*)uniform_data,opts);
}

void finufft2d1_mex(int M, int N1, int N2, double *uniform_data, double *nonuniform_locations, double *nonuniform_data, int isign, double tol, int num_threads) {
    nufft_opts opts;
    opts.debug = 1;            // to see some timings

    if (!num_threads) num_threads=omp_get_max_threads();
    omp_set_num_threads(num_threads);
    finufft2d1(M,&nonuniform_locations[0],&nonuniform_locations[M],(dcomplex*)nonuniform_data,isign,tol,N1,N2,(dcomplex*)uniform_data,opts);
}

void finufft3d1_mex(int M, int N1, int N2, int N3, double *uniform_data, double *nonuniform_locations, double *nonuniform_data, int isign, double tol, int num_threads) {
    nufft_opts opts;
    opts.debug = 1;            // to see some timings

    if (!num_threads) num_threads=omp_get_max_threads();
    omp_set_num_threads(num_threads);
    finufft3d1(M,&nonuniform_locations[0],&nonuniform_locations[M],&nonuniform_locations[M*2],(dcomplex*)nonuniform_data,isign,tol,N1,N2,N3,(dcomplex*)uniform_data,opts);
}

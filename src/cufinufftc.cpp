#include <cufinufft.h>

extern "C" {

int cufinufftc_default_opts(int type, int dim, cufinufft_opts *opts)
{
    return cufinufft_default_opts(type, dim, *opts);
}

int cufinufftc_makeplan(int type, int dim, int *n_modes, int iflag,
    int ntransf, FLT tol, int ntransfcufftplan, cufinufft_plan *d_plan)
{
    return cufinufft_makeplan(type, dim, n_modes, iflag, ntransf, tol, ntransfcufftplan, d_plan);
}

int cufinufftc_setNUpts(int M, FLT* h_kx, FLT* h_ky, FLT* h_kz, int N, FLT
        *h_s, FLT *h_t, FLT *h_u, cufinufft_plan *d_plan)
{
    return cufinufft_setNUpts(M, h_kx, h_ky, h_kz, N, h_s, h_t, h_u, d_plan);
}

int cufinufftc_exec(CUCPX* h_c, CUCPX* h_fk, cufinufft_plan *d_plan)
{
    return cufinufft_exec(h_c, h_fk, d_plan);
}

int cufinufftc_destroy(cufinufft_plan *d_plan)
{
    return cufinufft_destroy(d_plan);
}

}

#include <cufinufft_eitherprec.h>

extern "C" {

int CUFINUFFTC_DEFAULT_OPTS(int type, int dim, CUFINUFFT_OPTS *opts)
{
    return CUFINUFFT_DEFAULT_OPTS(type, dim, opts);
}

int CUFINUFFTC_MAKEPLAN(int type, int dim, int *n_modes, int iflag,
    int ntransf, FLT tol, int maxbatchsize, CUFINUFFT_PLAN *d_plan)
{
    return CUFINUFFT_MAKEPLAN(type, dim, n_modes, iflag, ntransf, tol, maxbatchsize, d_plan);
}

int CUFINUFFTC_SETNUPTS(int M, FLT* h_kx, FLT* h_ky, FLT* h_kz, int N, FLT
        *h_s, FLT *h_t, FLT *h_u, CUFINUFFT_PLAN *d_plan)
{
    return CUFINUFFT_SETNUPTS(M, h_kx, h_ky, h_kz, N, h_s, h_t, h_u, d_plan);
}

int CUFINUFFTC_EXEC(CUCPX* h_c, CUCPX* h_fk, CUFINUFFT_PLAN *d_plan)
{
    return CUFINUFFT_EXEC(h_c, h_fk, d_plan);
}

int CUFINUFFTC_DESTROY(CUFINUFFT_PLAN *d_plan)
{
    return CUFINUFFT_DESTROY(d_plan);
}

}

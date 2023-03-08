#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuComplex.h>
#include <cuda.h>

#include <cufinufft/common.h>
#include <cufinufft/defs.h>
#include <cufinufft/precision_independent.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

#include <legendre_rule_fast.h>

namespace cufinufft {
namespace common {
using namespace cufinufft::spreadinterp;
using std::max;

/* Kernel for computing approximations of exact Fourier series coeffs of
   cnufftspread's real symmetric kernel. */
// a , f are intermediate results from function onedim_fseries_kernel_precomp()
// (see cufinufft/contrib/common.cpp for description)
__global__ void FseriesKernelCompute(int nf1, int nf2, int nf3, CUFINUFFT_FLT *f, cuDoubleComplex *a,
                                     CUFINUFFT_FLT *fwkerhalf1, CUFINUFFT_FLT *fwkerhalf2, CUFINUFFT_FLT *fwkerhalf3,
                                     int ns) {
    CUFINUFFT_FLT J2 = ns / 2.0;
    int q = (int)(2 + 3.0 * J2);
    int nf;
    cuDoubleComplex *at = a + threadIdx.y * MAX_NQUAD;
    CUFINUFFT_FLT *ft = f + threadIdx.y * MAX_NQUAD;
    CUFINUFFT_FLT *oarr;
    if (threadIdx.y == 0) {
        oarr = fwkerhalf1;
        nf = nf1;
    } else if (threadIdx.y == 1) {
        oarr = fwkerhalf2;
        nf = nf2;
    } else {
        oarr = fwkerhalf3;
        nf = nf3;
    }

    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nf / 2 + 1; i += blockDim.x * gridDim.x) {
        int brk = 0.5 + i;
        CUFINUFFT_FLT x = 0.0;
        for (int n = 0; n < q; n++) {
            x += ft[n] * 2 * (pow(cabs(at[n]), brk) * cos(brk * carg(at[n])));
        }
        oarr[i] = x;
    }
}

int CUFSERIESKERNELCOMPUTE(int dim, int nf1, int nf2, int nf3, CUFINUFFT_FLT *d_f, cuDoubleComplex *d_a,
                           CUFINUFFT_FLT *d_fwkerhalf1, CUFINUFFT_FLT *d_fwkerhalf2, CUFINUFFT_FLT *d_fwkerhalf3,
                           int ns)
/*
    wrapper for approximation of Fourier series of real symmetric spreading
    kernel.

    Melody Shih 2/20/22
*/
{
    int nout = max(max(nf1 / 2 + 1, nf2 / 2 + 1), nf3 / 2 + 1);

    dim3 threadsPerBlock(16, dim);
    dim3 numBlocks((nout + 16 - 1) / 16, 1);

    FseriesKernelCompute<<<numBlocks, threadsPerBlock>>>(nf1, nf2, nf3, d_f, d_a, d_fwkerhalf1, d_fwkerhalf2,
                                                         d_fwkerhalf3, ns);
    return 0;
}

int setup_spreader_for_nufft(finufft_spread_opts &spopts, CUFINUFFT_FLT eps, cufinufft_opts opts)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Report status of setup_spreader.  Barnett 10/30/17
{
    int ier = setup_spreader(spopts, eps, opts.upsampfac, opts.gpu_kerevalmeth);
    spopts.pirange = 1; // could allow user control?
    return ier;
}

void SET_NF_TYPE12(CUFINUFFT_BIGINT ms, cufinufft_opts opts, finufft_spread_opts spopts, CUFINUFFT_BIGINT *nf,
                   CUFINUFFT_BIGINT bs)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
    *nf = (CUFINUFFT_BIGINT)(opts.upsampfac * ms);
    if (*nf < 2 * spopts.nspread)
        *nf = 2 * spopts.nspread; // otherwise spread fails
    if (*nf < MAX_NF) {           // otherwise will fail anyway
        if (opts.gpu_method == 4) // expensive at huge nf
            *nf = utils::next235beven(*nf, bs);
        else
            *nf = utils::next235beven(*nf, 1);
    }
}

void onedim_fseries_kernel(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *fwkerhalf, finufft_spread_opts opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 CUFINUFFT_FLTs)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18
  Melody 2/20/22 separate into precomp & comp functions defined below.
 */
{
    CUFINUFFT_FLT f[MAX_NQUAD];
    dcomplex a[MAX_NQUAD];
    onedim_fseries_kernel_precomp(nf, f, a, opts);
    onedim_fseries_kernel_compute(nf, f, a, fwkerhalf, opts);
}

/*
  Precomputation of approximations of exact Fourier series coeffs of cnufftspread's
  real symmetric kernel.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  a - phase winding rates
  f - funciton values at quadrature nodes multiplied with quadrature weights
  (a, f are provided as the inputs of onedim_fseries_kernel_compute() defined below)
*/
void onedim_fseries_kernel_precomp(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, finufft_spread_opts opts) {
    CUFINUFFT_FLT J2 = opts.nspread / 2.0; // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    int q = (int)(2 + 3.0 * J2); // not sure why so large? cannot exceed MAX_NQUAD
    double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];

    finufft::quadrature::legendre_compute_glr(2 * q, z, w); // only half the nodes used, eg on (0,1)
    for (int n = 0; n < q; ++n) {                           // set up nodes z_n and vals f_n
        z[n] *= J2;                                         // rescale nodes
        f[n] = J2 * (CUFINUFFT_FLT)w[n] * evaluate_kernel((CUFINUFFT_FLT)z[n], opts);  // vals & quadr wei
        a[n] = exp(2 * PI * IMA * (CUFINUFFT_FLT)(nf / 2 - z[n]) / (CUFINUFFT_FLT)nf); // phase winding rates
    }
}

void onedim_fseries_kernel_compute(CUFINUFFT_BIGINT nf, CUFINUFFT_FLT *f, dcomplex *a, CUFINUFFT_FLT *fwkerhalf,
                                   finufft_spread_opts opts) {
    CUFINUFFT_FLT J2 = opts.nspread / 2.0;        // J/2, half-width of ker z-support
    int q = (int)(2 + 3.0 * J2);                  // not sure why so large? cannot exceed MAX_NQUAD
    CUFINUFFT_BIGINT nout = nf / 2 + 1;           // how many values we're writing to
    int nt = MIN(nout, MY_OMP_GET_MAX_THREADS()); // how many chunks
    std::vector<CUFINUFFT_BIGINT> brk(nt + 1);    // start indices for each thread
    for (int t = 0; t <= nt; ++t)                 // split nout mode indices btw threads
        brk[t] = (CUFINUFFT_BIGINT)(0.5 + nout * t / (double)nt);
#pragma omp parallel
    {
        int t = MY_OMP_GET_THREAD_NUM();
        if (t < nt) {               // could be nt < actual # threads
            dcomplex aj[MAX_NQUAD]; // phase rotator for this thread
            for (int n = 0; n < q; ++n)
                aj[n] = pow(a[n], (CUFINUFFT_FLT)brk[t]);            // init phase factors for chunk
            for (CUFINUFFT_BIGINT j = brk[t]; j < brk[t + 1]; ++j) { // loop along output array
                CUFINUFFT_FLT x = 0.0;                               // accumulator for answer at this j
                for (int n = 0; n < q; ++n) {
                    x += f[n] * 2 * real(aj[n]); // include the negative freq
                    aj[n] *= a[n];               // wind the phases
                }
                fwkerhalf[j] = x;
            }
        }
    }
}

} // namespace common
} // namespace cufinufft

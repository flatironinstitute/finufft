#include "finufft.h"

#include <iostream>
#include <iomanip>
using namespace std;

int finufft1d1(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts)
 /*  Type-1 1D complex nonuniform FFT.

               1 nj-1
     fk(k1) = -- SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2 
              nj j=0                            
   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj     location of sources on interval [-pi,pi].
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex Fourier transform values (size ms, increasing mode ordering)
            stored as alternating Re & Im parts (2*ms doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        corresponding coefficient from the kernel alone.
     The latter kernel FFT is precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 1/22/17
 */
{
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double params[4];
  get_kernel_params_for_eps(params,eps); // todo: use either params or spopts?
  BIGINT nf1 = set_nf(ms,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("1d1: ms=%d nf1=%d nj=%d ...\n",ms,nf1,nj); 

  // STEP 0: get FT of real symmetric spreading kernel
  CNTime timer; timer.start();
  double *fwkerhalf = (double*)malloc(sizeof(double)*(nf1/2+1));
  double prefac_unused_dims;
  onedim_fseries_kernel(nf1, fwkerhalf, prefac_unused_dims, spopts);
  double t=timer.elapsedsec();
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,t);
  //for (int j=0;j<=nf1/2;++j) cout<<fwkerhalf[j]<<endl;

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1);    // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_1d(nf1,fw,fw,fftsign, FFTW_ESTIMATE);  // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // Step 1: spread from irregular points to regular grid
  timer.restart();
  int ier_spread = twopispread1d(nf1,(double*)fw,nj,xj,cj,1,params,opts.spread_debug);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n", nth, timer.elapsedsec());
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;
  //for (int j=0;j<=nf1/2;++j) cout<<fwkerhalf[j]<<endl;

  // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  timer.restart();
  double prefac = 1.0/(prefac_unused_dims*prefac_unused_dims*nj);  // 1/nj norm
  deconvolveshuffle1d(1,prefac,fwkerhalf,ms,fk,nf1,fw);
  if (opts.debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());
  //for (int j=0;j<ms;++j) cout<<fk[2*j]<<"\t"<<fk[2*j+1]<<endl;

  fftw_free(fw); fftw_free(fwkerhalf); if (opts.debug) printf("freed\n");
  return 0;
}


int finufft1d2(BIGINT nj,double* xj,double* cj,int iflag,double eps,BIGINT ms,
	       double* fk, nufft_opts opts)
 /*  Type-2 1D complex nonuniform FFT.

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1 
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

   Inputs:
     nj     number of target (integer of type BIGINT; see utils.h)
     xj     location of targets on interval [-pi,pi].
     fk     complex Fourier transform values (size ms, increasing mode ordering)
            stored as alternating Re & Im parts (2*ms doubles)
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     ms     number of Fourier modes input, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     complex answers at targets interleaving Re & Im parts (2*nj doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 1/25/17
 */
{
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double params[4];
  get_kernel_params_for_eps(params,eps); // todo: use either params or spopts?
  BIGINT nf1 = set_nf(ms,opts,spopts);
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("1d2: ms=%d nf1=%d nj=%d ...\n",ms,nf1,nj); 

  // STEP 0: get FT of real symmetric spreading kernel
  CNTime timer; timer.start();
  double *fwkerhalf = (double*)malloc(sizeof(double)*(nf1/2+1));
  double prefac_unused_dims;
  onedim_fseries_kernel(nf1, fwkerhalf, prefac_unused_dims, spopts);
  double t=timer.elapsedsec();
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,t);

  int nth = omp_get_max_threads();     // set up multithreaded fftw stuff
#ifdef _OPENMP
  fftw_init_threads();
  fftw_plan_with_nthreads(nth);
#endif
  timer.restart();
  fftw_complex *fw = fftw_alloc_complex(nf1);    // working upsampled array
  int fftsign = (iflag>0) ? 1 : -1;
  fftw_plan p = fftw_plan_dft_1d(nf1,fw,fw,fftsign, FFTW_ESTIMATE); // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // STEP 1: amplify Fourier coeffs fk and copy into upsampled array fw
  timer.restart();
  double prefac = 1.0/(prefac_unused_dims*prefac_unused_dims);
  deconvolveshuffle1d(2,prefac,fwkerhalf,ms,fk,nf1,fw);
  if (opts.debug) printf("amplify & copy in:\t %.3g s\n",timer.elapsedsec());
  //cout<<"fw:\n"; for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  fftw_execute(p);
  fftw_destroy_plan(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n",nth,timer.elapsedsec());

  // Step 3: unspread (interpolate) from regular to irregular target pts
  timer.restart();
  int ier_spread = twopispread1d(nf1,(double*)fw,nj,xj,cj,2,params,opts.spread_debug);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n",ier_spread,timer.elapsedsec());

  fftw_free(fw); fftw_free(fwkerhalf); if (opts.debug) printf("freed\n");
  return ier_spread;
}


int finufft1d3(BIGINT nj,double* xj,double* cj,int iflag, double eps, BIGINT nk, double* s, double* fk, nufft_opts opts)
 /*  Type-3 1D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i s[k] xj[j]),      for k = 0, ..., nk-1
               j=0
   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj     location of sources on interval [-pi,pi].
     cj     complex source strengths, interleaving Re & Im parts (2*nj doubles)
     nk     number of frequency target points
     s      frequency locations of targets on the real line in [-A,A] for now
     iflag  if >0, uses + sign in exponential, otherwise - sign.
     eps    precision requested
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex Fourier transform values at the target frequencies sk
            stored as alternating Re & Im parts (2*nk doubles)
     returned value - error return code, as returned by cnufftspread:
                      0 : success.

     The type 3 algorithm is a type 2 wrapped inside a type 1, see [LG].

   Written with FFTW style complex arrays. Barnett 2/8/17
 */
{
  spread_opts spopts;
  int ier_set = set_KB_opts_from_eps(spopts,eps);
  double X1,C1,S1,D1,params[4];
  get_kernel_params_for_eps(params,eps); // todo: use either params or spopts?
  cout << scientific << setprecision(15);  // for debug

  // decide x and s intervals and shifts and scalings...
  arraywidcen(nj,xj,X1,C1);    // width and center of interval containing {x_j}
  arraywidcen(nk,s,S1,D1);     // width and center of interval containing {s_k}
  // *** group these 3 in one call...
  BIGINT nf1 = set_nf_type3(S1*X1,opts,spopts);
  double h1 = 2*M_PI/nf1;       // upsampled grid spacing
  double gam1 = (X1/M_PI)*(1.0 + spopts.nspread/(double)nf1);   // x scale fac
  double* xp = (double*)malloc(sizeof(double)*nj);
  for (BIGINT j=0;j<nj;++j) xp[j] = (xj[j]-C1) / gam1;       // rescaled x'_j

  if (opts.debug) printf("1d3: S1=%.3g X1=%.3g nf1=%ld nj=%ld nk=%ld...\n",S1,X1,nf1,nj,nk);

  // Step 1: spread from irregular sources to regular grid as in type 1
  dcomplex* fw = (dcomplex*)malloc(sizeof(dcomplex)*nf1);
  CNTime timer; timer.start();
  int ier_spread = twopispread1d(nf1,(double*)fw,nj,xp,cj,1,params,opts.spread_debug);
  free(xp);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;  // problem
  //for (int j=0;j<nf1;++j) printf("fw[%d]=%.3g\n",j,real(fw[j]));

  // Step 2: call type-2 to eval regular as Fourier series at rescaled targs
  timer.restart();
  double *ss = (double*)malloc(sizeof(double)*nk);
  // *** insert best translation of s too which rephases fw
  for (BIGINT k=0;k<nk;++k) ss[k] = h1*gam1*s[k];    // should have |ss| < pi/R
  int ier_t2 = finufft1d2(nk,ss,fk,iflag,eps,nf1,(double*)fw,opts);
  free(fw);
  if (opts.debug) printf("type-2 (ier=%d):\t\t %.3g s\n",ier_t2,timer.elapsedsec());
  //for (int k=0;k<nk;++k) printf("fk[%d]=(%.3g,%.3g)\n",k,fk[2*k],fk[2*k+1]);

  // Step 3: correct by dividing by Fourier transform of kernel at targets
  timer.restart();
  double *fkker = (double*)malloc(sizeof(double)*nk);
  double prefac_unused_dims;
  onedim_nuft_kernel(nk, ss, fkker, prefac_unused_dims, spopts); // fill fkker
  if (opts.debug) printf("kernel FT (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());
  timer.restart();
  double prefac = 1.0/(prefac_unused_dims*prefac_unused_dims); // since 2 unused
            // *** replace by func :
  dcomplex *fkc = (dcomplex*)fk;    // handle output as complex array
  for (BIGINT k=0;k<nk;++k)
    fkc[k] *= (dcomplex)(prefac/fkker[k]) * exp(ima*s[k]*C1);
  if (opts.debug) printf("deconvolve:\t\t %.3g s\n",timer.elapsedsec());

  free(fkker); free(ss);
  return ier_t2;
}

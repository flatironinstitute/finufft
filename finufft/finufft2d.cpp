#include "finufft.h"
#include "common.h"
#include "cnufftspread.h"
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

int finufft2d1_cpu(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
	           FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)
 /*  Type-1 2D complex nonuniform FFT.

                  nj-1
     f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                  j=0

     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

     The output array is k1 (fast), then k2 (slow), with each dimension
     determined by opts.modeord.
     If iflag>0 the + sign is used, otherwise the - sign is used,
     in the exponential.

   Inputs:
     nj     number of sources (int64)
     xj,yj     x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
     cj     size-nj complex FLT array of source strengths,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     ms,mt  number of Fourier modes requested in x and y (int64);
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex FLT array of Fourier transform values
            (size ms*mt, fast in ms then slow in mt,
            ie Fortran ordering).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the
        Fourier series coefficient of the kernel.
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 2/1/17
 */
{
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  BIGINT nf1; set_nf_type12(ms,opts,spopts,&nf1);
  BIGINT nf2; set_nf_type12(mt,opts,spopts,&nf2);
  if (nf1*nf2>MAX_NF) {
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("2d1: (ms,mt)=(%ld,%ld) (nf1,nf2)=(%ld,%ld) nj=%ld ...\n",(int64_t)ms,(int64_t)mt,(int64_t)nf1,(int64_t)nf2,(int64_t)nj);

  // STEP 0: get Fourier coeffs of spread kernel in each dim:
  CNTime timer; timer.start();
  FLT *fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
  FLT *fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
  onedim_fseries_kernel(nf1, fwkerhalf1, spopts);
  onedim_fseries_kernel(nf2, fwkerhalf2, spopts);
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());

  int nth = MY_OMP_GET_MAX_THREADS();
  if (nth>1) {             // set up multithreaded fftw stuff...
    FFTW_INIT();
    FFTW_PLAN_TH(nth);
  }
  timer.restart();
  FFTW_CPX *fw = FFTW_ALLOC_CPX(nf1*nf2);  // working upsampled array
  int fftsign = (iflag>=0) ? 1 : -1;
  FFTW_PLAN p = FFTW_PLAN_2D(nf2,nf1,fw,fw,fftsign, opts.fftw);  // in-place
  if (opts.debug) printf("fftw plan (%d)    \t %.3g s\n",opts.fftw,timer.elapsedsec());

  // Step 1: spread from irregular points to regular grid
  timer.restart();
  spopts.spread_direction = 1;
  FLT *dummy;
  int ier_spread = cnufftspread(nf1,nf2,1,(FLT*)fw,nj,xj,yj,dummy,(FLT*)cj,spopts);

  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  // Step 2:  Call FFT
  timer.restart();
  FFTW_EX(p);
  FFTW_DE(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n", nth, timer.elapsedsec());

  // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  timer.restart();
  deconvolveshuffle2d(1,1.0,fwkerhalf1,fwkerhalf2,ms,mt,(FLT*)fk,nf1,nf2,fw,opts.modeord);
  if (opts.debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());

  FFTW_FR(fw); free(fwkerhalf1); free(fwkerhalf2);
  if (opts.debug) printf("freed\n");
  return 0;
}
int finufft2d2_cpu(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	           BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

 /*  Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

   Inputs:
     nj     number of sources (int64)
     xj,yj     x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
     fk     FLT complex array of Fourier transform values (size ms*mt,
            increasing fast in ms then slow in mt, ie Fortran ordering).
            Along each dimension the ordering is set by opts.modeord.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     ms,mt  numbers of Fourier modes given in x and y (int64)
            each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex FLT array of target values
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     returned value - 0 if success, else see ../docs/usage.rst

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Barnett 2/1/17
 */
{
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  BIGINT nf1; set_nf_type12(ms,opts,spopts,&nf1);
  BIGINT nf2; set_nf_type12(mt,opts,spopts,&nf2);
  if (nf1*nf2>MAX_NF) {
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("2d2: (ms,mt)=(%ld,%ld) (nf1,nf2)=(%ld,%ld) nj=%ld ...\n",(int64_t)ms,(int64_t)mt,(int64_t)nf1,(int64_t)nf2,(int64_t)nj);

  // STEP 0: get Fourier coeffs of spread kernel in each dim:
  CNTime timer; timer.start();
  FLT *fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
  FLT *fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
  onedim_fseries_kernel(nf1, fwkerhalf1, spopts);
  onedim_fseries_kernel(nf2, fwkerhalf2, spopts);
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());

  int nth = MY_OMP_GET_MAX_THREADS();
  if (nth>1) {             // set up multithreaded fftw stuff...
    FFTW_INIT();
    FFTW_PLAN_TH(nth);
  }
  timer.restart();
  FFTW_CPX *fw = FFTW_ALLOC_CPX(nf1*nf2);  // working upsampled array
  int fftsign = (iflag>=0) ? 1 : -1;
  FFTW_PLAN p = FFTW_PLAN_2D(nf2,nf1,fw,fw,fftsign, opts.fftw);  // in-place
  if (opts.debug) printf("fftw plan (%d)    \t %.3g s\n",opts.fftw,timer.elapsedsec());

  // STEP 1: amplify Fourier coeffs fk and copy into upsampled array fw
  timer.restart();
  deconvolveshuffle2d(2,1.0,fwkerhalf1,fwkerhalf2,ms,mt,(FLT*)fk,nf1,nf2,fw,opts.modeord);
  if (opts.debug) printf("amplify & copy in:\t %.3g s\n",timer.elapsedsec());
  //cout<<"fw:\n"; for (int j=0;j<nf1*nf2;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  FFTW_EX(p);
  FFTW_DE(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n",nth,timer.elapsedsec());

  // Step 3: unspread (interpolate) from regular to irregular target pts
  timer.restart();
  FLT *dummy;
  spopts.spread_direction = 2;
  int ier_spread = cnufftspread(nf1,nf2,1,(FLT*)fw,nj,xj,yj,dummy,(FLT*)cj,spopts);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  FFTW_FR(fw); free(fwkerhalf1); free(fwkerhalf2);
  if (opts.debug) printf("freed\n");
  return 0;
}

int finufft2d1_gpu(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
	           FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)
{
  cufinufft_plan dplan;
  cufinufft_opts cuopts;
  CNTime timer;
  timer.start();
  int ier=setup_cuspreader(cuopts,eps,opts.upsampfac);
  cuopts.spread_direction=1;
  ier=cufinufft2d_plan(nj, xj, yj, cj, ms, mt, fk, iflag, cuopts, &dplan);
  if (opts.debug) printf("[time  ] cufinufft2d1 plan:\t\t %.3g s\n", timer.elapsedsec());

  timer.restart();
  ier=cufinufft2d1_exec(cuopts, &dplan);
  if (opts.debug) printf("[time  ] cufinufft2d1 exec:\t\t %.3g s\n", timer.elapsedsec());

  timer.restart();
  ier=cufinufft2d_destroy(cuopts, &dplan);
  if (opts.debug) printf("[time  ] cufinufft2d1 destroy:\t\t %.3g s\n", timer.elapsedsec());
  return 0;
}

int finufft2d2_gpu(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
	           FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)
{
  cufinufft_plan dplan;
  cufinufft_opts cuopts;
  CNTime timer;
  timer.start();
  int ier=setup_cuspreader(cuopts,eps,opts.upsampfac);
  cuopts.spread_direction=2;

  ier=cufinufft2d_plan(nj, xj, yj, cj, ms, mt, fk, iflag, cuopts, &dplan);
  if (opts.debug) printf("[time  ] cufinufft2d1 plan:\t\t %.3g s\n", timer.elapsedsec());

  timer.restart();
  ier=cufinufft2d2_exec(cuopts, &dplan);
  if (opts.debug) printf("[time  ] cufinufft2d2 exec:\t\t %.3g s\n", timer.elapsedsec());

  timer.restart();
  ier=cufinufft2d_destroy(cuopts, &dplan);
  if (opts.debug) printf("[time  ] cufinufft2d1 destroy:\t\t %.3g s\n", timer.elapsedsec());
  return 0;
}

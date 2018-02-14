#include "finufft.h"
#include "common.h"
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

int finufft2d1(INT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
	       FLT eps, INT ms, INT mt, CPX* fk, nufft_opts opts)
 /*  Type-1 2D complex nonuniform FFT.

                  nj-1
     f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                  j=0
 
     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

     The output array is in increasing k1 ordering (fast), then increasing
     k2 ordering (slow). If iflag>0 the + sign is
     used, otherwise the - sign is used, in the exponential.
                           
   Inputs:
     nj     number of sources
     xj,yj     x,y locations of sources on 2D domain [-pi,pi]^2.
     cj     size-nj complex FLT array of source strengths, 
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms,mt  number of Fourier modes requested in x and y; each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex FLT array of Fourier transform values
            (size ms*mt, increasing fast in ms then slow in mt,
            ie Fortran ordering).
     returned value - 0 if success, else:
                      1 : eps too small
		      2 : size of arrays to malloc exceed MAX_NF
                      other codes: as returned by cnufftspread

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
  INT64 nf1; set_nf_type12((BIGINT)ms,opts,spopts,&nf1);
  INT64 nf2; set_nf_type12((BIGINT)mt,opts,spopts,&nf2);
  if (nf1*nf2>MAX_NF) {
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("2d1: (ms,mt)=(%ld,%ld) (nf1,nf2)=(%ld,%ld) nj=%ld ...\n",(INT64)ms,(INT64)mt,nf1,nf2,(INT64)nj); 

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
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // Step 1: spread from irregular points to regular grid
  timer.restart();
  spopts.debug = opts.spread_debug;
  spopts.sort = opts.spread_sort;
  spopts.spread_direction = 1;
  spopts.pirange = 1; FLT *dummy;
  spopts.chkbnds = 1;
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

int finufft2d2(INT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       INT ms, INT mt, CPX* fk, nufft_opts opts)

 /*  Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2 
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 

   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj,yj     x,y locations of sources on 2D domain [-pi,pi]^2.
     fk     FLT complex array of Fourier transform values (size ms*mt,
            increasing fast in ms then slow in mt, ie Fortran ordering),
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms,mt  numbers of Fourier modes given in x and y; each may be even or odd;
            in either case the mode range is integers lying in [-m/2, (m-1)/2].
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     size-nj complex FLT array of source strengths
     returned value - 0 if success, else:
                      1 : eps too small
		      2 : size of arrays to malloc exceed MAX_NF
                      other codes: as returned by cnufftspread

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
  INT64 nf1; set_nf_type12((BIGINT)ms,opts,spopts,&nf1);
  INT64 nf2; set_nf_type12((BIGINT)mt,opts,spopts,&nf2);
  if (nf1*nf2>MAX_NF) {
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("2d2: (ms,mt)=(%ld,%ld) (nf1,nf2)=(%ld,%ld) nj=%ld ...\n",(INT64)ms,(INT64)mt,nf1,nf2,(INT64)nj); 

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
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

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
  spopts.spread_direction = 2;
  FLT *dummy;
  int ier_spread = cnufftspread(nf1,nf2,1,(FLT*)fw,nj,xj,yj,dummy,(FLT*)cj,spopts);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  FFTW_FR(fw); free(fwkerhalf1); free(fwkerhalf2);
  if (opts.debug) printf("freed\n");
  return 0;
}

int finufft2d3(INT nj,FLT* xj,FLT* yj,CPX* cj,int iflag, FLT eps, INT nk, FLT* s, FLT *t, CPX* fk, nufft_opts opts)
 /*  Type-3 2D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j]),    for k=0,...,nk-1
               j=0
   Inputs:
     nj     number of sources (integer of type BIGINT; see utils.h)
     xj,yj  x,y location of sources in R^2.
     cj     size-nj complex FLT array of source strengths, 
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     nk     number of frequency target points
     s,t    (k_x,k_y) frequency locations of targets in R^2.
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     complex FLT Fourier transform values at the target frequencies sk
     returned value - 0 if success, else:
                      1 : eps too small
		      2 : size of arrays to malloc exceed MAX_NF
                      other codes: as returned by cnufftspread or finufft2d2

     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1. See [LG].
     Beyond this, the new twists are:
     i) number of upsampled points for the type-1 in each dim, depends on the
       product of interval widths containing input and output points (X*S), for
       that dim.
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf, in each dim.

   No references to FFTW are needed here. Some CPX arithmetic is used,
   thus compile with -Ofast in GNU.
   Barnett 2/17/17, 6/12/17
 */
{
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  INT64 nf1,nf2;
  FLT X1,C1,S1,D1,h1,gam1,X2,C2,S2,D2,h2,gam2;
  cout << scientific << setprecision(15);  // for debug

  // pick x, s intervals & shifts, then apply these to xj, cj (twist iii)...
  CNTime timer; timer.start();
  arraywidcen((BIGINT)nj,xj,&X1,&C1);  // get half-width, center, containing {x_j}
  arraywidcen((BIGINT)nk,s,&S1,&D1);   // {s_k}
  arraywidcen((BIGINT)nj,yj,&X2,&C2);  // {y_j}
  arraywidcen((BIGINT)nk,t,&S2,&D2);   // {t_k}
  // todo: if C1<X1/10 etc then set C1=0.0 and skip the slow-ish rephasing?
  set_nhg_type3(S1,X1,opts,spopts,&nf1,&h1,&gam1);          // applies twist i)
  set_nhg_type3(S2,X2,opts,spopts,&nf2,&h2,&gam2);
  if (opts.debug) printf("2d3: X1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%ld X2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%ld nj=%ld nk=%ld...\n",X1,C1,S1,D1,gam1,nf1,X2,C2,S2,D2,gam2,nf2,(INT64)nj,(INT64)nk);
  if (nf1*nf2>MAX_NF) {
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  FLT* xpj = (FLT*)malloc(sizeof(FLT)*nj);
  FLT* ypj = (FLT*)malloc(sizeof(FLT)*nj);
  for (BIGINT j=0;j<nj;++j) {
    xpj[j] = (xj[j]-C1) / gam1;          // rescale x_j
    ypj[j] = (yj[j]-C2) / gam2;          // rescale y_j
  }
  CPX imasign = (iflag>=0) ? ima : -ima;
  CPX* cpj = (CPX*)malloc(sizeof(CPX)*nj);  // c'_j rephased src
  if (D1!=0.0 || D2!=0.0) {
#pragma omp parallel for schedule(dynamic)               // since cexp slow
    for (BIGINT j=0;j<nj;++j)
      cpj[j] = cj[j] * exp(imasign*(D1*xj[j]+D2*yj[j])); // rephase c_j -> c'_j
    if (opts.debug) printf("prephase:\t\t %.3g s\n",timer.elapsedsec());
  } else
    for (BIGINT j=0;j<nj;++j)
      cpj[j] = cj[j];                                    // just copy over

  // Step 1: spread from irregular sources to regular grid as in type 1
  CPX* fw = (CPX*)malloc(sizeof(CPX)*nf1*nf2);
  timer.restart();
  spopts.spread_direction = 1;
  FLT *dummy;
  int ier_spread = cnufftspread(nf1,nf2,1,(FLT*)fw,nj,xpj,ypj,dummy,(FLT*)cpj,spopts);
  free(xpj); free(ypj); free(cpj);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  // Step 2: call type-2 to eval regular as Fourier series at rescaled targs
  timer.restart();
  FLT *sp = (FLT*)malloc(sizeof(FLT)*nk);     // rescaled targs s'_k
  FLT *tp = (FLT*)malloc(sizeof(FLT)*nk);     // t'_k
  for (BIGINT k=0;k<nk;++k) {
    sp[k] = h1*gam1*(s[k]-D1);                         // so that |s'_k| < pi/R
    tp[k] = h2*gam2*(t[k]-D2);                         // so that |t'_k| < pi/R
  }
  int ier_t2 = finufft2d2(nk,sp,tp,fk,iflag,eps,(INT)nf1,(INT)nf2,fw,opts);
  free(fw);
  if (opts.debug) printf("total type-2 (ier=%d):\t %.3g s\n",ier_t2,timer.elapsedsec());
  if (ier_t2) exit(ier_t2);

  // Step 3a: compute Fourier transform of scaled kernel at targets
  timer.restart();
  FLT *fkker1 = (FLT*)malloc(sizeof(FLT)*nk);
  FLT *fkker2 = (FLT*)malloc(sizeof(FLT)*nk);
  // exploit that Fourier transform separates because kernel built separable...
  onedim_nuft_kernel(nk, sp, fkker1, spopts);           // fill fkker1
  onedim_nuft_kernel(nk, tp, fkker2, spopts);           // fill fkker2
  if (opts.debug) printf("kernel FT (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());
  free(sp); free(tp);
  // Step 3b: correct for spreading by dividing by the Fourier transform from 3a
  timer.restart();
  if (isfinite(C1) && isfinite(C2) && (C1!=0.0 || C2!=0.0))
#pragma omp parallel for schedule(dynamic)              // since cexps slow
    for (BIGINT k=0;k<nk;++k)         // also phases to account for C1,C2 shift
      fk[k] *= (CPX)(1.0/(fkker1[k]*fkker2[k])) *
	exp(imasign*((s[k]-D1)*C1 + (t[k]-D2)*C2));
  else
#pragma omp parallel for schedule(dynamic)
    for (BIGINT k=0;k<nk;++k)         // also phases to account for C1,C2 shift
      fk[k] *= (CPX)(1.0/(fkker1[k]*fkker2[k]));
  if (opts.debug) printf("deconvolve:\t\t %.3g s\n",timer.elapsedsec());

  free(fkker1); free(fkker2); if (opts.debug) printf("freed\n");
  return 0;
}

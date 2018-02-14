#include "finufft.h"
#include "common.h"
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

int finufft1d1(INT nj,FLT* xj,CPX* cj,int iflag,FLT eps,INT ms,
	       CPX* fk, nufft_opts opts)
 /*  Type-1 1D complex nonuniform FFT.

              nj-1
     fk(k1) = SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2
              j=0                            
   Inputs:
     nj     number of sources (type INT; see utils.h)
     xj     location of sources on interval [-pi,pi].
     cj     size-nj FLT complex array of source strengths
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms     number of Fourier modes computed, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-ms FLT complex array of Fourier transform values
            (increasing mode ordering)
            stored as alternating Re & Im parts (2*ms FLTs)
     returned value - 0 if success, else:
                      1 : eps too small
		      2 : size of arrays to malloc exceed MAX_NF
                      other codes: as returned by cnufftspread

     The type 1 NUFFT proceeds in three main steps (see [GL]):
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the kernel
        Fourier series coeffs (not merely FFT of kernel), shuffle to output.

   Written with FFTW style complex arrays. Step 3a internally uses CPX,
   and Step 3b internally uses real arithmetic and FFTW style complex.
   Becuase of the former, compile with -Ofast in GNU.
   Barnett 1/22/17
 */
{
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  INT64 nf1; set_nf_type12((BIGINT)ms,opts,spopts,&nf1);
  if (nf1>MAX_NF) {
    fprintf(stderr,"nf1=%.3g exceeds MAX_NF of %.3g\n",(double)nf1,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("1d1: ms=%ld nf1=%ld nj=%ld ...\n",(INT64)ms,nf1,(INT64)nj);

  CNTime timer; timer.start();
  int nth = MY_OMP_GET_MAX_THREADS();
  if (nth>1) {             // set up multithreaded fftw stuff...
    FFTW_INIT();
    FFTW_PLAN_TH(nth);
  }
  FFTW_CPX *fw = FFTW_ALLOC_CPX(nf1);    // working upsampled array
  int fftsign = (iflag>=0) ? 1 : -1;
  FFTW_PLAN p = FFTW_PLAN_1D(nf1,fw,fw,fftsign, opts.fftw);  // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // Step 1: spread from irregular points to regular grid
  timer.restart();
  spopts.spread_direction = 1;
  FLT *dummy;
  int ier_spread = cnufftspread(nf1,1,1,(FLT*)fw,nj,xj,dummy,dummy,(FLT*)cj,spopts);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread, timer.elapsedsec());
  if (ier_spread>0) return ier_spread;
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  FFTW_EX(p);
  FFTW_DE(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n", nth, timer.elapsedsec());
  //for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // STEP 3a: get FT (series) of real symmetric spreading kernel
  timer.restart();
  FLT *fwkerhalf = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
  onedim_fseries_kernel(nf1, fwkerhalf, spopts);
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread, timer.elapsedsec());
  //for (int j=0;j<=nf1/2;++j) cout<<fwkerhalf[j]<<endl;

  // Step 3b: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  timer.restart();
  deconvolveshuffle1d(1,1.0,fwkerhalf,ms,(FLT*)fk,nf1,fw,opts.modeord);  // prefac now 1
  if (opts.debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());
  //for (int j=0;j<ms;++j) cout<<fk[j]<<endl;

  FFTW_FR(fw); free(fwkerhalf); if (opts.debug) printf("freed\n");
  return 0;
}


int finufft1d2(INT nj,FLT* xj,CPX* cj,int iflag,FLT eps,INT ms,
	       CPX* fk, nufft_opts opts)
 /*  Type-2 1D complex nonuniform FFT.

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1 
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

   Inputs:
     nj     number of targets
     xj     location of targets on interval [-pi,pi].
     fk     complex Fourier transform values (size ms, increasing mode ordering)
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     ms     number of Fourier modes input, may be even or odd;
            in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
     opts   struct controlling options (see finufft.h)
   Outputs:
     cj     complex FLT array of nj answers at targets
     returned value - 0 if success, else:
                      1 : eps too small
		      2 : size of arrays to malloc exceed MAX_NF
                      other codes: as returned by cnufftspread

     The type 2 algorithm proceeds in three main steps (see [GL]).
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.

   Written with FFTW style complex arrays. Step 0 internally uses CPX,
   and Step 1 internally uses real arithmetic and FFTW style complex.
   Because of the former, compile with -Ofast in GNU.
   Barnett 1/25/17
 */
{
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  INT64 nf1; set_nf_type12((BIGINT)ms,opts,spopts,&nf1);
  if (nf1>MAX_NF) {
    fprintf(stderr,"nf1=%.3g exceeds MAX_NF of %.3g\n",(double)nf1,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("1d2: ms=%ld nf1=%ld nj=%ld ...\n",(INT64)ms,nf1,(INT64)nj); 

  // STEP 0: get FT of real symmetric spreading kernel
  CNTime timer; timer.start();
  FLT *fwkerhalf = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
  onedim_fseries_kernel(nf1, fwkerhalf, spopts);
  if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread, timer.elapsedsec());

  int nth = MY_OMP_GET_MAX_THREADS();
  if (nth>1) {             // set up multithreaded fftw stuff...
    FFTW_INIT();
    FFTW_PLAN_TH(nth);
  }
  timer.restart();
  FFTW_CPX *fw = FFTW_ALLOC_CPX(nf1);    // working upsampled array
  int fftsign = (iflag>=0) ? 1 : -1;
  FFTW_PLAN p = FFTW_PLAN_1D(nf1,fw,fw,fftsign, opts.fftw); // in-place
  if (opts.debug) printf("fftw plan\t\t %.3g s\n", timer.elapsedsec());

  // STEP 1: amplify Fourier coeffs fk and copy into upsampled array fw
  timer.restart();
  deconvolveshuffle1d(2,1.0,fwkerhalf,ms,(FLT*)fk,nf1,fw,opts.modeord);
  free(fwkerhalf);        // in 1d could help to free up
  if (opts.debug) printf("amplify & copy in:\t %.3g s\n", timer.elapsedsec());
  //cout<<"fw:\n"; for (int j=0;j<nf1;++j) cout<<fw[j][0]<<"\t"<<fw[j][1]<<endl;

  // Step 2:  Call FFT
  timer.restart();
  FFTW_EX(p);
  FFTW_DE(p);
  if (opts.debug) printf("fft (%d threads):\t %.3g s\n", nth, timer.elapsedsec());

  // Step 3: unspread (interpolate) from regular to irregular target pts
  timer.restart();
  spopts.spread_direction = 2;
  FLT *dummy;
  int ier_spread = cnufftspread(nf1,1,1,(FLT*)fw,nj,xj,dummy,dummy,(FLT*)cj,spopts);
  //int ier_spread = twopispread1d(nf1,(CPX*)fw,nj,xj,cj,spopts);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n", ier_spread, timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  FFTW_FR(fw); if (opts.debug) printf("freed\n");
  return 0;
}


int finufft1d3(INT nj,FLT* xj,CPX* cj,int iflag, FLT eps, INT nk, FLT* s, CPX* fk, nufft_opts opts)
 /*  Type-3 1D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i s[k] xj[j]),      for k = 0, ..., nk-1
               j=0
   Inputs:
     nj     number of sources
     xj     location of sources in R (real line).
     cj     size-nj FLT complex array of source strengths
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     nk     number of frequency target points
     s      frequency locations of targets in R.
     iflag  if >=0, uses + sign in exponential, otherwise - sign.
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk FLT complex Fourier transform values at target
            frequencies sk
     returned value - 0 if success, else:
                      1 : eps too small
		      2 : size of arrays to malloc exceed MAX_NF
                      other codes: as returned by cnufftspread or finufft1d2

     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1. See [LG].
     Beyond this, the new twists are:
     i) nf1, number of upsampled points for the type-1, depends on the product
       of interval widths containing input and output points (X*S).
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf1.

   No references to FFTW are needed here. CPX arithmetic is used,
   thus compile with -Ofast in GNU.
   Barnett 2/7/17-6/9/17. 
 */
{
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  INT64 nf1;
  FLT X1,C1,S1,D1,h1,gam1;
  cout << scientific << setprecision(15);  // for debug

  // pick x, s intervals & shifts, then apply these to xj, cj (twist iii)...
  CNTime timer; timer.start();
  arraywidcen((BIGINT)nj,xj,&X1,&C1);  // get half-width, center, containing {x_j}
  arraywidcen((BIGINT)nk,s,&S1,&D1);   // get half-width, center, containing {s_k}
  set_nhg_type3(S1,X1,opts,spopts,&nf1,&h1,&gam1);          // applies twist i)
  if (opts.debug) printf("1d3: X1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%ld nj=%ld nk=%ld...\n",X1,C1,S1,D1,gam1,nf1,(INT64)nj,(INT64)nk);
  if (nf1>MAX_NF) {
    fprintf(stderr,"nf1=%.3g exceeds MAX_NF of %.3g\n",(double)nf1,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  FLT* xpj = (FLT*)malloc(sizeof(FLT)*nj);
  for (BIGINT j=0;j<nj;++j)
    xpj[j] = (xj[j]-C1) / gam1;                          // rescale x_j
  CPX imasign = (iflag>=0) ? ima : -ima;
  CPX* cpj = (CPX*)malloc(sizeof(CPX)*nj); // c'_j rephased src
  if (D1!=0.0) {
#pragma omp parallel for schedule(dynamic)               // since cexp slow
    for (BIGINT j=0;j<nj;++j)
      cpj[j] = cj[j] * exp(imasign*D1*xj[j]);            // rephase c_j -> c'_j
    if (opts.debug) printf("prephase:\t\t %.3g s\n",timer.elapsedsec());
  } else
    for (BIGINT j=0;j<nj;++j)
      cpj[j] = cj[j];                                    // just copy over
  
  // Step 1: spread from irregular sources to regular grid as in type 1
  CPX* fw = (CPX*)malloc(sizeof(CPX)*nf1);
  timer.restart();
  spopts.spread_direction = 1;
  FLT *dummy;
  int ier_spread = cnufftspread(nf1,1,1,(FLT*)fw,nj,xpj,dummy,dummy,(FLT*)cpj,spopts);
  free(xpj); free(cpj);
  if (opts.debug) printf("spread (ier=%d):\t\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;
  //for (int j=0;j<nf1;++j) printf("fw[%d]=%.3g\n",j,real(fw[j]));

  // Step 2: call type-2 to eval regular as Fourier series at rescaled targs
  timer.restart();
  FLT *sp = (FLT*)malloc(sizeof(FLT)*nk);     // rescaled targs s'_k
  for (BIGINT k=0;k<nk;++k)
    sp[k] = h1*gam1*(s[k]-D1);                         // so that |s'_k| < pi/R
  int ier_t2 = finufft1d2(nk,sp,fk,iflag,eps,(INT)nf1,fw,opts);  // the meat
  free(fw);
  if (opts.debug) printf("total type-2 (ier=%d):\t %.3g s\n",ier_t2,timer.elapsedsec());
  if (ier_t2) return ier_t2;
  //for (int k=0;k<nk;++k) printf("fk[%d]=(%.3g,%.3g)\n",k,real(fk[k]),imag(fk[k]));

  // Step 3a: compute Fourier transform of scaled kernel at targets
  timer.restart();
  FLT *fkker = (FLT*)malloc(sizeof(FLT)*nk);
  onedim_nuft_kernel(nk, sp, fkker, spopts);           // fill fkker
  if (opts.debug) printf("kernel FT (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());
  free(sp);
  // Step 3b: correct for spreading by dividing by the Fourier transform from 3a
  timer.restart();
  if (isfinite(C1) && C1!=0.0)
#pragma omp parallel for schedule(dynamic)              // since cexps slow
    for (BIGINT k=0;k<nk;++k)          // also phases to account for C1 x-shift
      fk[k] *= (CPX)(1.0/fkker[k]) * exp(imasign*(s[k]-D1)*C1);
  else
#pragma omp parallel for schedule(dynamic)
    for (BIGINT k=0;k<nk;++k)
      fk[k] *= (CPX)(1.0/fkker[k]);
  if (opts.debug) printf("deconvolve:\t\t %.3g s\n",timer.elapsedsec());

  free(fkker); if (opts.debug) printf("freed\n");
  return 0;
}

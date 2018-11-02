#include "finufft.h"
#include "common.h"
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,
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
     nj     number of sources (int64, aka BIGINT)
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

     The type 1 NUFFT proceeds in three main steps:
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

  if (opts.debug) printf("2d1: (ms,mt)=(%lld,%lld) (nf1,nf2)=(%lld,%lld) nj=%lld ...\n",(long long)ms,(long long)mt,(long long)nf1,(long long)nf2,(long long)nj);

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
  FLT *dummy=NULL;
  int ier_spread = spreadinterp(nf1,nf2,1,(FLT*)fw,nj,xj,yj,dummy,(FLT*)cj,spopts);
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


int finufft2d1many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c,
		   int iflag, FLT eps, BIGINT ms, BIGINT mt, CPX* fk,
		   nufft_opts opts)
/*
  Type-1 2D complex nonuniform FFT for multiple strength vectors, same NU pts.

                    nj
    f[k1,k2,d] =   SUM  c[j,d] exp(+-i (k1 x[j] + k2 y[j]))
                   j=1

    for -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, d = 0,...,ndata-1

    The output array is in increasing k1 ordering (fast), then increasing
    k2 ordering (slow), then increasing d (slowest). If iflag>0 the + sign
    is used, otherwise the - sign is used, in the exponential.
  Inputs:
    ndata  number of strength vectors
    nj     number of sources (int64, aka BIGINT)
    xj,yj  x,y locations of sources (each a size-nj FLT array) in [-3pi,3pi]
    c      a size nj*ndata complex FLT array of source strengths,
           increasing fast in nj then slow in ndata.
    iflag  if >=0, uses + sign in exponential, otherwise - sign.
    eps    precision requested (>1e-16)
    ms,mt  number of Fourier modes requested in x and y; each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2].
	   ms*mt must not exceed 2^31.
    opts   struct controlling options (see finufft.h)
  Outputs:
    fk     complex FLT array of Fourier transform values
           (size ms*mt*ndata, increasing fast in ms then slow in mt then in ndata
           ie Fortran ordering).
    returned value - 0 if success, else see ../docs/usage.rst

  Note: nthreads times the RAM is needed, so this is good only for small problems.
  By Melody Shih, originally called "manysimul" (many_seq=0 opt). Jun 2018.
 */
{
  if (ndata<1) {
    fprintf(stderr,"ndata should be at least 1 (ndata=%d)\n",ndata);
    return ERR_NDATA_NOTVALID;
  }
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  BIGINT nf1; set_nf_type12((BIGINT)ms,opts,spopts,&nf1);
  BIGINT nf2; set_nf_type12((BIGINT)mt,opts,spopts,&nf2);
  if (nf1*nf2>MAX_NF) {
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  cout << scientific << setprecision(15);  // for debug

  if (opts.debug) printf("2d1many: ndata=%d (ms,mt)=(%lld,%lld) (nf1,nf2)=(%lld,%lld) nj=%lld ...\n", ndata,(long long)ms,(long long)mt,(long long)nf1,(long long)nf2,(long long)nj);

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

  FFTW_CPX *fw = FFTW_ALLOC_CPX(nf1*nf2*nth);  // nthreads copies of upsampled array
  int fftsign = (iflag>=0) ? 1 : -1;
  const int n[] = {int(nf2), int(nf1)};
  // http://www.fftw.org/fftw3_doc/Row_002dmajor-Format.html#Row_002dmajor-Format

  timer.restart();
  FFTW_PLAN p = FFTW_PLAN_MANY_DFT(2, n, nth, fw, n, 1, n[0]*n[1], fw, n, 1,
                                   n[0]*n[1], fftsign, opts.fftw);
  if (opts.debug) printf("fftw plan (%d)    \t %.3g s\n",opts.fftw,timer.elapsedsec());

  spopts.debug = opts.spread_debug;
  spopts.sort = opts.spread_sort;
  spopts.spread_direction = 1;
  spopts.pirange = 1; FLT *dummy=NULL;
  spopts.chkbnds = opts.chkbnds;

  int ier_check = spreadcheck(nf1,nf2,1,nj,xj,yj,dummy,spopts);
  if (ier_check>0) return ier_check;
  
  timer.restart();          // sort
  BIGINT *sort_indices = (BIGINT*)malloc(sizeof(BIGINT)*nj);
  int did_sort = spreadsort(sort_indices,nf1,nf2,1,nj,xj,yj,dummy,spopts);
  if (opts.debug) printf("[many] sort (did_sort=%d):\t %.3g s\n", did_sort,
			 timer.elapsedsec());
  
  double time_fft = 0.0, time_spread = 0.0, time_deconv = 0.0;
  // since can't return within omp block, need this array to catch errors...
  int *ier_spreads = (int*)calloc(nth,sizeof(int));

#if _OPENMP
  // make sure only single threaded spreadinterp used for each data...
  MY_OMP_SET_NESTED(0);       // note this doesn't change omp_get_max_nthreads()
#endif
  
  for (int j = 0; j*nth < ndata; ++j) { // main loop over data blocks of size nth
    
    // Step 1: spread from irregular points to regular grid
    timer.restart();
    int blksize = min(ndata-j*nth,nth); // size of this block
#pragma omp parallel for
    for (int i=0; i<blksize; ++i) {
      CPX *cstart  = c + (i+j*nth)*nj;  // ptr to strengths this thread spreads
      FFTW_CPX *fwstart = fw + i*nf1*nf2;    // ptr to output grid for this thread
      int ier = spreadwithsortidx(sort_indices,nf1,nf2,1,(FLT*)fwstart,
				  nj,xj,yj,dummy,(FLT*)cstart,spopts,did_sort);
      if (ier!=0)
	ier_spreads[i] = ier;           // thank-you Melody for catching this
    }
    time_spread += timer.elapsedsec();
    for (int i = 0; i<blksize; ++i)         // exit if any thr had error
      if (ier_spreads[i]!=0)
        return ier_spreads[i];              // tell us one of these errors

    // Step 2:  Call FFT many
    timer.restart();
    FFTW_EX(p);                             // in-place, on all nth copies in fw
    time_fft += timer.elapsedsec();

    // Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
    timer.restart();
#pragma omp parallel for
    for (int i=0; i<blksize; ++i) {
      FFTW_CPX *fwstart = fw + i*nf1*nf2;    // this thr input
      CPX *fkstart = fk + (i+j*nth)*ms*mt;   // this thr output
      deconvolveshuffle2d(1,1.0,fwkerhalf1,fwkerhalf2,ms,mt,(FLT*)fkstart,nf1,
			  nf2,fwstart,opts.modeord);
    }
    time_deconv += timer.elapsedsec();
  }

  if (opts.debug) printf("[many] spread:\t\t\t %.3g s\n", time_spread);
  if (opts.debug) printf("[many] fft (%d threads):\t\t %.3g s\n", nth, time_fft);
  if (opts.debug) printf("[many] deconvolve & copy out:\t %.3g s\n", time_deconv);
  //  if (opts.debug) printf("[many] total execute time (exclude fftw_plan, etc.) %.3g s\n", time_spread+time_fft+time_deconv);

  FFTW_DE(p);
  FFTW_FR(fw); free(fwkerhalf1); free(fwkerhalf2); free(sort_indices);
  free(ier_spreads);
  if (opts.debug) printf("freed\n");
  return 0;
}


int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
	       BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)

 /*  Type-2 2D complex nonuniform FFT.

     cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))      for j = 0,...,nj-1
             k1,k2
     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

   Inputs:
     nj     number of targets (int64, aka BIGINT)
     xj,yj     x,y locations of targets (each a size-nj FLT array) in [-3pi,3pi]
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

     The type 2 algorithm proceeds in three main steps:
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

  if (opts.debug) printf("2d2: (ms,mt)=(%lld,%lld) (nf1,nf2)=(%lld,%lld) nj=%lld ...\n",(long long)ms,(long long)mt,(long long)nf1,(long long)nf2,(long long)nj);

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
  spopts.spread_direction = 2;
  FLT *dummy=NULL;
  int ier_spread = spreadinterp(nf1,nf2,1,(FLT*)fw,nj,xj,yj,dummy,(FLT*)cj,spopts);
  if (opts.debug) printf("unspread (ier=%d):\t %.3g s\n",ier_spread,timer.elapsedsec());
  if (ier_spread>0) return ier_spread;

  FFTW_FR(fw); free(fwkerhalf1); free(fwkerhalf2);
  if (opts.debug) printf("freed\n");
  return 0;
}


int finufft2d2many(int ndata, BIGINT nj, FLT* xj, FLT *yj, CPX* c, int iflag,
		   FLT eps, BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts)
/*
  Type-2 2D complex nonuniform FFT for multiple coeff vectors, same NU pts.

	     cj[j,d] =  SUM   fk[k1,k2,d] exp(+/-i (k1 xj[j] + k2 yj[j]))
	               k1,k2
	     for j = 0,...,nj-1,  d = 0,...,ndata-1
	     where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

  Inputs:
    ndata  number of mode coefficient vectors
    nj     number of targets (int64, aka BIGINT)
    xj,yj  x,y locations of targets (each a size-nj FLT array) in [-3pi,3pi]
    fk     FLT complex array of Fourier transform values (size ms*mt*ndata,
           increasing fast in ms then slow in mt then in ndata, ie Fortran
           ordering). Along each dimension the ordering is set by opts.modeord.
    iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
    eps    precision requested (>1e-16)
    ms,mt  numbers of Fourier modes given in x and y (int64)
           each may be even or odd;
           in either case the mode range is integers lying in [-m/2, (m-1)/2].
	   ms*mt must not exceed 2^31.
    opts   struct controlling options (see finufft.h)
  Outputs:
    cj     size-nj*ndata complex FLT array of target values, (ie, stored as
           2*nj*ndata FLTs interleaving Re, Im), increasing fast in nj then
           slow in ndata.
    returned value - 0 if success, else see ../docs/usage.rst

  Note: nthreads times the RAM is needed, so this is good only for small problems.
  By Melody Shih, originally called "manysimul" (many_seq=0 opt). Jun 2018.
*/
{
  if (ndata<1) {
    fprintf(stderr,"ndata should be at least 1 (ndata=%d)\n",ndata);
    return ERR_NDATA_NOTVALID;
  }
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

  if (opts.debug) printf("2d2: ndata=%d (ms,mt)=(%lld,%lld) (nf1,nf2)=(%lld,%lld) nj=%lld ...\n",
                         ndata,(long long)ms,(long long)mt,(long long)nf1,(long long)nf2,(long long)nj);

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

  FFTW_CPX *fw = FFTW_ALLOC_CPX(nf1*nf2*nth);  // nthreads copies of upsampled array
  int fftsign = (iflag>=0) ? 1 : -1;
  const int n[] = {int(nf2), int(nf1)};
  // http://www.fftw.org/fftw3_doc/Row_002dmajor-Format.html#Row_002dmajor-Format

  timer.restart();
  FFTW_PLAN p = FFTW_PLAN_MANY_DFT(2, n, nth, fw, n, 1, n[0]*n[1], fw, n, 1,
                                   n[0]*n[1], fftsign, opts.fftw);
  if (opts.debug) printf("fftw plan (%d)    \t %.3g s\n",opts.fftw,timer.elapsedsec());

  spopts.debug = opts.spread_debug;
  spopts.sort = opts.spread_sort;
  spopts.spread_direction = 2;
  spopts.pirange = 1; FLT *dummy=NULL;
  spopts.chkbnds = opts.chkbnds;

  int ier_check = spreadcheck(nf1,nf2,1,nj,xj,yj,dummy,spopts);
  if (ier_check>0) return ier_check;

  timer.restart();            // sort
  BIGINT* sort_indices = (BIGINT*)malloc(sizeof(BIGINT)*nj);
  int did_sort = spreadsort(sort_indices,nf1,nf2,1,nj,xj,yj,dummy,spopts);
  if (opts.debug) printf("[many] sort (did_sort=%d):\t %.3g s\n", did_sort,
			 timer.elapsedsec());

  double time_fft = 0.0, time_spread = 0.0, time_deconv = 0.0;
  // since can't return within omp block, need this array to catch errors...
  int *ier_spreads = (int*)calloc(nth,sizeof(int));

#if _OPENMP
  // make sure only single threaded spreadinterp used for each data...
  MY_OMP_SET_NESTED(0);       // note this doesn't change omp_get_max_nthreads()
#endif
  
  for (int j = 0; j*nth < ndata; ++j) {   // main loop over data blocks of size nth

    // STEP 1: amplify Fourier coeffs fk and copy into upsampled array fw
    timer.restart();
    int blksize = min(ndata-j*nth,nth);
#pragma omp parallel for
    for (int i = 0; i<blksize; ++i) {
      CPX *fkstart = fk + (i+j*nth)*ms*mt; // ptr to coeffs this thread copies
      FFTW_CPX* fwstart = fw + i*nf1*nf2;  // ptr to upsampled FFT array of this thread
      deconvolveshuffle2d(2,1.0,fwkerhalf1,fwkerhalf2,ms,mt,(FLT*)fkstart,nf1,nf2,
			  fwstart,opts.modeord);
    }
    time_deconv += timer.elapsedsec();

    // Step 2:  Call FFT many
    timer.restart();
    FFTW_EX(p);                             // in-place, on all nth copies in fw
    time_fft += timer.elapsedsec();

    // Step 3: unspread (interpolate) from regular to irregular target pts
    timer.restart();
#pragma omp parallel for
    for (int i=0; i<blksize; ++i) {
      FFTW_CPX *fwstart = fw + i*nf1*nf2;      // ptr to input values for thread
      CPX *cstart  = c + (i+j*nth)*nj;         // ptr to output vals for thread
      int ier = spreadwithsortidx(sort_indices,nf1,nf2,1,(FLT*)fwstart,nj,
				  xj,yj,dummy,(FLT*)cstart,spopts,did_sort);
      if (ier!=0)
	ier_spreads[i] = ier;           // thank-you Melody for catching this
    }
    time_spread+=timer.elapsedsec();
    for (int i = 0; i<blksize; ++i)         // exit if any thr had error
      if (ier_spreads[i]!=0)
        return ier_spreads[i];              // tell us one of these errors
  }
  if (opts.debug) printf("[many] amplify & copy in:\t %.3g s\n", time_deconv);
  if (opts.debug) printf("[many] fft (%d threads):\t\t %.3g s\n", nth, time_fft);
  if (opts.debug) printf("[many] unspread:\t\t %.3g s\n", time_spread);
  //if (opts.debug) printf("[many] total execute time (exclude fftw_plan, etc.) %.3g s\n",time_spread+time_fft+time_deconv);

  FFTW_FR(fw); free(fwkerhalf1); free(fwkerhalf2); free(sort_indices);
  free(ier_spreads);
  if (opts.debug) printf("freed\n");
  return 0;
}


int finufft2d3(BIGINT nj,FLT* xj,FLT* yj,CPX* cj,int iflag, FLT eps, BIGINT nk, FLT* s, FLT *t, CPX* fk, nufft_opts opts)
 /*  Type-3 2D complex nonuniform FFT.

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j]),    for k=0,...,nk-1
               j=0
   Inputs:
     nj     number of sources (int64, aka BIGINT)
     xj,yj  x,y location of sources in the plane R^2 (each size-nj FLT array)
     cj     size-nj complex FLT array of source strengths,
            (ie, stored as 2*nj FLTs interleaving Re, Im).
     nk     number of frequency target points (int64)
     s,t    (k_x,k_y) frequency locations of targets in R^2.
     iflag  if >=0, uses + sign in exponential, otherwise - sign (int)
     eps    precision requested (>1e-16)
     opts   struct controlling options (see finufft.h)
   Outputs:
     fk     size-nk complex FLT Fourier transform values at the
            target frequencies sk
     returned value - 0 if success, else see ../docs/usage.rst

     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1.
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

   No references to FFTW are needed here. Some CPX arithmetic is used.
   Barnett 2/17/17, 6/12/17
 */
{
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts,eps,opts);
  if (ier_set) return ier_set;
  BIGINT nf1,nf2;
  FLT X1,C1,S1,D1,h1,gam1,X2,C2,S2,D2,h2,gam2;
  cout << scientific << setprecision(15);  // for debug

  // pick x, s intervals & shifts, then apply these to xj, cj (twist iii)...
  CNTime timer; timer.start();
  arraywidcen(nj,xj,&X1,&C1);  // get half-width, center, containing {x_j}
  arraywidcen(nk,s,&S1,&D1);   // {s_k}
  arraywidcen(nj,yj,&X2,&C2);  // {y_j}
  arraywidcen(nk,t,&S2,&D2);   // {t_k}
  set_nhg_type3(S1,X1,opts,spopts,&nf1,&h1,&gam1);          // applies twist i)
  set_nhg_type3(S2,X2,opts,spopts,&nf2,&h2,&gam2);
  if (opts.debug) printf("2d3: X1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%lld X2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%lld nj=%lld nk=%lld...\n",X1,C1,S1,D1,gam1,(long long)nf1,X2,C2,S2,D2,gam2,(long long)nf2,(long long)nj,(long long)nk);
  if ((int64_t)nf1*nf2>MAX_NF) {
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
  }
  FLT* xpj = (FLT*)malloc(sizeof(FLT)*nj);
  FLT* ypj = (FLT*)malloc(sizeof(FLT)*nj);
  for (BIGINT j=0;j<nj;++j) {
    xpj[j] = (xj[j]-C1) / gam1;          // rescale x_j
    ypj[j] = (yj[j]-C2) / gam2;          // rescale y_j
  }
  CPX imasign = (iflag>=0) ? IMA : -IMA;
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
  FLT *dummy=NULL;
  int ier_spread = spreadinterp(nf1,nf2,1,(FLT*)fw,nj,xpj,ypj,dummy,(FLT*)cpj,spopts);
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
  int ier_t2 = finufft2d2(nk,sp,tp,fk,iflag,eps,nf1,nf2,fw,opts);
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

#include <finufft_legacy.h>
#include <invokeGuru.h>
#include <common.h>
#include <utils.h>
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

 */
{


  BIGINT n_modes[3] = {ms,mt,1};
  int n_dims = 2;
  int n_transf = 1;
  finufft_type type = type1;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj, iflag,
		      eps, n_modes, fk, opts);

  
  return ier; 
  
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

 */
{

  if (ndata<1) {
    fprintf(stderr,"ndata should be at least 1 (ndata=%d)\n",ndata);
    return ERR_NDATA_NOTVALID;
  }

  BIGINT n_modes[3] = {ms,mt,1};
  int n_dims = 2;
  finufft_type type = type1;
  
  int ier = invokeGuruInterface(n_dims, type, ndata, nj, xj, yj,NULL, c, iflag,
		      eps, n_modes, fk, opts);


  return ier; 
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

 */
{
  
  BIGINT n_modes[3] = {ms,mt,0};
  int n_dims = 2;
  int n_transf = 1;
  finufft_type type = type2;
  int ier = invokeGuruInterface(n_dims, type, n_transf, nj, xj, yj, NULL, cj, iflag,
		      eps, n_modes, fk, opts);
 
  
  return ier;
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

*/
{

  if (ndata<1) {
    fprintf(stderr,"ndata should be at least 1 (ndata=%d)\n",ndata);
    return ERR_NDATA_NOTVALID;
  }

  BIGINT n_modes[3] = {ms,mt,1};
  int n_dims = 2;
  finufft_type type = type2;
  int ier = invokeGuruInterface(n_dims, type, ndata, nj, xj, yj, NULL, c, iflag,
		      eps, n_modes, fk, opts);

  return ier; 
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
  if(!xpj){
    fprintf(stderr, "Call to malloc failed for x source coordinate array allocation!");
    return ERR_MAXNALLOC;
  }
  FLT* ypj = (FLT*)malloc(sizeof(FLT)*nj);
  if(!ypj){
    fprintf(stderr, "Call to malloc failed for y source coordinate array allocation!");
    free(xpj);
    return ERR_MAXNALLOC;
  }

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

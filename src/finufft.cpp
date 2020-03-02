#include <finufft.h>
#include <utils.h>
#include <iostream>
#include <common.h>
#include <iomanip>

/* The main guru functions for FINUFFT.

   Guru interface written by Andrea Malleo, summer 2019, with help from
   Alex Barnett.
   As of v1.2 these replace the old hand-coded separate 9 finufft?d?() functions
   and the two finufft2d?many() functions.
   The (now 18) simple interfaces are in simpleinterfaces.cpp

Notes on algorithms taken from old finufft?d?() documentation, Feb-Jun 2017:

   TYPE 1:
     The type 1 NUFFT proceeds in three main steps:
     1) spread data to oversampled regular mesh using kernel.
     2) compute FFT on uniform mesh
     3) deconvolve by division of each Fourier mode independently by the kernel
        Fourier series coeffs (not merely FFT of kernel), shuffle to output.
     The kernel coeffs are precomputed in what is called step 0 in the code.
   Written with FFTW style complex arrays. Step 3a internally uses CPX,
   and Step 3b internally uses real arithmetic and FFTW style complex.

   TYPE 2:
     The type 2 algorithm proceeds in three main steps:
     1) deconvolve (amplify) each Fourier mode, dividing by kernel Fourier coeff
     2) compute inverse FFT on uniform fine grid
     3) spread (dir=2, ie interpolate) data to regular mesh
     The kernel coeffs are precomputed in what is called step 0 in the code.
   Written with FFTW style complex arrays. Step 0 internally uses CPX,
   and Step 1 internally uses real arithmetic and FFTW style complex.

   TYPE 3:
     The type 3 algorithm is basically a type 2 (which is implemented precisely
     as call to type 2) replacing the middle FFT (Step 2) of a type 1.
     Beyond this, the new twists are:
     i) nf1, number of upsampled points for the type-1, depends on the product
       of interval widths containing input and output points (X*S).
     ii) The deconvolve (post-amplify) step is division by the Fourier transform
       of the scaled kernel, evaluated on the *nonuniform* output frequency
       grid; this is done by direct approximation of the Fourier integral
       using quadrature of the kernel function times exponentials.
     iii) Shifts in x (real) and s (Fourier) are done to minimize the interval
       half-widths X and S, hence nf1.
   No references to FFTW are needed here. CPX arithmetic is used.

   MULTIPLE STRENGTH VECTORS FOR THE SAME NONUNIFORM POINTS (n_transf>1):
     blksize (set to max_num_omp_threads) times the RAM is needed, so this is
     good only for small problems.

Design notes for guru interface implementation:

* Since finufft_plan is C-compatible, we need to use malloc/free for its
  allocatable arrays, keeping it quite low-level. We can't use std::vector
  since the only survive in the scope of each function.

*/


int* n_for_fftw(finufft_plan* p){
// helper func returns a new int array of length n_dims, extracted from
// the finufft plan, that fft_many_many_dft needs as its 2nd argument.
  int* nf;
  if(p->n_dims == 1){ 
    nf = new int[1];
    nf[0] = (int)p->nf1;
  }
  else if (p->n_dims == 2){ 
    nf = new int[2];
    nf[0] = (int)p->nf2;
    nf[1] = (int)p->nf1; 
  }   // fftw enforced row major ordering, ie dims are backwards ordering
  else{ 
    nf = new int[3];
    nf[0] = (int)p->nf3;
    nf[1] = (int)p->nf2;
    nf[2] = (int)p->nf1;
  }
  return nf;
}


// PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
int finufft_makeplan(int type, int n_dims, BIGINT* n_modes, int iflag,
                     int n_transf, FLT tol, int threadBlkSize,
                     finufft_plan* p, nufft_opts* opts)
// Populates the fields of finufft_plan which is pointed to by "p".
// opts is ptr to a nufft_opts to choose options, or NULL to use defaults.
// For types 1,2 allocates memory for internal working arrays,
// evaluates spreading kernel coefficients, and instantiates the fftw_plan
{  
  cout << scientific << setprecision(15);  // for debug outputs

  if((type!=1)&&(type!=2)&&(type!=3)) {
    fprintf(stderr, "Invalid type (%d), type should be 1, 2 or 3.",type);
    return ERR_TYPE_NOTVALID;
  }
  if((n_dims!=1)&&(n_dims!=2)&&(n_dims!=3)) {
    fprintf(stderr, "Invalid n_dims (%d), should be 1, 2 or 3.",n_dims);
    return ERR_DIM_NOTVALID;
  }
  if (n_transf<1) {
    fprintf(stderr,"n_transf (%d) should be at least 1.\n",n_transf);
    return ERR_NDATA_NOTVALID;
  }

  if (opts==NULL)                        // use default opts
    finufft_default_opts(&(p->opts));
  else                                   // or read from what's passed in
    p->opts = *opts;    // does deep copy; changing *opts now has no effect
  // write into plan's spread options...
  int ier_set = setup_spreader_for_nufft(p->spopts, tol, p->opts);
  if (ier_set)
    return ier_set;

  // get stuff from args...
  p->type = type;
  p->n_dims = n_dims;
  p->n_transf = n_transf;
  p->tol = tol;
  p->fftsign = (iflag>=0) ? 1 : -1;          // clean up flag input
  if (threadBlkSize==0)            // set default
    p->threadBlkSize = min(MY_OMP_GET_MAX_THREADS(), MAX_USEFUL_NTHREADS);
  else
    p->threadBlkSize = threadBlkSize;

  // set others as defaults (or unallocated for arrays)...
  p->X = NULL;
  p->Y = NULL;
  p->Z = NULL;
  p->X_orig = NULL;
  p->Y_orig = NULL;
  p->Z_orig = NULL;
  p->sp = NULL;
  p->tp = NULL;
  p->up = NULL;
  p->nf1 = 1;             // crucial to leave as 1 for unused dims
  p->nf2 = 1;
  p->nf3 = 1;
  p->isInnerT2 = false;
  p->ms = 1;             // crucial to leave as 1 for unused dims
  p->mt = 1;
  p->mu = 1;

  //  ------------------------ types 1,2: planning needed ---------------------
  if((type == 1) || (type == 2)) {

    int nth = MY_OMP_GET_MAX_THREADS();    // tell FFTW what it has access to
    FFTW_INIT();           // only does anything when OMP=ON for >1 threads
    FFTW_PLAN_TH(nth);
    p->spopts.spread_direction = type;
    // do no more work than necessary if the number of transforms is smaller than the the threadBlkSize...
    int transfPerBatch = min(p->threadBlkSize, p->n_transf); 
    
    // read in mode array dims then determine fine grid sizes, sanity check...
    p->ms = n_modes[0];
    int ier_nf = set_nf_type12(p->ms,p->opts,p->spopts,&(p->nf1));
    if (ier_nf) return ier_nf;  // nf too big; we're outta here
    if (n_dims > 1) {
      p->mt = n_modes[1];
      ier_nf = set_nf_type12(p->mt, p->opts, p->spopts, &(p->nf2));
      if (ier_nf) return ier_nf;
    }
    if (n_dims > 2) {
      p->mu = n_modes[2];
      ier_nf = set_nf_type12(p->mu, p->opts, p->spopts, &(p->nf3)); 
      if (ier_nf) return ier_nf;
    }

    if (p->opts.debug)
      printf("[finufft_plan] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) (nf1,nf2,nf3)=(%lld,%lld,%lld) batch=%d\n",
             n_dims, type, (long long)p->ms,(long long)p->mt,
             (long long) p->mu, (long long)p->nf1,(long long)p->nf2,
             (long long)p->nf3,transfPerBatch);

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid dim
    CNTime timer; timer.start(); 
    BIGINT NphiHat;    // size of concatenated kernel Fourier xforms
    NphiHat = p->nf1/2 + 1; 
    if (n_dims > 1) NphiHat += (p->nf2/2 + 1);
    if (n_dims > 2) NphiHat += (p->nf3/2 + 1);
    p->phiHat = (FLT *)malloc(sizeof(FLT)*NphiHat);
    if(!p->phiHat){
      fprintf(stderr, "finufft_plan: malloc failed for Fourier coeff array!");
      return ERR_ALLOC;
    }
    onedim_fseries_kernel(p->nf1, p->phiHat, p->spopts);
    if (n_dims > 1)       // stack the 2nd dim of phiHat
      onedim_fseries_kernel(p->nf2, p->phiHat + (p->nf1/2+1), p->spopts);
    if (n_dims > 2)
      onedim_fseries_kernel(p->nf3, p->phiHat + (p->nf1/2+1) + (p->nf2/2+1), p->spopts);
    if (p->opts.debug) printf("[finufft_plan] kernel fser (ns=%d):\t\t %.3g s\n", p->spopts.nspread,timer.elapsedsec());    

    BIGINT nfTotal = p->nf1*p->nf2*p->nf3;   // each fine grid size
    // ensure size of upsampled grid does not exceed MAX, otherwise give up
    if (nfTotal*transfPerBatch>MAX_NF) { 
      fprintf(stderr,"nf1*nf2*nf3*p->threadBlkSize=%.3g exceeds MAX_NF of %.3g\n",
              (double)nfTotal*transfPerBatch,(double)MAX_NF);
      return ERR_MAXNALLOC;
    }
    p->fw = FFTW_ALLOC_CPX(nfTotal*transfPerBatch);
    if(!p->fw){
      fprintf(stderr, "fftw malloc failed for working upsampled array(s)\n");
      free(p->phiHat);
      return ERR_ALLOC; 
    }
   
    timer.restart();            // plan the FFTW
    int *n = n_for_fftw(p);
    // fftw_plan_many_dft args: rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
    p->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, n, transfPerBatch, p->fw,
                     NULL, 1, nfTotal, p->fw, NULL, 1,
                     nfTotal, p->fftsign, p->opts.fftw);
    if (p->opts.debug) printf("[finufft_plan] fftw plan (mode %d):\t\t %.3g s\n",p->opts.fftw,timer.elapsedsec());
    delete []n;
    
  } else {  // -------------------------- type 3 (no planning) ----------------
    if (p->opts.debug) printf("[finufft_plan] %dd%d\n",n_dims,type);
    p->fftwPlan = NULL;
    // type 3 will call finufft_makeplan for type 2, thus no need to init FFTW
  }
  return 0;
}


// SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
int finufft_setpts(finufft_plan* p, BIGINT nj, FLT* xj, FLT* yj, FLT* zj,
                   BIGINT nk, FLT* s, FLT* t, FLT* u)
// For type 1,2: just checks and sorts the NU points.
// For type 3: allocates internal working arrays, scales/centers the NU points
// and NU target freqs, evaluates spreading kernel FT at all target freqs.
{
  CNTime timer; timer.start();
  p->nj = nj;    // the user choosing how many NU (x,y,z) pts
  
  if (p->type!=3) {   // ------------------ TYPE 1,2 SETPTS ---------------
    
    int ier_check = spreadcheck(p->nf1,p->nf2 , p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (ier_check) return ier_check;
    if (p->opts.debug>1) printf("[finufft_setpts] spreadcheck (%d):\t %.3g s\n", p->spopts.chkbnds, timer.elapsedsec());
    
    timer.restart();
    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*p->nj);
    p->didSort = indexSort(p->sortIndices, p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->opts.debug) printf("[finufft_setpts] sort (did_sort=%d):\t %.3g s\n", p->didSort, timer.elapsedsec());
    
    p->X = xj; // we just point to user's data, which must be length >=nj
    p->Y = yj;
    p->Z = zj;
    p->s=NULL;
    p->t= NULL;
    p->u = NULL;
    
  } else {    // ------------------------- TYPE 3 SETPTS ---------------------
    
    p->nk = nk;     // user set # targ freq pts
    int transfPerBatch = min(p->threadBlkSize, p->n_transf); // batch size

    // pick x, s intervals & shifts & # fine grid pts (nf) in each dim...
    CNTime timer; timer.start();
    FLT S1,S2,S3;       // get half-width X, center C, which contains {x_j}...
    arraywidcen(p->nj,xj,&(p->t3P.X1),&(p->t3P.C1));
    arraywidcen(p->nk,s,&S1,&(p->t3P.D1));   // same D, S, but for {s_k}
    set_nhg_type3(S1,p->t3P.X1,p->opts,p->spopts,
           &(p->nf1),&(p->t3P.h1),&(p->t3P.gam1));  // applies twist i)
    if(p->n_dims > 1){
      arraywidcen(p->nj,yj,&(p->t3P.X2),&(p->t3P.C2));  // {y_j}
      arraywidcen(p->nk,t,&S2,&(p->t3P.D2));               // {t_k}
      set_nhg_type3(S2,p->t3P.X2,p->opts,p->spopts,&(p->nf2),
                    &(p->t3P.h2),&(p->t3P.gam2));
    }    
    if(p->n_dims > 2){
      arraywidcen(p->nj,zj,&(p->t3P.X3),&(p->t3P.C3));  // {z_j}
      arraywidcen(p->nk,u,&S3,&(p->t3P.D3));               // {u_k}
      set_nhg_type3(S3,p->t3P.X3,p->opts,p->spopts,
                    &(p->nf3),&(p->t3P.h3),&(p->t3P.gam3));
    }

    if (p->opts.debug){  // report on choices of shifts, centers, etc...
      printf("%dd3: X1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%lld M=%lld N=%lld\n", p->n_dims, p->t3P.X1, p->t3P.C1,S1, p->t3P.D1, p->t3P.gam1,(long long) p->nf1, (long long)p->nj,(long long)p->nk);  
      if(p->n_dims > 1) printf("     X2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%lld \n",p->t3P.X2, p->t3P.C2,S2, p->t3P.D2, p->t3P.gam2,(long long) p->nf2);
      if(p->n_dims > 2) printf("     X3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%lld \n", p->t3P.X3, p->t3P.C3,S3, p->t3P.D3, p->t3P.gam3,(long long) p->nf3);
    }

    // alloc rescaled NU src pts x'_j...
    FLT* xpj = (FLT*)malloc(sizeof(FLT)*p->nj);
    if(!xpj){
      fprintf(stderr, "malloc failed for type-3 rescaled x coordinates\n");
      return ERR_ALLOC; 
    }    
    FLT* ypj = NULL;
    if(p->n_dims > 1){
      ypj = (FLT*)malloc(sizeof(FLT)*nj);
      if(!ypj){
        fprintf(stderr, "malloc failed for type-3 rescaled y coordinates\n");
        return ERR_ALLOC; 
      }
    } 
    FLT* zpj = NULL;
    if(p->n_dims > 2){
      zpj = (FLT*)malloc(sizeof(FLT)*nj);
      if(!zpj){
        fprintf(stderr, "malloc failed for type-3 rescaled z coordinates\n");
        return ERR_ALLOC; 
      }
    }
    // always shift as use gam to rescale x_j to x'_j, etc (twist iii)...
#pragma omp parallel for schedule(static)
    for (BIGINT j=0;j<nj;++j) {
      xpj[j] = (xj[j] - p->t3P.C1) / p->t3P.gam1;         // rescale x_j
      if(p->n_dims > 1)
        ypj[j] = (yj[j]- p->t3P.C2) / p->t3P.gam2;        // rescale y_j
      if(p->n_dims > 2)
        zpj[j] = (zj[j] - p->t3P.C3) / p->t3P.gam3;       // rescale z_j
    }

    // alloc (batch of) working fine grid(s)...
    if ((int64_t)p->nf1*p->nf2*p->nf3*transfPerBatch > MAX_NF) {
      fprintf(stderr,"nf1*nf2*nf3*batch=%.3g exceeds MAX_NF of %.3g\n",(double)p->nf1*p->nf2*p->nf3*transfPerBatch,(double)MAX_NF);
      return ERR_MAXNALLOC;
    }
    p->fw = FFTW_ALLOC_CPX(p->nf1*p->nf2*p->nf3*transfPerBatch);  
    if(!p->fw){
      fprintf(stderr, "fftw malloc fail for t3 working upsampled grid fw\n");
      return ERR_ALLOC; 
    }
    if (p->opts.debug) printf("[finufft_setpts] t3 alloc & x scale:\t %.3g s\n",timer.elapsedsec());

    timer.restart();  // sort for spread to fine grid, Step 1 (no bndschk needed)
    p->spopts.spread_direction = 1;
    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*p->nj);
    p->didSort = indexSort(p->sortIndices, p->nf1, p->nf2, p->nf3, p->nj, xpj, ypj, zpj, p->spopts);
    if (p->opts.debug) printf("[finufft_setpts] sort (did_sort=%d):\t\t %.3g s\n", p->didSort, timer.elapsedsec());
    
    p->X = xpj; p->Y = ypj; p->Z = zpj;             // rescaled x' to feed to t2
    p->X_orig = xj; p->Y_orig = yj; p->Z_orig = zj; // keep unscaled x   *** NEED?

    timer.restart();           // Allocate rescaled targets s'_k, etc...
    FLT *sp = (FLT*)malloc(sizeof(FLT)*p->nk);       // s'_k
    if (!sp){
      fprintf(stderr, "Call to malloc failed for rescaled s target freqs\n");
      return ERR_ALLOC;
    }    
    FLT *tp = NULL;
    if(p->n_dims > 1){
      tp = (FLT*)malloc(sizeof(FLT)*p->nk);          // t'_k
      if (!tp){
        fprintf(stderr, "Call to malloc failed for rescaled t target freqs\n");
        return ERR_ALLOC; 
      }
    }
    FLT *up = NULL;
    if(p->n_dims > 2){
      up = (FLT*)malloc(sizeof(FLT)*p->nk);          // u'_k
      if (!up){
        fprintf(stderr, "Call to malloc failed for rescaled u target freqs\n");
        return ERR_ALLOC;
      }
    }
#pragma omp parallel for schedule(static)
    for (BIGINT k=0;k<p->nk;++k) {    // loop to rescale all targs ........
      sp[k] = p->t3P.h1*p->t3P.gam1*(s[k]-p->t3P.D1);  // so that |s'_k| < pi/R
      if (p->n_dims > 1)
        tp[k] = p->t3P.h2*p->t3P.gam2*(t[k]-p->t3P.D2);
      if (p->n_dims > 2)
        up[k] = p->t3P.h3*p->t3P.gam3*(u[k]-p->t3P.D3);
    }
    if (p->opts.debug) printf("[finufft_setpts] rescaling NU targ freqs:\t %.3g s\n", timer.elapsedsec());

    // precompute Step 3a: alloc & fill phiHat, scaled kernel FT at NU targs...
    timer.restart();
    p->phiHat = (FLT*)malloc(sizeof(FLT)*p->nk * p->n_dims);  // *** shrink!

    // phiHat spreading kernel fourier weights for non uniform target freqs := referred to as fkker in older code
    onedim_nuft_kernel(p->nk, sp, p->phiHat, p->spopts);         
    if(p->n_dims > 1)
      onedim_nuft_kernel(p->nk, tp, p->phiHat + p->nk, p->spopts);           
    if(p->n_dims > 2)
      onedim_nuft_kernel(p->nk, up, p->phiHat + 2*p->nk, p->spopts);
    if (p->opts.debug) printf("[finufft_setpts] kernel FT (ns=%d):\t\t %.3g s\n", p->spopts.nspread,timer.elapsedsec());

    //precompute product of phiHat for 2 and 3 dimensions 
    if(p->n_dims > 1){
#pragma omp parallel for schedule(static)              
      for(BIGINT k=0; k < p->nk; k++)
        p->phiHat[k]*=(p->phiHat+p->nk)[k];
    }

    if(p->n_dims > 2){
#pragma omp parallel for schedule(static)              
      for(BIGINT k=0; k < p->nk; k++)
        p->phiHat[k]*=(p->phiHat + 2*p->nk)[k];
    }

    plan->s = s;
    plan->sp = sp;
    
    //NULL if 1 dim
    plan->t = t;
    plan->tp = tp;
    
    //NULL if 2 dim
    plan->u = u;
    plan->up = up;

    BIGINT n_modes[3];
    n_modes[0] = plan->nf1;
    n_modes[1] = plan->nf2;
    n_modes[2] = plan->nf3;

    int ier_t2;
    
    // Preparations for the interior type 2 finufft call
    // 1) a single call to construct a finufft_plan
    // 2) a single call to finufft_setpts where scaled target freqs become the type2 x,y,z coordinates 
    finufft_plan *t2Plan = new finufft_plan;
    finufft_default_opts(&(t2Plan->opts));
    t2Plan->opts.debug = plan->opts.debug;
    t2Plan->opts.spread_debug = plan->opts.spread_debug;

    int batchSize = min(plan->n_transf, plan->threadBlkSize);
    timer.restart();
    ier_t2 = finufft_makeplan(2, plan->n_dims, n_modes, plan->fftsign, batchSize, plan->tol,
                              plan->threadBlkSize, t2Plan, &t2Plan->opts);
    double t_innerPlan = timer.elapsedsec();
    
    if(ier_t2){
      printf("inner type 2 plan creation failed\n");
      return ier_t2;  
    }
    
    t2Plan->isInnerT2 = true;

    timer.restart();
    ier_t2 = finufft_setpts(t2Plan, plan->nk, plan->sp, plan->tp, plan->up, 0, NULL, NULL, NULL);
    double t_innerSet = timer.elapsedsec();
    if(ier_t2){
      printf("inner type 2 set points failed\n");
      return ier_t2;
    }

    plan->innerT2Plan = t2Plan;

    if(plan->opts.debug){
      printf("[finufft_setpts] type-3 inner type2 plan creation: %.3g s\n", t_innerPlan+t_innerSet);
    }
  }
  
  return 0;
}
// ............ end setpts ..................................................



/////////////////////////////////////////////////////////////////////////////
void spreadAllSetsInBatch(int nSetsThisBatch, int blkNum, finufft_plan* p, CPX* c, int* ier_spreads){
  // Type 1 + Type 3: Spreads coordinate weights from c into internal workspace
  // fw for sending into fftw

  // nSetsThisBatch is the threadBlockSize, except for the last round if
  // threadBlockSize does not divide evenly into n_transf, prevents c overrun.

  BIGINT fwRowSize = p->nf1*p->nf2*p->nf3; 
  int blkJump = blkNum*p->threadBlkSize; 

  // default sequential maximum multithreaded: execute
  //the for-loop down below on THIS thread (spawn no others)
  //and leave all the multithreading for inside of the spreadSorted call
  int n_outerThreads = 0;
  if(p->opts.spread_scheme==1) // simultaneous singlethreaded/nested multi
    n_outerThreads = nSetsThisBatch; //spawn as many threads as sets, if fewer sets than available threads
                                     //the extra threads used for work inside of spreadSorted 
  
  MY_OMP_SET_NESTED(1); 
#pragma omp parallel for num_threads(n_outerThreads)
  for(int i = 0; i < nSetsThisBatch; i++){ 

    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = p->fw + fwRowSize*i;

    //for type 3, c is "cpj", scaled weights, and spreading is done in batches of size threadBlockSize
    CPX *cStart;
    if(p->type == 3)
      cStart = c + p->nj*i;

    //for type1+2, c is the client's array and of size nj*n_transforms
    else
      cStart = c + p->nj*(i + blkJump); 
    
    int ier = spreadSorted(p->sortIndices,
                           p->nf1, p->nf2, p->nf3, (FLT*)fwStart,
                           p->nj, p->X, p->Y, p->Z, (FLT *)cStart,
                           p->spopts, p->didSort) ;
    if(ier)
      ier_spreads[i] = ier;
  }
  MY_OMP_SET_NESTED(0);
}

void interpAllSetsInBatch(int nSetsThisBatch, int batchNum, finufft_plan* p, CPX* c, int* ier_interps)
// Type 2: Interpolates from weights at uniform points in fw to non uniform points in c
{
  BIGINT fwRowSize =  p->nf1*p->nf2*p->nf3;
  int blkJump = batchNum*p->threadBlkSize; 

  //default sequential maximum multithreaded: execute
  //the for-loop down below on THIS thread (spawn no others)
  //and leave all the multithreading for inside of the interpSorted call
  int n_outerThreads = 0;
  if(p->opts.spread_scheme){
    //spread_scheme == 1 -> simultaneous singlethreaded/nested multi
    n_outerThreads = nSetsThisBatch; //spawn as many threads as sets, if fewer sets than available threads
                                     //the extra threads used for work inside of spreadSorted 
  }
  
  MY_OMP_SET_NESTED(1);
#pragma omp parallel for num_threads(n_outerThreads)
  for(int i = 0; i < nSetsThisBatch; i++){ 
        
    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = p->fw + fwRowSize*i; //fw gets reread on each iteration of j

    CPX * cStart;

    //If this is a type 2 being executed inside of a type 3, c is an internal array of size nj*threadBlockSize
    if(p->isInnerT2)
      cStart = c + p->nj*i;

    //for type 1+ regular 2, c is the result array, size nj*n_transforms
    else
      cStart = c + p->nj*(i + blkJump);

    int ier = interpSorted(p->sortIndices,
                           p->nf1, p->nf2, p->nf3, (FLT*)fwStart,
                           p->nj, p->X, p->Y, p->Z, (FLT *)cStart,
                           p->spopts, p->didSort) ;

    if(ier)
      ier_interps[i] = ier;
  }
  MY_OMP_SET_NESTED(0);
}

void deconvolveInParallel(int nSetsThisBatch, int batchNum, finufft_plan* p, CPX* fk)
/* Type 1: deconvolves (amplifies) from interior fw array into user-supplied fk.
   Type 2: deconvolves from user-supplied fk into interior fw array.
   This is mostly a parallel loop calling deconvolveshuffle?d in the needed dim.
*/
{
  // phiHat = kernel FT arrays (stacked version of fwker in 2017 code)
  FLT* phiHat1 = p->phiHat, *phiHat2=NULL, *phiHat3=NULL;
  if(p->n_dims > 1)
    phiHat2 = p->phiHat + p->nf1/2 + 1;
  if(p->n_dims > 2)
    phiHat3 = p->phiHat+(p->nf1/2+1)+(p->nf2/2+1);

  BIGINT fkRowSize = p->ms*p->mt*p->mu;
  BIGINT fwRowSize = p->nf1*p->nf2*p->nf3;
  int blockJump = batchNum*p->threadBlkSize;

#pragma omp parallel for
  for(int i = 0; i < nSetsThisBatch; i++) {
    CPX *fkStart;

    //If this is a type 2 being executed inside of a type 3, fk is internal array of size nj*threadBlockSize
    if(p->isInnerT2)
      fkStart = fk + i*fkRowSize;

    //otherwise it is a user supplied array of size ms*mt*mu*n_transforms
    else
      fkStart = fk + (i+blockJump)*fkRowSize;
    
    FFTW_CPX *fwStart = p->fw + fwRowSize*i;

    //deconvolveshuffle?d are not multithreaded inside, so called in parallel here
    //prefactors hardcoded to 1...
    if(p->n_dims == 1)
      deconvolveshuffle1d(p->spopts.spread_direction, 1.0, phiHat1,
                          p->ms, (FLT *)fkStart,
                          p->nf1, fwStart, p->opts.modeord);
    else if (p->n_dims == 2)
      deconvolveshuffle2d(p->spopts.spread_direction,1.0, phiHat1, phiHat2,
                          p->ms, p->mt, (FLT *)fkStart,
                          p->nf1, p->nf2, fwStart, p->opts.modeord);
    else
      deconvolveshuffle3d(p->spopts.spread_direction, 1.0, phiHat1, phiHat2,
                          phiHat3, p->ms, p->mt, p->mu,
                          (FLT *)fkStart, p->nf1, p->nf2, p->nf3,
                          fwStart, p->opts.modeord);
  }
}


void type3PrePhaseInParallel(int nSetsThisBatch, int batchNum, finufft_plan* p, CPX* cj, CPX* cpj)
// Type 3 multithreaded prephase all nj scaled weights for all of the sets of weights in this batch
// occurs inplace of internal finufft array cpj (sized nj*threadBlkSize)
{

  bool notZero = p->t3P.D1 != 0.0;
  if(p->n_dims > 1) notZero |=  (p->t3P.D2 != 0.0);
  if(p->n_dims > 2) notZero |=  (p->t3P.D3 != 0.0);
    
  // note that schedule(dynamic) actually slows this down (was in v<=1.1.2):
#pragma omp parallel for
  for (BIGINT i=0; i<p->nj;i++){

    FLT sumCoords = p->t3P.D1*p->X_orig[i];
    if(p->n_dims > 1)
      sumCoords += p->t3P.D2*p->Y_orig[i];
    if(p->n_dims > 2)
      sumCoords += p->t3P.D3*p->Z_orig[i];
          
    CPX multiplier = exp((FLT)p->fftsign * IMA*sumCoords); // rephase
          
    for(int k = 0; k < nSetsThisBatch; k++){ // *** jumps around in RAM - speed?
      int cpjIndex = k*p->nj + i;
      int cjIndex = batchNum*p->threadBlkSize*p->nj + cpjIndex;

      cpj[cpjIndex] = cj[cjIndex]; 
      if(notZero)
	cpj[cpjIndex] *= multiplier;
    }
  }
}


void type3DeconvolveInParallel(int nSetsThisBatch, int batchNum, finufft_plan *p, CPX *fk)
// Type 3 multithreaded in-place deconvolve of user supplied result array fk
// (size nk*n_transf). Ie, do step 3b.
{  
  bool Cfinite  = isfinite(p->t3P.C1);          // *** is this needed?
  if(p->n_dims > 1 ) Cfinite &= isfinite(p->t3P.C2);
  if(p->n_dims > 2 ) Cfinite &= isfinite(p->t3P.C3);
  bool Cnotzero = p->t3P.C1!=0.0;
  if(p->n_dims > 1 ) Cnotzero |= (p->t3P.C2 != 0.0);
  if(p->n_dims > 2 ) Cnotzero |= (p->t3P.C3 != 0.0);

#pragma omp parallel for
  for (BIGINT k=0;k<p->nk;++k){         // .... loop over NU targ freqs

    //  *** THIS CAN BE PRECOMPUTED EARLIER, IN SETPTS! :
    
    FLT sumCoords = (p->s[k] - p->t3P.D1)*p->t3P.C1;
    if(p->n_dims > 1)
      sumCoords += (p->t[k] - p->t3P.D2)*p->t3P.C2;
    if(p->n_dims > 2)
      sumCoords += (p->u[k] - p->t3P.D3)*p->t3P.C3;
    FLT prodPhiHat = p->phiHat[k]; // already the product of phiHat in each dimension
    
    for(int i = 0; i < nSetsThisBatch ; i++){

      CPX *fkStart = fk + (i+batchNum*p->threadBlkSize)*p->nk; //array of size nk*n_transforms

      fkStart[k] *= (CPX)(1.0/prodPhiHat);    // *** UGH - PRECOMPUTE!

      if (Cfinite && Cnotzero)
        fkStart[k] *= exp((FLT)p->fftsign * IMA*sumCoords);
    }
  }
}



// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
int finufft_exec(finufft_plan* p, CPX* cj, CPX* fk){
  // Performs spread/interp, pre/post deconvolve, and fftw_exec as appropriate
  // for 3 types. For cases of n_transf > 1, performs work in batches of size
  // min(n_transf, threadBlkSize)
  CNTime timer; 
  double t_spread = 0.0;
  double t_exec = 0.0;
  double t_deconv = 0.0;
  
  int *ier_spreads = (int *)calloc(p->threadBlkSize,sizeof(int));      

  if (p->type!=3){ // --------------------- TYPE 1,2 EXEC ------------------
  
    for(int batchNum = 0; batchNum*p->threadBlkSize < p->n_transf; batchNum++){
          
      int nSetsThisBatch = min(p->n_transf - batchNum*p->threadBlkSize, p->threadBlkSize);

      //Type 1 Step 1: Spread to Regular Grid    
      if(p->type == 1){
        timer.restart();
        spreadAllSetsInBatch(nSetsThisBatch, batchNum, p, cj, ier_spreads);
        t_spread += timer.elapsedsec();

        for(int i = 0; i < nSetsThisBatch; i++){
          if(ier_spreads[i])
            return ier_spreads[i];
        }
      }

      //Type 2 Step 1: amplify Fourier coeffs fk and copy into fw
      else if(p->type == 2){
        timer.restart();
        deconvolveInParallel(nSetsThisBatch, batchNum, p,fk);
        t_deconv += timer.elapsedsec();
      }
             
      //Type 1/2 Step 2: Call FFT   
      timer.restart();
      FFTW_EX(p->fftwPlan);
      double temp_t = timer.elapsedsec();
      t_exec += temp_t;

      //Type 1 Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output 
      if(p->type == 1){
        timer.restart();
        deconvolveInParallel(nSetsThisBatch, batchNum, p,fk);
        t_deconv += timer.elapsedsec();
      }

      //Type 2 Step 3: interpolate from regular to irregular target pts
      else if(p->type == 2){
        timer.restart();
        interpAllSetsInBatch(nSetsThisBatch, batchNum, p, cj, ier_spreads);
        t_spread += timer.elapsedsec(); 
      }
    }
    

    if(p->opts.debug){
      if(p->type == 1)
        printf("[finufft_exec] spread:\t\t\t %.3g s\n",t_spread);
      else //type 2
        printf("[finufft_exec] interp:\t\t\t %.3g s\n",t_spread);
      printf("[finufft_exec] fft :\t\t\t %.3g s\n", t_exec);
      printf("[finufft_exec] deconvolve :\t\t %.3g s\n", t_deconv);
    }
  }

  else{  // ----------------------------- TYPE 3 EXEC ---------------------

    //Allocate only nj*threadBlkSize array for scaled coordinate weights
    //this array will be recomputed for each batch/iteration 
    CPX *cpj = (CPX*)malloc(sizeof(CPX)*p->nj*p->threadBlkSize);  // c'_j rephased src
    if(!cpj){
      fprintf(stderr, "Call to malloc failed for rescaled input weights \n");
      return ERR_ALLOC; 
    }

    BIGINT n_modes[] = {p->nf1,p->nf2,p->nf3};
    
    // Preparations for the interior type 2 finufft call
    // 1) a single call to construct a finufft_plan
    // 2) a single call to finufft_setpts where scaled target freqs become the type2 x,y,z coordinates 

    finufft_plan t2Plan;
    finufft_default_opts(&t2Plan.opts);
    t2Plan.opts.debug = p->opts.debug;
    t2Plan.opts.spread_debug = p->opts.spread_debug;

    int batchSize = min(p->n_transf, p->threadBlkSize);
    timer.restart();
    int ier_t2 = finufft_makeplan(2, p->n_dims, n_modes, p->fftsign,
              batchSize, p->tol, p->threadBlkSize, &t2Plan, &t2Plan.opts);
    if(ier_t2){
      printf("inner type 2 plan creation failed\n");
      return ier_t2;  
    }
    double t_innerPlan = timer.elapsedsec();
    t2Plan.isInnerT2 = true;

    timer.restart();
    ier_t2 = finufft_setpts(&t2Plan, p->nk, p->sp, p->tp, p->up, 0, NULL, NULL, NULL);
    if(ier_t2){
      printf("inner type 2 set points failed\n");
      return ier_t2;
    }

    double t_innerSet = timer.elapsedsec();
    double t_prePhase = 0, t_innerExec = 0, t_deConvShuff = 0;  // total times
    t_spread = 0;

    // Loop over blocks of size <= threadBlkSize up to n_transf, executing t2...
    for(int batchNum = 0; batchNum*p->threadBlkSize < p->n_transf;
        batchNum++) {   // ................ batch loop
      bool lastRound = ((batchNum+1)*p->threadBlkSize > p->n_transf);
      int nSetsThisBatch = min(p->n_transf - batchNum*p->threadBlkSize, p->threadBlkSize);
     
      // prephase this batch of coordinate weights
      timer.restart();
      type3PrePhaseInParallel(nSetsThisBatch, batchNum, p, cj, cpj);
      double t = timer.elapsedsec();
      t_prePhase += t;
      
      //Spread from cpj to internal fw array (only threadBlockSize)
      timer.restart();      
      spreadAllSetsInBatch(nSetsThisBatch, batchNum, p, cpj, ier_spreads);
      t_spread += timer.elapsedsec();

      //Indicate to inner type 2 that only nSetsThisBatch transforms are left.
      //This ensures the batch loop for type 2 execute will not attempt to
      //access beyond allocated size of user supplied arrays: cj and fk.

      if(lastRound){
        plan->innerT2Plan->n_transf = nSetsThisBatch;
       }

      //carry out a finufft execution of size threadBlockSize, indexing appropriately into
      //fk (size nk*n_transforms) each iteration 
      timer.restart();

      ier_t2 = finufft_exec(plan->innerT2Plan, fk+(batchNum*plan->threadBlkSize*plan->nk), (CPX *)plan->fw);

      t_innerExec += timer.elapsedsec();
      if (ier_t2>0) exit(ier_t2);         // *** return? crashes it?
      
      //deconvolve this chunk of fk newly output from finufft_exec
      timer.restart();
      type3DeconvolveInParallel(nSetsThisBatch, batchNum, p, fk);
      t_deConvShuff += timer.elapsedsec();
    }  // ............... end batch loop

    if(p->opts.debug){
      printf("[finufft_exec] prephase:\t\t %.3g s\n",t_prePhase);
      printf("[finufft_exec] spread:\t\t\t %.3g s\n",t_spread);
      printf("[finufft_exec] total type-2 (ier=%d):\t %.3g s\n",ier_t2, t_innerExec);
      printf("[finufft_exec] deconvolve:\t\t %.3g s\n", t_deConvShuff);
    }
    
    free(cpj);
  }
  
  free(ier_spreads);
  return 0; 
}


// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
int finufft_destroy(finufft_plan* p)
  // free everything we allocated inside of finufft_plan pointed to by p
{ 
  if(p->fftwPlan)
    FFTW_DE(p->fftwPlan);  // destroy the FFTW plan
  // rest is dealloc...
  if(p->phiHat)
    free(p->phiHat);
  if(p->sortIndices)
    free(p->sortIndices);
  if(p->fw)
    FFTW_FR(p->fw);
  // for type 3, original coordinates are kept in {X,Y,Z}_orig,
  // but we must free the X,Y,Z we allocated which hold x',y',z':
  if(p->type == 3){
    free(p->X);
    if(p->Y)
      free(p->Y);
    if(p->Z)
      free(p->Z);
    free(p->sp);
    if(p->tp)
      free(p->tp);
    if(p->up)
      free(p->up);
    finufft_destroy(p->innerT2Plan);  // doesn't recurse since inner plan is t2
  }
  return 0;
}

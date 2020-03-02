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


int* n_for_fftw(finufft_plan *plan){
// helper func returns a new int array of length n_dims, extracted from
// the finufft plan, that fft_many_many_dft needs as its 2nd argument.
  int* nf;
  if(plan->n_dims == 1){ 
    nf = new int[1];
    nf[0] = (int)plan->nf1;
  }
  else if (plan->n_dims == 2){ 
    nf = new int[2];
    nf[0] = (int)plan->nf2;
    nf[1] = (int)plan->nf1; 
  }   // fftw enforced row major ordering, ie dims are backwards ordering
  else{ 
    nf = new int[3];
    nf[0] = (int)plan->nf3;
    nf[1] = (int)plan->nf2;
    nf[2] = (int)plan->nf1;
  }
  return nf;
}


// PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
int finufft_makeplan(int type, int n_dims, BIGINT *n_modes, int iflag,
                     int n_transf, FLT tol, int threadBlkSize,
                     finufft_plan *plan, nufft_opts *opts)
// Populates the fields of finufft_plan which is pointed to by "plan".
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
    finufft_default_opts(&(plan->opts));
  else                                   // or read from what's passed in
    plan->opts = *opts;    // does deep copy; changing *opts now has no effect
  // write into plan's spread options...
  int ier_set = setup_spreader_for_nufft(plan->spopts, tol, plan->opts);
  if (ier_set)
    return ier_set;

  // get stuff from args...
  plan->type = type;
  plan->n_dims = n_dims;
  plan->n_transf = n_transf;
  plan->tol = tol;
  plan->fftsign = (iflag>=0) ? 1 : -1;          // clean up flag input
  if (threadBlkSize==0)            // set default
    plan->threadBlkSize = min(MY_OMP_GET_MAX_THREADS(), MAX_USEFUL_NTHREADS);
  else
    plan->threadBlkSize = threadBlkSize;

  // set others as defaults (or unallocated for arrays)...
  plan->X = NULL;
  plan->Y = NULL;
  plan->Z = NULL;
  plan->X_orig = NULL;
  plan->Y_orig = NULL;
  plan->Z_orig = NULL;
  plan->sp = NULL;
  plan->tp = NULL;
  plan->up = NULL;
  plan->nf1 = 1;             // crucial to leave as 1 for unused dims
  plan->nf2 = 1;
  plan->nf3 = 1;
  plan->isInnerT2 = false;
  plan->ms = 1;             // crucial to leave as 1 for unused dims
  plan->mt = 1;
  plan->mu = 1;

  //  ------------------------ types 1,2: planning needed ---------------------
  if((type == 1) || (type == 2)) {

    plan->spopts.spread_direction = type;
    
    if (plan->threadBlkSize>1) {
      FFTW_INIT();         // only does anything when OMP=ON for >1 threads
      FFTW_PLAN_TH(plan->threadBlkSize);
    }

    // read in mode array dims then determine fine grid sizes, sanity check...
    plan->ms = n_modes[0];
    int ier_nf = set_nf_type12(plan->ms,plan->opts,plan->spopts,&(plan->nf1));
    if (ier_nf) return ier_nf;  // nf too big; we're outta here
    if (n_dims > 1) {
      plan->mt = n_modes[1];
      ier_nf = set_nf_type12(plan->mt, plan->opts, plan->spopts, &(plan->nf2));
      if (ier_nf) return ier_nf;
    }
    if (n_dims > 2) {
      plan->mu = n_modes[2];
      ier_nf = set_nf_type12(plan->mu, plan->opts, plan->spopts, &(plan->nf3)); 
      if (ier_nf) return ier_nf;
    }

    if (plan->opts.debug)
      printf("[finufft_plan] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) (nf1,nf2,nf3)=(%lld,%lld,%lld) blksiz=%d\n",
             n_dims, type, (long long)plan->ms,(long long)plan->mt,
             (long long) plan->mu, (long long)plan->nf1,(long long)plan->nf2,
             (long long)plan->nf3,plan->threadBlkSize);

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid dim
    CNTime timer; timer.start(); 
    BIGINT NphiHat;    // size of concatenated kernel Fourier xforms
    NphiHat = plan->nf1/2 + 1; 
    if (n_dims > 1) NphiHat += (plan->nf2/2 + 1);
    if (n_dims > 2) NphiHat += (plan->nf3/2 + 1);
    plan->phiHat = (FLT *)malloc(sizeof(FLT)*NphiHat);
    if(!plan->phiHat){
      fprintf(stderr, "finufft_plan: malloc failed for Fourier coeff array!");
      return ERR_ALLOC;
    }
    onedim_fseries_kernel(plan->nf1, plan->phiHat, plan->spopts);
    if (n_dims > 1)       // stack the 2nd dim of phiHat
      onedim_fseries_kernel(plan->nf2, plan->phiHat + (plan->nf1/2+1), plan->spopts);
    if (n_dims > 2)
      onedim_fseries_kernel(plan->nf3, plan->phiHat + (plan->nf1/2+1) + (plan->nf2/2+1), plan->spopts);
    if (plan->opts.debug) printf("[finufft_plan] kernel fser (ns=%d):\t\t %.3g s\n", plan->spopts.nspread,timer.elapsedsec());    

    BIGINT nfTotal = plan->nf1*plan->nf2*plan->nf3;   // each fine grid size
    // do no more work than necessary if the number of transforms is smaller than the the threadBlkSize
    int transfPerBatch = min(plan->threadBlkSize, plan->n_transf); 

    // ensure size of upsampled grid does not exceed MAX, otherwise give up
    if (nfTotal*transfPerBatch>MAX_NF) { 
      fprintf(stderr,"nf1*nf2*nf3*plan->threadBlkSize=%.3g exceeds MAX_NF of %.3g\n",
              (double)nfTotal*transfPerBatch,(double)MAX_NF);
      return ERR_MAXNALLOC;
    }
    plan->fw = FFTW_ALLOC_CPX(nfTotal*transfPerBatch);
    if(!plan->fw){
      fprintf(stderr, "fftw malloc failed for working upsampled array(s)\n");
      free(plan->phiHat);
      return ERR_ALLOC; 
    }
   
    timer.restart(); // plan the FFTW .........
    int *n = n_for_fftw(plan);
    // fftw_plan_many_dft args: rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
    plan->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, n, transfPerBatch, plan->fw,
                     NULL, 1, nfTotal, plan->fw, NULL, 1,
                     nfTotal, plan->fftsign, plan->opts.fftw);
    if (plan->opts.debug) printf("[finufft_plan] fftw plan (mode %d):\t\t %.3g s\n",plan->opts.fftw,timer.elapsedsec());
    delete []n;
    
  } else {  // -------------------------- type 3 -----------------------------
    if (plan->opts.debug) printf("[finufft_plan] %dd%d, blksiz=%d\n",n_dims, type,plan->threadBlkSize);
    plan->fftwPlan = NULL;
    // type 3 will call finufft_makeplan for type2, thus no need to init FFTW
  }
  return 0;
}


// SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
int finufft_setpts(finufft_plan * plan , BIGINT nj, FLT *xj, FLT *yj, FLT *zj,
                   BIGINT nk, FLT * s, FLT *t, FLT * u)
// For type 1,2: sorts the NU points.
// For type 3: allocates internal working arrays, scales/centers the NU points
// and NU target freqs, evaluates spreading kernel FT at all target freqs.
{
  CNTime timer; timer.start();
  plan->nj = nj;
  
  if ((plan->type == 1) || (plan->type == 2)){   // ------- TYPE 1,2 SETPTS ---
    int ier_check = spreadcheck(plan->nf1,plan->nf2 , plan->nf3, plan->nj, xj, yj, zj, plan->spopts);
    if(ier_check) return ier_check;

    timer.restart();
    plan->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*plan->nj);
    plan->didSort = indexSort(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, plan->nj, xj, yj, zj, plan->spopts);

    if (plan->opts.debug) printf("[finufft_setpts] sort (did_sort=%d):\t %.3g s\n", plan->didSort, timer.elapsedsec());
  

    plan->X = xj; // we just point to user's data, which must be length >=nj
    plan->Y = yj;
    plan->Z = zj;

    plan->s = NULL; // unused for t1,t2
    plan->t = NULL;
    plan->u = NULL;
  }

  else {    // ------------------------- TYPE 3 SETPTS ---------------------
    plan->nk = nk;     // # targ freq pts
    plan->spopts.spread_direction = 1;

    FLT S1, S2, S3 = 0;
    
    // pick x, s intervals & shifts, then apply these to xj, cj (twist iii)...
    CNTime timer; timer.start();
    arraywidcen(plan->nj,xj,&(plan->t3P.X1),&(plan->t3P.C1));  // get half-width, center, containing {x_j}
    arraywidcen(plan->nk,s,&S1,&(plan->t3P.D1));   // {s_k}
    set_nhg_type3(S1,plan->t3P.X1,plan->opts,plan->spopts,
                  &(plan->nf1),&(plan->t3P.h1),&(plan->t3P.gam1));          // applies twist i)

    if(plan->n_dims > 1){
      arraywidcen(plan->nj,yj,&(plan->t3P.X2),&(plan->t3P.C2));  // {y_j}
      arraywidcen(plan->nk,t,&S2,&(plan->t3P.D2));   // {t_k}
      set_nhg_type3(S2,plan->t3P.X2,plan->opts,plan->spopts,&(plan->nf2),
                    &(plan->t3P.h2),&(plan->t3P.gam2));
    }
    
    if(plan->n_dims > 2){
      arraywidcen(plan->nj,zj,&(plan->t3P.X3),&(plan->t3P.C3));  // {z_j}
      arraywidcen(plan->nk,u,&S3,&(plan->t3P.D3));   // {u_k}
      set_nhg_type3(S3,plan->t3P.X3,plan->opts,plan->spopts,
                    &(plan->nf3),&(plan->t3P.h3),&(plan->t3P.gam3));
    }

    if (plan->opts.debug){
      printf("%dd3: X1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%lld M=%lld N=%lld \n", plan->n_dims,
             plan->t3P.X1, plan->t3P.C1,S1, plan->t3P.D1, plan->t3P.gam1,(long long) plan->nf1,
             (long long)plan->nj,(long long)plan->nk);
      
      if(plan->n_dims > 1 ) printf("X2=%.3g C2=%.3g S2=%.3g D2=%.3g gam2=%g nf2=%lld \n",plan->t3P.X2, plan->t3P.C2,S2,
                                   plan->t3P.D2, plan->t3P.gam2,(long long) plan->nf2);
      if(plan->n_dims > 2 ) printf("X3=%.3g C3=%.3g S3=%.3g D3=%.3g gam3=%g nf3=%lld \n", plan->t3P.X3, plan->t3P.C3,
                                   S3, plan->t3P.D3, plan->t3P.gam3,(long long) plan->nf3);
    }

    //do no more work than necessary if the number of transforms is smaller than the the threadBlkSize
    int transfPerBatch = min(plan->threadBlkSize, plan->n_transf);
    
    if ((int64_t)plan->nf1*plan->nf2*plan->nf3*transfPerBatch>MAX_NF) {
      fprintf(stderr,"nf1*nf2*nf3*threadBlkSize=%.3g exceeds MAX_NF of %.3g\n",(double)plan->nf1*plan->nf2*plan->nf3*transfPerBatch,(double)MAX_NF);
      return ERR_MAXNALLOC;
    }


    plan->fw = FFTW_ALLOC_CPX(plan->nf1*plan->nf2*plan->nf3*transfPerBatch);  

    if(!plan->fw){
      fprintf(stderr, "Call to malloc failed for working upsampled array allocation\n");
      return ERR_MAXNALLOC; 
    }

    FLT* xpj = (FLT*)malloc(sizeof(FLT)*plan->nj);
    if(!xpj){
      fprintf(stderr, "Call to malloc failed for rescaled x coordinates\n");
      return ERR_MAXNALLOC; 
    }    
    FLT *ypj = NULL;
    FLT* zpj = NULL;

    if(plan->n_dims > 1){
      ypj = (FLT*)malloc(sizeof(FLT)*nj);
      if(!ypj){
        fprintf(stderr, "Call to malloc failed for rescaled y coordinates\n");
        return ERR_MAXNALLOC; 
      }
    }
    if(plan->n_dims > 2){
      zpj = (FLT*)malloc(sizeof(FLT)*nj);
      if(!zpj){
        fprintf(stderr, "Call to malloc failed for rescaled z coordinates\n");
        return ERR_MAXNALLOC; 
      }
    }

    timer.restart();
#pragma omp parallel for schedule(static)
    for (BIGINT j=0;j<nj;++j) {
      xpj[j] = (xj[j] - plan->t3P.C1) / plan->t3P.gam1;          // rescale x_j
      if(plan->n_dims > 1)
        ypj[j] = (yj[j]- plan->t3P.C2) / plan->t3P.gam2;          // rescale y_j
      if(plan->n_dims > 2)
        zpj[j] = (zj[j] - plan->t3P.C3) / plan->t3P.gam3;          // rescale z_j
    }
    if (plan->opts.debug) printf("[finufft_setpts] t3 coord scale:\t\t %.3g s\n",timer.elapsedsec());

    int ier_check = spreadcheck(plan->nf1,plan->nf2 , plan->nf3, plan->nj, xpj, ypj, zpj, plan->spopts);
    if(ier_check) return ier_check;

    timer.restart();
    plan->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*plan->nj);
    plan->didSort = indexSort(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, plan->nj, xpj, ypj, zpj, plan->spopts);

    if (plan->opts.debug) printf("[finufft_setpts] sort (did_sort=%d):\t %.3g s\n", plan->didSort, timer.elapsedsec());
    
    plan->X = xpj;
    plan->X_orig = xj;
    plan->Y = ypj;
    plan->Y_orig = yj;
    plan->Z = zpj;
    plan->Z_orig = zj;
    
    
    FLT *sp = (FLT*)malloc(sizeof(FLT)*plan->nk);     // rescaled targs s'_k
    if(!sp){
      fprintf(stderr, "Call to malloc failed for rescaled s target freqs\n");
      return ERR_MAXNALLOC; 
    }
    
    FLT *tp = NULL;
    if(plan->n_dims > 1 ){
      tp = (FLT*)malloc(sizeof(FLT)*plan->nk);     // t'_k
      if(!tp){
        fprintf(stderr, "Call to malloc failed for rescaled t target freqs\n");
        return ERR_MAXNALLOC; 
      }
    }

    FLT *up = NULL;
    if(plan->n_dims > 2 ){
      up = (FLT*)malloc(sizeof(FLT)*plan->nk);     // u'_k
      if(!up){
        fprintf(stderr, "Call to malloc failed for rescaled u target freqs\n");
        return ERR_MAXNALLOC; 
      }
    }

    //Originally performed right before Step 2 recursive call to finufftxd2
    timer.restart();
#pragma omp parallel for schedule(static) //static appropriate for load balance across loop iterations 
    for (BIGINT k=0;k<plan->nk;++k) {
      sp[k] = plan->t3P.h1*plan->t3P.gam1*(s[k]-plan->t3P.D1);      // so that |s'_k| < pi/R
      if(plan->n_dims > 1 )
        tp[k] = plan->t3P.h2*plan->t3P.gam2*(t[k]-plan->t3P.D2);      // so that |t'_k| < pi/R
      if(plan->n_dims > 2)
        up[k] = plan->t3P.h3*plan->t3P.gam3*(u[k]-plan->t3P.D3);      // so that |u'_k| < pi/R
    }
    if(plan->opts.debug) printf("[finufft_setpts] rescaling target-freqs: \t %.3g s\n", timer.elapsedsec());

    // Originally Step 3a: compute Fourier transform of scaled kernel at targets
    timer.restart();
    plan->phiHat = (FLT *)malloc(sizeof(FLT)*plan->nk*plan->n_dims);
    if(!plan->phiHat){
      fprintf(stderr, "Call to Malloc failed for Fourier coeff array allocation\n");
      return ERR_MAXNALLOC;
    }

    //phiHat spreading kernel fourier weights for non uniform target freqs := referred to as fkker in older code
    onedim_nuft_kernel(plan->nk, sp, plan->phiHat, plan->spopts);         
    if(plan->n_dims > 1)
      onedim_nuft_kernel(plan->nk, tp, plan->phiHat + plan->nk, plan->spopts);           
    if(plan->n_dims > 2)
      onedim_nuft_kernel(plan->nk, up, plan->phiHat + 2*plan->nk, plan->spopts);
    if (plan->opts.debug) printf("[finufft_setpts] kernel FT (ns=%d):\t\t %.3g s\n", plan->spopts.nspread,timer.elapsedsec());

    //precompute product of phiHat for 2 and 3 dimensions 
    if(plan->n_dims > 1){
#pragma omp parallel for schedule(static)              
      for(BIGINT k=0; k < plan->nk; k++)
        plan->phiHat[k]*=(plan->phiHat+plan->nk)[k];
    }

    if(plan->n_dims > 2){
#pragma omp parallel for schedule(static)              
      for(BIGINT k=0; k < plan->nk; k++)
        plan->phiHat[k]*=(plan->phiHat+plan->nk + plan->nk)[k];
    }
    
    plan->s = s;
    plan->sp = sp;
    
    //NULL if 1 dim
    plan->t = t;
    plan->tp = tp;
    
    //NULL if 2 dim
    plan->u = u;
    plan->up = up;
    
  }
  
  return 0;
}


void spreadAllSetsInBatch(int nSetsThisBatch, int blkNum, finufft_plan *plan, CPX * c, int *ier_spreads){
  // Type 1 + Type 3: Spreads coordinate weights from c into internal workspace
  // fw for sending into fftw

  // nSetsThisBatch is the threadBlockSize, except for the last round if
  // threadBlockSize does not divide evenly into n_transf, prevents c overrun.

  BIGINT fwRowSize = plan->nf1*plan->nf2*plan->nf3; 
  int blkJump = blkNum*plan->threadBlkSize; 

  // default sequential maximum multithreaded: execute
  //the for-loop down below on THIS thread (spawn no others)
  //and leave all the multithreading for inside of the spreadSorted call
  int n_outerThreads = 0;
  if(plan->opts.spread_scheme==1) // simultaneous singlethreaded/nested multi
    n_outerThreads = nSetsThisBatch; //spawn as many threads as sets, if fewer sets than available threads
                                     //the extra threads used for work inside of spreadSorted 
  
  MY_OMP_SET_NESTED(1); 
#pragma omp parallel for num_threads(n_outerThreads)
  for(int i = 0; i < nSetsThisBatch; i++){ 

    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = plan->fw + fwRowSize*i;

    //for type 3, c is "cpj", scaled weights, and spreading is done in batches of size threadBlockSize
    CPX *cStart;
    if(plan->type == 3)
      cStart = c + plan->nj*i;

    //for type1+2, c is the client's array and of size nj*n_transforms
    else
      cStart = c + plan->nj*(i + blkJump); 
    
    int ier = spreadSorted(plan->sortIndices,
                           plan->nf1, plan->nf2, plan->nf3, (FLT*)fwStart,
                           plan->nj, plan->X, plan->Y, plan->Z, (FLT *)cStart,
                           plan->spopts, plan->didSort) ;
    if(ier)
      ier_spreads[i] = ier;
  }
  MY_OMP_SET_NESTED(0);
}

/*Type 2: Interpolates from weights at uniform points in fw to non uniform points in c*/
void interpAllSetsInBatch(int nSetsThisBatch, int batchNum, finufft_plan *plan, CPX * c, int *ier_interps){

  BIGINT fwRowSize =  plan->nf1*plan->nf2*plan->nf3;
  int blkJump = batchNum*plan->threadBlkSize; 

  //default sequential maximum multithreaded: execute
  //the for-loop down below on THIS thread (spawn no others)
  //and leave all the multithreading for inside of the interpSorted call
  int n_outerThreads = 0;
  if(plan->opts.spread_scheme){
    //spread_scheme == 1 -> simultaneous singlethreaded/nested multi
    n_outerThreads = nSetsThisBatch; //spawn as many threads as sets, if fewer sets than available threads
                                     //the extra threads used for work inside of spreadSorted 
  }
  
  MY_OMP_SET_NESTED(1);
#pragma omp parallel for num_threads(n_outerThreads)
  for(int i = 0; i < nSetsThisBatch; i++){ 
        
    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = plan->fw + fwRowSize*i; //fw gets reread on each iteration of j

    CPX * cStart;

    //If this is a type 2 being executed inside of a type 3, c is an internal array of size nj*threadBlockSize
    if(plan->isInnerT2)
      cStart = c + plan->nj*i;

    //for type 1+ regular 2, c is the result array, size nj*n_transforms
    else
      cStart = c + plan->nj*(i + blkJump);

    int ier = interpSorted(plan->sortIndices,
                           plan->nf1, plan->nf2, plan->nf3, (FLT*)fwStart,
                           plan->nj, plan->X, plan->Y, plan->Z, (FLT *)cStart,
                           plan->spopts, plan->didSort) ;

    if(ier)
      ier_interps[i] = ier;
  }
  MY_OMP_SET_NESTED(0);
}

void deconvolveInParallel(int nSetsThisBatch, int batchNum, finufft_plan *plan, CPX *fk)
/* Type 1: deconvolves (amplifies) from interior fw array into user-supplied fk.
   Type 2: deconvolves from user-supplied fk into interior fw array.
   This is mostly a parallel loop calling deconvolveshuffle?d in the needed dim.
*/
{
  // phiHat = kernel FT arrays (stacked version of fwker in 2017 code)
  FLT* phiHat1 = plan->phiHat, *phiHat2=NULL, *phiHat3=NULL;
  if(plan->n_dims > 1)
    phiHat2 = plan->phiHat + plan->nf1/2 + 1;
  if(plan->n_dims > 2)
    phiHat3 = plan->phiHat+(plan->nf1/2+1)+(plan->nf2/2+1);

  BIGINT fkRowSize = plan->ms*plan->mt*plan->mu;
  BIGINT fwRowSize = plan->nf1*plan->nf2*plan->nf3;
  int blockJump = batchNum*plan->threadBlkSize;

#pragma omp parallel for
  for(int i = 0; i < nSetsThisBatch; i++) {
    CPX *fkStart;

    //If this is a type 2 being executed inside of a type 3, fk is internal array of size nj*threadBlockSize
    if(plan->isInnerT2)
      fkStart = fk + i*fkRowSize;

    //otherwise it is a user supplied array of size ms*mt*mu*n_transforms
    else
      fkStart = fk + (i+blockJump)*fkRowSize;
    
    FFTW_CPX *fwStart = plan->fw + fwRowSize*i;

    //deconvolveshuffle?d are not multithreaded inside, so called in parallel here
    //prefactors hardcoded to 1...
    if(plan->n_dims == 1)
      deconvolveshuffle1d(plan->spopts.spread_direction, 1.0, phiHat1,
                          plan->ms, (FLT *)fkStart,
                          plan->nf1, fwStart, plan->opts.modeord);
    else if (plan->n_dims == 2)
      deconvolveshuffle2d(plan->spopts.spread_direction,1.0, phiHat1, phiHat2,
                          plan->ms, plan->mt, (FLT *)fkStart,
                          plan->nf1, plan->nf2, fwStart, plan->opts.modeord);
    else
      deconvolveshuffle3d(plan->spopts.spread_direction, 1.0, phiHat1, phiHat2,
                          phiHat3, plan->ms, plan->mt, plan->mu,
                          (FLT *)fkStart, plan->nf1, plan->nf2, plan->nf3,
                          fwStart, plan->opts.modeord);
  }
}

/*Type 3 multithreaded prephase all nj scaled weights for all of the sets of weights in this batch*/
//occurs inplace of internal finufft array cpj (sized nj*threadBlkSize)
void type3PrePhaseInParallel(int nSetsThisBatch, int batchNum, finufft_plan * plan, CPX *cj, CPX *cpj){

  bool notZero = plan->t3P.D1 != 0.0;
  if(plan->n_dims > 1) notZero |=  (plan->t3P.D2 != 0.0);
  if(plan->n_dims > 2) notZero |=  (plan->t3P.D3 != 0.0);
    
  // note that schedule(dynamic) actually slows this down (was in v<=1.1.2):
#pragma omp parallel for
  for (BIGINT i=0; i<plan->nj;i++){

    FLT sumCoords = plan->t3P.D1*plan->X_orig[i];
    if(plan->n_dims > 1)
      sumCoords += plan->t3P.D2*plan->Y_orig[i];
    if(plan->n_dims > 2)
      sumCoords += plan->t3P.D3*plan->Z_orig[i];
          
    CPX multiplier = exp((FLT)plan->fftsign * IMA*sumCoords); // rephase
          
    for(int k = 0; k < nSetsThisBatch; k++){ // *** jumps around in RAM - speed?
      int cpjIndex = k*plan->nj + i;
      int cjIndex = batchNum*plan->threadBlkSize*plan->nj + cpjIndex;

      cpj[cpjIndex] = cj[cjIndex]; 
      if(notZero)
	cpj[cpjIndex] *= multiplier;
    }
  }
}


/*Type 3 multithreaded In place deconvolve of user supplied result array fk (size nk*n_transf)*/
void type3DeconvolveInParallel(int nSetsThisBatch, int batchNum, finufft_plan *plan, CPX *fk){

  bool finite  = isfinite(plan->t3P.C1);
  if(plan->n_dims > 1 ) finite &=  isfinite(plan->t3P.C2);
  if(plan->n_dims > 2 ) finite &=  isfinite(plan->t3P.C3);
  bool notzero = plan->t3P.C1!=0.0;
  if(plan->n_dims > 1 ) notzero |=  (plan->t3P.C2 != 0.0);
  if(plan->n_dims > 2 ) notzero |=  (plan->t3P.C3 != 0.0);

#pragma omp parallel for
  for (BIGINT k=0;k<plan->nk;++k){     
        
    FLT sumCoords = (plan->s[k] - plan->t3P.D1)*plan->t3P.C1;
    FLT prodPhiHat = plan->phiHat[k]; //already the product of phiHat in each dimension

    if(plan->n_dims > 1 ){
      sumCoords += (plan->t[k] - plan->t3P.D2)*plan->t3P.C2 ;
    }
        
    if(plan->n_dims > 2){
      sumCoords += (plan->u[k] - plan->t3P.D3)*plan->t3P.C3;
    }

    for(int i = 0; i < nSetsThisBatch ; i++){

      CPX *fkStart = fk + (i+batchNum*plan->threadBlkSize)*plan->nk; //array of size nk*n_transforms

      fkStart[k] *= (CPX)(1.0/prodPhiHat);

      if(finite && notzero)
        fkStart[k] *= exp((FLT)plan->fftsign * IMA*sumCoords);
    }
  }
}



// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
int finufft_exec(finufft_plan * plan , CPX * cj, CPX * fk){
  // Performs spread/interp, pre/post deconvolve, and fftw_exec as appropriate
  // for 3 types. For cases of n_transf > 1, performs work in batches of size
  // min(n_transf, threadBlkSize)
  CNTime timer; 
  double t_spread = 0.0;
  double t_exec = 0.0;
  double t_deconv = 0.0;
  
  int *ier_spreads = (int *)calloc(plan->threadBlkSize,sizeof(int));      

  if (plan->type!=3){ // --------------------- TYPE 1,2 EXEC ------------------
  
    for(int batchNum = 0; batchNum*plan->threadBlkSize < plan->n_transf; batchNum++){
          
      int nSetsThisBatch = min(plan->n_transf - batchNum*plan->threadBlkSize, plan->threadBlkSize);

      //Type 1 Step 1: Spread to Regular Grid    
      if(plan->type == 1){
        timer.restart();
        spreadAllSetsInBatch(nSetsThisBatch, batchNum, plan, cj, ier_spreads);
        t_spread += timer.elapsedsec();

        for(int i = 0; i < nSetsThisBatch; i++){
          if(ier_spreads[i])
            return ier_spreads[i];
        }
      }

      //Type 2 Step 1: amplify Fourier coeffs fk and copy into fw
      else if(plan->type == 2){
        timer.restart();
        deconvolveInParallel(nSetsThisBatch, batchNum, plan,fk);
        t_deconv += timer.elapsedsec();
      }
             
      //Type 1/2 Step 2: Call FFT   
      timer.restart();
      FFTW_EX(plan->fftwPlan);
      double temp_t = timer.elapsedsec();
      t_exec += temp_t;

      //Type 1 Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output 
      if(plan->type == 1){
        timer.restart();
        deconvolveInParallel(nSetsThisBatch, batchNum, plan,fk);
        t_deconv += timer.elapsedsec();
      }

      //Type 2 Step 3: interpolate from regular to irregular target pts
      else if(plan->type == 2){
        timer.restart();
        interpAllSetsInBatch(nSetsThisBatch, batchNum, plan, cj, ier_spreads);
        t_spread += timer.elapsedsec(); 
      }
    }
    

    if(plan->opts.debug){
      if(plan->type == 1)
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
    CPX *cpj = (CPX*)malloc(sizeof(CPX)*plan->nj*plan->threadBlkSize);  // c'_j rephased src
    if(!cpj){
      fprintf(stderr, "Call to malloc failed for rescaled input weights \n");
      return ERR_ALLOC; 
    }

    BIGINT n_modes[3];
    n_modes[0] = plan->nf1;
    n_modes[1] = plan->nf2;
    n_modes[2] = plan->nf3;

    t_spread = 0;
    double t_innerExec= 0;
    double t_deConvShuff = 0;
    int ier_t2;
    
    // Preparations for the interior type 2 finufft call
    // 1) a single call to construct a finufft_plan
    // 2) a single call to finufft_setpts where scaled target freqs become the type2 x,y,z coordinates 

    finufft_plan t2Plan;
    finufft_default_opts(&t2Plan.opts);
    t2Plan.opts.debug = plan->opts.debug;
    t2Plan.opts.spread_debug = plan->opts.spread_debug;

    int batchSize = min(plan->n_transf, plan->threadBlkSize);
    timer.restart();
    ier_t2 = finufft_makeplan(2, plan->n_dims, n_modes, plan->fftsign, batchSize, plan->tol,
                              plan->threadBlkSize, &t2Plan, &t2Plan.opts);
    if(ier_t2){
      printf("inner type 2 plan creation failed\n");
      return ier_t2;  
    }
    double t_innerPlan = timer.elapsedsec();
    t2Plan.isInnerT2 = true;

    timer.restart();
    ier_t2 = finufft_setpts(&t2Plan, plan->nk, plan->sp, plan->tp, plan->up, 0, NULL, NULL, NULL);
    if(ier_t2){
      printf("inner type 2 set points failed\n");
      return ier_t2;
    }

    double t_innerSet = timer.elapsedsec();
    double t_prePhase = 0; 

    //Loop over blocks of size plan->threadBlkSize until n_transforms have been computed
    for(int batchNum = 0; batchNum*plan->threadBlkSize < plan->n_transf; batchNum++){

      bool lastRound = false;
      
      int nSetsThisBatch = min(plan->n_transf - batchNum*plan->threadBlkSize, plan->threadBlkSize);
     
      //Is this the last iteration ? 
      if((batchNum+1)*plan->threadBlkSize > plan->n_transf)
        lastRound = true;

      //prephase this block of coordinate weights
      timer.restart();
      type3PrePhaseInParallel(nSetsThisBatch, batchNum, plan, cj, cpj);
      double t = timer.elapsedsec();
      t_prePhase += t;
      
      //Spread from cpj to internal fw array (only threadBlockSize)
      timer.restart();      
      spreadAllSetsInBatch(nSetsThisBatch, batchNum, plan, cpj, ier_spreads);
      t_spread += timer.elapsedsec();

      //Indicate to inner type 2 that only nSetsThisBatch transforms are left.
      //This ensures the batch loop for type 2 execute will not attempt to
      //access beyond allocated size of user supplied arrays: cj and fk.
      if(lastRound){
        t2Plan.n_transf = nSetsThisBatch;
       }

      //carry out a finufft execution of size threadBlockSize, indexing appropriately into
      //fk (size nk*n_transforms) each iteration 
      timer.restart();
      ier_t2 = finufft_exec(&t2Plan, fk+(batchNum*plan->threadBlkSize*plan->nk), (CPX *)plan->fw);
      t_innerExec += timer.elapsedsec();
      
      if (ier_t2>0) exit(ier_t2);
      
      //deconvolve this chunk of fk newly output from finufft_exec
      timer.restart();
      type3DeconvolveInParallel(nSetsThisBatch, batchNum, plan, fk);
      t_deConvShuff += timer.elapsedsec();

    }

    if(plan->opts.debug){
      printf("[finufft_exec] prephase:\t\t %.3g s\n",t_prePhase);
      printf("[finufft_exec] spread:\t\t\t %.3g s\n",t_spread);
      printf("[finufft_exec] total type-2 (ier=%d):\t %.3g s\n",ier_t2, t_innerPlan + t_innerSet + t_innerExec);
      printf("[finufft_exec] deconvolve:\t\t %.3g s\n", t_deConvShuff);
    }
    
    finufft_destroy(&t2Plan);
    free(cpj);
  }
  
  free(ier_spreads);
  return 0; 
}


// ..........................................................................
int finufft_destroy(finufft_plan* plan)
  // free everything we allocated inside of finufft_plan
{ 
  if(plan->fftwPlan)
    FFTW_DE(plan->fftwPlan);  // destroy the FFTW plan
  // rest is dealloc...
  if(plan->phiHat)
    free(plan->phiHat);
  if(plan->sortIndices)
    free(plan->sortIndices);
  if(plan->fw)
    FFTW_FR(plan->fw);
  // for type 3, original coordinates are kept in {X,Y,Z}_orig,
  // but we must free the X,Y,Z we allocated which hold x',y',z':
  if(plan->type == 3){
    free(plan->X);
    if(plan->Y)
      free(plan->Y);
    if(plan->Z)
      free(plan->Z);
    free(plan->sp);
    if(plan->tp)
      free(plan->tp);
    if(plan->up)
      free(plan->up);
  }   
  return 0;
}

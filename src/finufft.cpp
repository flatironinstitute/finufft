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
     maxBatchSize (set to max_num_omp_threads) times the RAM is needed, so this is
     good only for small problems.

Design notes for guru interface implementation:

* Since finufft_plan is C-compatible, we need to use malloc/free for its
  allocatable arrays, keeping it quite low-level. We can't use std::vector
  since the only survive in the scope of each function.

*/


int* gridsize_for_fftw(finufft_plan* p){
// helper func returns a new int array of length dim, extracted from
// the finufft plan, that fft_many_many_dft needs as its 2nd argument.
  int* nf;
  if(p->dim == 1){ 
    nf = new int[1];
    nf[0] = (int)p->nf1;
  }
  else if (p->dim == 2){ 
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
int finufft_makeplan(int type, int dim, BIGINT* n_modes, int iflag,
                     int n_transf, FLT tol, int maxBatchSize,
                     finufft_plan* p, nufft_opts* opts)
// Populates the fields of finufft_plan which is pointed to by "p".
// opts is ptr to a nufft_opts to set options, or NULL to use defaults.
// For types 1,2 allocates memory for internal working arrays,
// evaluates spreading kernel coefficients, and instantiates the fftw_plan
{  
  cout << scientific << setprecision(15);  // for debug outputs

  if((type!=1)&&(type!=2)&&(type!=3)) {
    fprintf(stderr, "Invalid type (%d), type should be 1, 2 or 3.",type);
    return ERR_TYPE_NOTVALID;
  }
  if((dim!=1)&&(dim!=2)&&(dim!=3)) {
    fprintf(stderr, "Invalid dim (%d), should be 1, 2 or 3.",dim);
    return ERR_DIM_NOTVALID;
  }
  if (n_transf<1) {
    fprintf(stderr,"n_transf (%d) should be at least 1.\n",n_transf);
    return ERR_NTRANSF_NOTVALID;
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
  p->dim = dim;
  p->n_transf = n_transf;
  p->tol = tol;
  p->fftSign = (iflag>=0) ? 1 : -1;          // clean up flag input
  if (maxBatchSize==0)                            // use default
    p->batchSize = min(MY_OMP_GET_MAX_THREADS(), MAX_USEFUL_NTHREADS);
  else
    p->batchSize = maxBatchSize;
  p->batchSize = min(p->batchSize,n_transf);     // don't overrun n_transf

  // set others as defaults (or unallocated for arrays)...
  p->X = NULL; p->Y = NULL; p->Z = NULL;
  p->phiHat2 = NULL; p->phiHat3 = NULL; 
  p->nf1 = 1; p->nf2 = 1; p->nf3 = 1;  // crucial to leave as 1 for unused dims
  p->ms = 1; p->mt = 1; p->mu = 1;     // crucial to leave as 1 for unused dims

  //  ------------------------ types 1,2: planning needed ---------------------
  if((type == 1) || (type == 2)) {

    int nth = MY_OMP_GET_MAX_THREADS();    // tell FFTW what it has access to
    FFTW_INIT();           // only does anything when OMP=ON for >1 threads
    FFTW_PLAN_TH(nth);     // "
    p->spopts.spread_direction = type;
    
    // read user mode array dims then determine fine grid sizes, sanity check...
    p->ms = n_modes[0];
    int ier_nf = set_nf_type12(p->ms,p->opts,p->spopts,&(p->nf1));
    if (ier_nf) return ier_nf;    // nf too big; we're outta here
    p->phiHat1 = (FLT*)malloc(sizeof(FLT)*(p->nf1/2 + 1));
    if (dim > 1) {
      p->mt = n_modes[1];
      ier_nf = set_nf_type12(p->mt, p->opts, p->spopts, &(p->nf2));
      if (ier_nf) return ier_nf;
      p->phiHat2 = (FLT*)malloc(sizeof(FLT)*(p->nf2/2 + 1));
    }
    if (dim > 2) {
      p->mu = n_modes[2];
      ier_nf = set_nf_type12(p->mu, p->opts, p->spopts, &(p->nf3)); 
      if (ier_nf) return ier_nf;
      p->phiHat3 = (FLT*)malloc(sizeof(FLT)*(p->nf3/2 + 1));
    }

    if (p->opts.debug)
      printf("[finufft_plan] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) (nf1,nf2,nf3)=(%lld,%lld,%lld) batchSize=%d\n",
             dim, type, (long long)p->ms,(long long)p->mt,
             (long long) p->mu, (long long)p->nf1,(long long)p->nf2,
             (long long)p->nf3, p->batchSize);

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid dim
    CNTime timer; timer.start();
    onedim_fseries_kernel(p->nf1, p->phiHat1, p->spopts);
    if (dim>1) onedim_fseries_kernel(p->nf2, p->phiHat2, p->spopts);
    if (dim>2) onedim_fseries_kernel(p->nf3, p->phiHat3, p->spopts);
    if (p->opts.debug) printf("[finufft_plan] kernel fser (ns=%d):\t\t %.3g s\n", p->spopts.nspread, timer.elapsedsec());

    p->nf = p->nf1*p->nf2*p->nf3;      // fine grid total number of points
    p->fwBatch = FFTW_ALLOC_CPX(p->nf * p->batchSize);
    if (p->opts.debug) printf("[finufft_plan] fwBatch alloc:\t\t\t %.3g s\n",timer.elapsedsec());
    if(!p->fwBatch) {
      fprintf(stderr, "fftw malloc failed for batch of working fine grids\n");
      free(p->phiHat1); free(p->phiHat2); free(p->phiHat3);
      return ERR_ALLOC; 
    }
   
    timer.restart();            // plan the FFTW
    int *ns = gridsize_for_fftw(p);
    // fftw_plan_many_dft args: rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
    p->fftwPlan = FFTW_PLAN_MANY_DFT(dim, ns, p->batchSize, p->fwBatch,
         NULL, 1, p->nf, p->fwBatch, NULL, 1, p->nf, p->fftSign, p->opts.fftw);
    if (p->opts.debug) printf("[finufft_plan] fftw plan (mode %d):\t\t %.3g s\n", p->opts.fftw, timer.elapsedsec());
    delete []ns;
    
  } else {  // -------------------------- type 3 (no planning) ----------------

    if (p->opts.debug) printf("[finufft_plan] %dd%d\n",dim,type);
    printf("*** guru t3 gutted for now :(");
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
    
    int ier_check = spreadcheck(p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (ier_check) return ier_check;
    if (p->opts.debug>1) printf("[finufft_setpts] spreadcheck (%d):\t %.3g s\n", p->spopts.chkbnds, timer.elapsedsec());
    
    timer.restart();
    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*p->nj);
    p->didSort = indexSort(p->sortIndices, p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->opts.debug) printf("[finufft_setpts] sort (did_sort=%d):\t %.3g s\n", p->didSort, timer.elapsedsec());
    
    p->X = xj;  // keep pointers to user's data, which must be length >=nj
    p->Y = yj;
    p->Z = zj;

  } else {    // ------------------------- TYPE 3 SETPTS ---------------------
    
  }
  
  return 0;
}
// ............ end setpts ..................................................





// BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
int spreadBatch(int batchSize, finufft_plan* p, CPX* cBatch)
/* Spreads a batch of batchSize strength vectors cBatch to the batch of fine
   working grids p->fw, using the same set of NU points p->X,Y,Z for each
   vector in the batch.
   Note that cBatch is already assumed to have the correct offset, ie it
   reads from the start of cBatch.
*/
{


  BIGINT Nfw = p->nf1*p->nf2*p->nf3;     // size of each fw 
  int blkJump = blkNum*p->batchSize; 

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
                           p->spopts, p->didSort);
    if(ier)
      ier_spreads[i] = ier;
  }
  MY_OMP_SET_NESTED(0);
}

void interpAllSetsInBatch(int nSetsThisBatch, int blk, finufft_plan* p, CPX* c, int* ier_interps)
// Type 2: Interpolates from weights at uniform points in fw to non uniform points in c
{
  BIGINT fwRowSize =  p->nf1*p->nf2*p->nf3;
  int blkJump = blk*p->batchSize; 

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

void deconvolveInParallel(int nSetsThisBatch, int blk, finufft_plan* p, CPX* fk)
/* Type 1: deconvolves (amplifies) from interior fw array into user-supplied fk.
   Type 2: deconvolves from user-supplied fk into interior fw array.
   This is mostly a parallel loop calling deconvolveshuffle?d in the needed dim.
*/
{
  // phiHat = kernel FT arrays (stacked version of fwker in 2017 code)
  FLT* phiHat1 = p->phiHat, *phiHat2=NULL, *phiHat3=NULL;
  if(p->dim > 1)
    phiHat2 = p->phiHat + p->nf1/2 + 1;
  if(p->dim > 2)
    phiHat3 = p->phiHat+(p->nf1/2+1)+(p->nf2/2+1);

  BIGINT fkRowSize = p->ms*p->mt*p->mu;
  BIGINT fwRowSize = p->nf1*p->nf2*p->nf3;
  int blockJump = blk*p->batchSize;

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
    if(p->dim == 1)
      deconvolveshuffle1d(p->spopts.spread_direction, 1.0, phiHat1,
                          p->ms, (FLT *)fkStart,
                          p->nf1, fwStart, p->opts.modeord);
    else if (p->dim == 2)
      deconvolveshuffle2d(p->spopts.spread_direction, 1.0, phiHat1, phiHat2,
                          p->ms, p->mt, (FLT *)fkStart,
                          p->nf1, p->nf2, fwStart, p->opts.modeord);
    else
      deconvolveshuffle3d(p->spopts.spread_direction, 1.0, phiHat1, phiHat2,
                          phiHat3, p->ms, p->mt, p->mu,
                          (FLT *)fkStart, p->nf1, p->nf2, p->nf3,
                          fwStart, p->opts.modeord);
  }
}



// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
int finufft_exec(finufft_plan* p, CPX* cj, CPX* fk){
  // Uses a new set of weights (cj) and performs NUFFTs with existing NU pts
  // and plan.
  // Performs spread/interp, pre/post deconvolve, and fftw_exec as appropriate
  // for 3 types. For cases of n_transf>1, performs work in blocks of size
  // up to batchSize.
  CNTime timer;
  double t_spread = 0.0, t_fft = 0.0, t_deconv = 0.0;  // accumulate timings
  
  int *ier_spreads = (int *)calloc(p->batchSize,sizeof(int)); // errors within blk

  if (p->type!=3){ // --------------------- TYPE 1,2 EXEC ------------------
  
    for(int blk=0; blk*p->batchSize < p->n_transf; blk++){  // loop over blocks...

      // current block either batchSize, or truncated if last one
      int thisBatchSize = min(p->n_transf - blk*p->batchSize, p->batchSize);

      // Type 1 Step 1: Spread to Regular Grid    
      if(p->type == 1) { // step 1: spread from NU pts p->X, weights cj,
        timer.restart();
        spreadAllSetsInBatch(thisBatchSize, blk, p, cj, ier_spreads);
        t_spread += timer.elapsedsec();

        for(int i = 0; i < thisBatchSize; i++){
          if(ier_spreads[i])
            return ier_spreads[i];
        }
      }

      //Type 2 Step 1: amplify Fourier coeffs fk and copy into fw
      else if(p->type == 2){
        timer.restart();
        deconvolveInParallel(thisBatchSize, blk, p,fk);
        t_deconv += timer.elapsedsec();
      }
             
      //Type 1 or 2. Step 2: call the preplanned FFT
      timer.restart();
      FFTW_EX(p->fftwPlan);                 // *** what if thisBatchSize<batchSize?
      t_fft += timer.elapsedsec();

      //Type 1 Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output 
      if(p->type == 1){
        timer.restart();
        deconvolveInParallel(thisBatchSize, blk, p,fk);
        t_deconv += timer.elapsedsec();
      }

      //Type 2 Step 3: interpolate from regular to irregular target pts
      else if(p->type == 2){
        timer.restart();
        interpAllSetsInBatch(thisBatchSize, blk, p, cj, ier_spreads);
        t_spread += timer.elapsedsec(); 
      }
    }
    

    if(p->opts.debug){
      if(p->type == 1)
        printf("[finufft_exec] tot spread:\t\t\t %.3g s\n",t_spread);
      else   // type 2
        printf("[finufft_exec] tot interp:\t\t\t %.3g s\n",t_spread);
      
      printf("[finufft_exec] tot fft:\t\t\t %.3g s\n", t_fft);
      printf("[finufft_exec] tot deconvolve:\t\t %.3g s\n", t_deconv);
    }
  }

  else{  // ----------------------------- TYPE 3 EXEC ---------------------

  }
  
  free(ier_spreads);
  return 0; 
}


// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
int finufft_destroy(finufft_plan* p)
  // free everything we allocated inside of finufft_plan pointed to by p
{ 
  FFTW_DE(p->fftwPlan);  // destroy any FFTW plan (t1,2 only)
  FFTW_FR(p->fwBatch);   // free the FFTW working array
  free(p->phiHat1);
  free(p->phiHat2);
  free(p->phiHat3);
  free(p->sortIndices);

  // for type 3, original coordinates are kept in {X,Y,Z}_orig,
  // but we must free the X,Y,Z we allocated which hold x',y',z':
  if(p->type == 3){
  }
  return 0;
}

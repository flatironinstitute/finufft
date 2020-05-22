#include <finufft.h>
#include <utils.h>
#include <iostream>
#include <common.h>
#include <iomanip>

/* The main guru functions for FINUFFT.

   Original guru interface written by Andrea Malleo, summer 2019, mentored
   by Alex Barnett. Many rewrites in early 2020 by Alex Barnett, Libin Lu.

   As of v1.2 these replace the old hand-coded separate 9 finufft?d?() functions
   and the two finufft2d?many() functions.
   The (now 18) simple C++ interfaces are in simpleinterfaces.cpp

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
     maxBatchSize (set to max_num_omp_threads) times the RAM is needed, so
     this is good only for small problems.


Design notes for guru interface implementation:

* Since finufft_plan is C-compatible, we need to use malloc/free for its
  allocatable arrays, keeping it quite low-level. We can't use std::vector
  since that would  only survive in the scope of each function.

*/


int* gridsize_for_fftw(finufft_plan* p){
// helper func returns a new int array of length dim, extracted from
// the finufft plan, that fft_plan_many_dft needs as its 2nd argument.
  int* nf;
  if(p->dim == 1){ 
    nf = new int[1];
    nf[0] = (int)p->nf1;
  }
  else if (p->dim == 2){ 
    nf = new int[2];
    nf[0] = (int)p->nf2;
    nf[1] = (int)p->nf1; 
  }   // fftw enforced row major ordering, ie dims are backwards ordered
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
                     int ntrans, FLT tol, finufft_plan* p, nufft_opts* opts)
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
  if (ntrans<1) {
    fprintf(stderr,"ntrans (%d) should be at least 1.\n",ntrans);
    return ERR_NTRANSF_NOTVALID;
  }

  if (opts==NULL)                        // use default opts
    finufft_default_opts(&(p->opts));
  else                                   // or read from what's passed in
    p->opts = *opts;    // does deep copy; changing *opts now has no effect
  // write into plan's spread options...
  int ier = setup_spreader_for_nufft(p->spopts, tol, p->opts);
  if (ier)
    return ier;

  // get stuff from args...
  p->type = type;
  p->dim = dim;
  p->ntrans = ntrans;
  p->tol = tol;
  p->fftSign = (iflag>=0) ? 1 : -1;          // clean up flag input
  int nth = min(MY_OMP_GET_MAX_THREADS(), MAX_USEFUL_NTHREADS);  // limit it
  if (p->opts.maxbatchsize==0) {             // logic to auto-set best batchsize
    int nbatches = (int)ceil((double)ntrans/nth);       // min # batches needed
    p->batchSize = (int)ceil((double)ntrans/nbatches);  // cut # thr in each b
  } else
    p->batchSize = min(p->opts.maxbatchsize,ntrans);    // user override
  if (p->opts.spread_thread==0)
    p->opts.spread_thread=1;                   // the auto choice, for now
  
  // set others as defaults (or unallocated for arrays)...
  p->X = NULL; p->Y = NULL; p->Z = NULL;
  p->phiHat1 = NULL; p->phiHat2 = NULL; p->phiHat3 = NULL; 
  p->nf1 = 1; p->nf2 = 1; p->nf3 = 1;  // crucial to leave as 1 for unused dims
  p->ms = 1; p->mt = 1; p->mu = 1;     // crucial to leave as 1 for unused dims

  //  ------------------------ types 1,2: planning needed ---------------------
  if((type == 1) || (type == 2)) {

    int nth_fft = MY_OMP_GET_MAX_THREADS();  // tell FFTW what it has access to
    // *** should limt max # threads here too? or set equal to batchsize?
    // *** put in logic for setting FFTW # thr based on o.spread_thread?
    FFTW_INIT();           // only does anything when OMP=ON for >1 threads
    FFTW_PLAN_TH(nth_fft); // "  (not batchSize since can be 1 but want mul-thr)
    p->spopts.spread_direction = type;
    
    // read user mode array dims then determine fine grid sizes, sanity check...
    p->ms = n_modes[0];
    ier = set_nf_type12(p->ms,p->opts,p->spopts,&(p->nf1));
    if (ier) return ier;    // nf too big; we're done
    p->phiHat1 = (FLT*)malloc(sizeof(FLT)*(p->nf1/2 + 1));
    if (dim > 1) {
      p->mt = n_modes[1];
      ier = set_nf_type12(p->mt, p->opts, p->spopts, &(p->nf2));
      if (ier) return ier;
      p->phiHat2 = (FLT*)malloc(sizeof(FLT)*(p->nf2/2 + 1));
    }
    if (dim > 2) {
      p->mu = n_modes[2];
      ier = set_nf_type12(p->mu, p->opts, p->spopts, &(p->nf3)); 
      if (ier) return ier;
      p->phiHat3 = (FLT*)malloc(sizeof(FLT)*(p->nf3/2 + 1));
    }

    if (p->opts.debug)  // "long long" here is to avoid warnings with printf...
      printf("[finufft_plan] %dd%d: (ms,mt,mu)=(%lld,%lld,%lld) (nf1,nf2,nf3)=(%lld,%lld,%lld)\n               nthr=%d batchSize=%d spread_thread=%d\n",
             dim, type, (long long)p->ms,(long long)p->mt,
             (long long) p->mu, (long long)p->nf1,(long long)p->nf2,
             (long long)p->nf3, nth, p->batchSize,p->opts.spread_thread);

    // STEP 0: get Fourier coeffs of spreading kernel along each fine grid dim
    CNTime timer; timer.start();
    onedim_fseries_kernel(p->nf1, p->phiHat1, p->spopts);
    if (dim>1) onedim_fseries_kernel(p->nf2, p->phiHat2, p->spopts);
    if (dim>2) onedim_fseries_kernel(p->nf3, p->phiHat3, p->spopts);
    if (p->opts.debug) printf("[finufft_plan] kernel fser (ns=%d):\t\t%.3g s\n", p->spopts.nspread, timer.elapsedsec());

    p->N = p->ms*p->mt*p->mu;          // N = total # modes
    p->nf = p->nf1*p->nf2*p->nf3;      // fine grid total number of points
    p->fwBatch = FFTW_ALLOC_CPX(p->nf * p->batchSize);    // the big workspace
    if (p->opts.debug) printf("[finufft_plan] fwBatch %.2fGB alloc:\t\t%.3g s\n", (double)1E-09*sizeof(CPX)*p->nf*p->batchSize, timer.elapsedsec());
    if(!p->fwBatch) {      // we don't catch all such mallocs, just this big one
      fprintf(stderr, "FFTW malloc failed for fwBatch (working fine grids)!\n");
      free(p->phiHat1); free(p->phiHat2); free(p->phiHat3);
      return ERR_ALLOC; 
    }
   
    timer.restart();            // plan the FFTW
    int *ns = gridsize_for_fftw(p);
    // fftw_plan_many_dft args: rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
    p->fftwPlan = FFTW_PLAN_MANY_DFT(dim, ns, p->batchSize, p->fwBatch,
         NULL, 1, p->nf, p->fwBatch, NULL, 1, p->nf, p->fftSign, p->opts.fftw);
    if (p->opts.debug) printf("[finufft_plan] FFTW plan (mode %d, nth=%d):\t%.3g s\n", p->opts.fftw, nth_fft, timer.elapsedsec());
    delete []ns;
    
  } else {  // -------------------------- type 3 (no planning) ----------------

    if (p->opts.debug) printf("[finufft_plan] %dd%d\n",dim,type);
    printf("*** guru t3 gutted for now :(\n");
    p->fftwPlan = NULL;
    // type 3 will call finufft_makeplan for type 2, thus no need to init FFTW
  }
  return 0;
}


// SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
int finufft_setpts(finufft_plan* p, BIGINT nj, FLT* xj, FLT* yj, FLT* zj,
                   BIGINT nk, FLT* s, FLT* t, FLT* u)
/* For type 1,2: just checks and (possibly) sorts the NU points, in prep for
   spreading.
   For type 3: allocates internal working arrays, scales/centers the NU points
   and NU target freqs, evaluates spreading kernel FT at all target freqs.
*/
{
  CNTime timer; timer.start();
  p->nj = nj;    // the user choosing how many NU (x,y,z) pts
  
  if (p->type!=3) {   // ------------------ TYPE 1,2 SETPTS ---------------
    
    int ier = spreadcheck(p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->opts.debug>1) printf("[finufft_setpts] spreadcheck (%d):\t%.3g s\n", p->spopts.chkbnds, timer.elapsedsec());
    if (ier)
      return ier;    
    timer.restart();
    p->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*p->nj);
    if (!p->sortIndices) {
      fprintf(stderr,"[finufft_setpts] failed to allocate sortIndices!\n");
      return ERR_SPREAD_ALLOC;
    }
    p->didSort = indexSort(p->sortIndices, p->nf1, p->nf2, p->nf3, p->nj, xj, yj, zj, p->spopts);
    if (p->opts.debug) printf("[finufft_setpts] sort (didSort=%d):\t\t%.3g s\n", p->didSort, timer.elapsedsec());
    
    p->X = xj;  // keep pointers to user's data, which must be length >=nj
    p->Y = yj;
    p->Z = zj;

  } else {    // ------------------------- TYPE 3 SETPTS ---------------------
    
  }
  
  return 0;
}
// ............ end setpts ..................................................





// BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

int spreadinterpSortedBatch(int batchSize, finufft_plan* p, CPX* cBatch)
/*
  Spreads (or interpolates) a batch of batchSize strength vectors in cBatch
  to (or from) the batch of fine working grids p->fw, using the same set of
  (index-sorted) NU points p->X,Y,Z for each vector in the batch.
  The direction (spread vs interpolate) is set by p->spopts.spread_direction.
  Returns 0, no error reporting for now.
  Notes:
  1) cBatch is already assumed to have the correct offset, ie here we
     read from the start of cBatch (unlike Malleo).
  2) this routine is a batched version of spreadinterpSorted in spreadinterp.cpp
  Barnett 5/19/20, based on Malleo 2019.
*/
{
  // OMP nesting. 0: any omp-parallelism inside the loop sees only 1 thread;
  // note this doesn't change omp_get_max_nthreads()
  // 1: omp par inside the loop sees all threads.  *** ?
  MY_OMP_SET_NESTED(p->opts.spread_thread!=2);
  int nthr_outer = p->opts.spread_thread==1 ? 1 : batchSize;
  
#pragma omp parallel for num_threads(nthr_outer)
  for (int i=0; i<batchSize; i++) {
    FFTW_CPX *fwi = p->fwBatch + i*p->nf;  // start of i'th fw array in wkspace
    CPX *ci = cBatch + i*p->nj;            // start of i'th c array in cBatch
    spreadinterpSorted(p->sortIndices, p->nf1, p->nf2, p->nf3, (FLT*)fwi, p->nj,
                       p->X, p->Y, p->Z, (FLT*)ci, p->spopts, p->didSort);
  }

  MY_OMP_SET_NESTED(0);                    // back to default
  return 0;
}

int deconvolveBatch(int batchSize, finufft_plan* p, CPX* fkBatch)
/*
  Type 1: deconvolves (amplifies) from each interior fw array in p->fwBatch
  into each output array fk in fkBatch.
  Type 2: deconvolves from user-supplied input fk to 0-padded interior fw,
  again looping over fk in fkBatch and fw in p->fwBatch.
  The direction (spread vs interpolate) is set by p->spopts.spread_direction.
  This is mostly a loop calling deconvolveshuffle?d for the needed dim batchSize
  times.
  Barnett 5/21/20, simplified from Malleo 2019 (eg t3 logic won't be in here)
*/
{
  // since deconvolveshuffle?d are single-thread, omp par seems to help here...
#pragma omp parallel for
  for (int i=0; i<batchSize; i++) {
    FFTW_CPX *fwi = p->fwBatch + i*p->nf;  // start of i'th fw array in wkspace
    CPX *fki = fkBatch + i*p->N;           // start of i'th fk array in fkBatch
    
    // Call routine from common.cpp for the dim; prefactors hardcoded to 1.0...
    if (p->dim == 1)
      deconvolveshuffle1d(p->spopts.spread_direction, 1.0, p->phiHat1,
                          p->ms, (FLT *)fki,
                          p->nf1, fwi, p->opts.modeord);
    else if (p->dim == 2)
      deconvolveshuffle2d(p->spopts.spread_direction,1.0, p->phiHat1,
                          p->phiHat2, p->ms, p->mt, (FLT *)fki,
                          p->nf1, p->nf2, fwi, p->opts.modeord);
    else
      deconvolveshuffle3d(p->spopts.spread_direction, 1.0, p->phiHat1,
                          p->phiHat2, p->phiHat3, p->ms, p->mt, p->mu,
                          (FLT *)fki, p->nf1, p->nf2, p->nf3,
                          fwi, p->opts.modeord);
  }

  return 0;
}



// EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
int finufft_exec(finufft_plan* p, CPX* cj, CPX* fk){
/* For given (batch of) weights cj, performs NUFFTs with existing
   (sorted) NU pts and existing plan.
   Performs spread/interp, pre/post deconvolve, and fftw_exec as appropriate
   for each of the 3 types.
   For cases of ntrans>1, performs work in blocks of size up to batchSize.
   Return value 0, no error reporting yet.
   Barnett 5/20/20 based on Malleo 2019.
*/
  CNTime timer;
  double t_sprint = 0.0, t_fft = 0.0, t_deconv = 0.0;  // accumulated timings
  
  if (p->type!=3){ // --------------------- TYPE 1,2 EXEC ------------------
  
    for (int b=0; b*p->batchSize < p->ntrans; b++) { // .....loop b over batches

      // current batch is either batchSize, or possibly truncated if last one
      int thisBatchSize = min(p->ntrans - b*p->batchSize, p->batchSize);
      int i = b*p->batchSize;          // index of vector, since batchsizes same
      CPX* cjb = cj + i*p->nj;         // point to batch of weights
      // *** to insert some t3 logic here changing fkb...
      CPX* fkb = fk + i*p->N;          // point to batch of mode coeffs

      // STEP 1 (varies by type)
      if (p->type == 1) {  // type 1: spread NU pts p->X, weights cj, to fw grid
        timer.restart();
        spreadinterpSortedBatch(thisBatchSize, p, cjb);
        t_sprint += timer.elapsedsec();
      } else {          //  type 2: amplify Fourier coeffs fk into 0-padded fw
        timer.restart();
        deconvolveBatch(thisBatchSize, p, fkb);
        t_deconv += timer.elapsedsec();
      }
             
      // STEP 2: call the pre-planned FFT on this batch
      timer.restart();
      FFTW_EX(p->fftwPlan);           // *** what if thisBatchSize<batchSize?
      t_fft += timer.elapsedsec();
      if (p->opts.debug>1)
        printf("\tFFTW exec:\t\t%.3g s\n", timer.elapsedsec());
      
      // STEP 3 (varies by type)
      if (p->type == 1) {   // type 1: deconvolve (amplify) fw and shuffle to fk
        timer.restart();
        deconvolveBatch(thisBatchSize, p, fkb);
        t_deconv += timer.elapsedsec();
      } else {          // type 2: interpolate unif fw grid to NU target pts
        timer.restart();
        spreadinterpSortedBatch(thisBatchSize, p, cjb);
        t_sprint += timer.elapsedsec(); 
      }
    }                                                       // .....end b loop
    
    if (p->opts.debug){  // report total times in the order they would happen...
      if(p->type == 1) {
        printf("[finufft_exec] tot spread:\t\t\t%.3g s\n",t_sprint);
        printf("               tot FFT:\t\t\t\t%.3g s\n", t_fft);
        printf("               tot deconvolve:\t\t\t%.3g s\n", t_deconv);
      } else {
        printf("[finufft_exec] tot deconvolve:\t\t\t%.3g s\n", t_deconv);
        printf("               tot FFT:\t\t\t\t%.3g s\n", t_fft);
        printf("               tot interp:\t\t\t%.3g s\n",t_sprint);
      }
    }
  }

  else{  // ----------------------------- TYPE 3 EXEC ---------------------

  }
  
  return 0; 
}


// DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
int finufft_destroy(finufft_plan* p)
  // free everything we allocated inside of finufft_plan pointed to by p
{ 
  if (p->type==1 || p->type==2) {
    FFTW_DE(p->fftwPlan);  // destroy any FFTW plan (t1,2 only)
    FFTW_FR(p->fwBatch);   // free the FFTW working array
    free(p->phiHat1);
    free(p->phiHat2);
    free(p->phiHat3);
    free(p->sortIndices);
  } else {
    // for type 3, original coordinates are kept in {X,Y,Z}_orig,
    // but we must free the X,Y,Z we allocated which hold x',y',z':

    
  }
  
  return 0;
}

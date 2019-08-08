#include <finufft.h>
#include <utils.h>
#include <iostream>
#include <common.h>
#include <iomanip>

//forward declaration

//converts a finufft_type to corresponding integer for printing output
int typeToInt(finufft_type type);

//helper function to construct a n_dim size array, containing nf_i for i=1:n_dim
int * buildNf(finufft_plan *plan);


int finufft_makeplan(finufft_type type, int n_dims, BIGINT *n_modes, int iflag, int n_transf,
                     FLT tol, int threadBlkSize, finufft_plan *plan, nufft_opts opts) {

  plan->opts = opts;
  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts, tol, plan->opts);
  if(ier_set) return ier_set;

  cout << scientific << setprecision(15);  // for debug    

  plan->spopts = spopts;    
  plan->type = type;
  plan->n_dims = n_dims;
  plan->n_transf = n_transf;

  plan->tol = tol;
  plan->iflag = iflag;
  plan->threadBlkSize = threadBlkSize;
  plan->X = NULL;
  plan->Y = NULL;
  plan->Z = NULL;
  plan->X_orig = NULL;
  plan->Y_orig = NULL;
  plan->Z_orig = NULL;
  plan->sp = NULL;
  plan->tp = NULL;
  plan->up = NULL;
  plan->nf1 = 1;
  plan->nf2 = 1;
  plan->nf3 = 1;
  plan->isInnerT2 = false;
  plan->ms = 1;
  plan->mt = 1; 
  plan->mu = 1;
    

  /******************************************************************/
  /* Type 1 and Type 2                                              */
  /******************************************************************/

  if((type == type1) || (type == type2)){
  
    if (plan->threadBlkSize>1) {          
      FFTW_INIT();
      FFTW_PLAN_TH(plan->threadBlkSize);
    }


    plan->ms = n_modes[0];
    plan->mt = n_modes[1];
    plan->mu = n_modes[2];
    
    //determine size of upsampled array
    set_nf_type12(plan->ms, plan->opts, spopts, &(plan->nf1)); 
    if(n_dims > 1)
      set_nf_type12(plan->mt, plan->opts, spopts, &(plan->nf2)); 
    if(n_dims > 2)
      set_nf_type12(plan->mu, plan->opts, spopts, &(plan->nf3)); 
    
    

    if (plan->opts.debug) printf("%dd%d: (ms,mt,mu)=(%lld,%lld,%lld) (nf1,nf2,nf3)=(%lld,%lld,%lld) ...\n",n_dims, typeToInt(type),
                                 (long long)plan->ms,(long long)plan->mt, (long long) plan->mu,
                                 (long long)plan->nf1,(long long)plan->nf2, (long long)plan->nf3);

    //STEP 0: get Fourier coeffs of spreading kernel for each dim
    BIGINT totCoeffs;
    
    totCoeffs = plan->nf1/2 + 1; 
    if(n_dims > 1)
      totCoeffs  += (plan->nf2/2 +1);
    if(n_dims > 2)
      totCoeffs += (plan->nf3/2+1);


    CNTime timer; timer.start();
      
    plan->phiHat = (FLT *)malloc(sizeof(FLT)*totCoeffs);
    if(!plan->phiHat){
      fprintf(stderr, "Call to Malloc failed for Fourier coeff array allocation");
      return ERR_MAXNALLOC;
    }
    onedim_fseries_kernel(plan->nf1, plan->phiHat, plan->spopts);
    if(n_dims > 1) onedim_fseries_kernel(plan->nf2, plan->phiHat + (plan->nf1/2+1), plan->spopts);
    if(n_dims > 2) onedim_fseries_kernel(plan->nf3, plan->phiHat + (plan->nf1/2+1) + (plan->nf2/2+1), spopts);
    if (plan->opts.debug) printf("[make plan] kernel fser (ns=%d):\t\t %.3g s\n", spopts.nspread,timer.elapsedsec());    

    BIGINT nfTotal = plan->nf1*plan->nf2*plan->nf3;

    //do no more work than necessary if the number of transforms is smaller than the the threadBlkSize
    int transfPerBatch = min(plan->threadBlkSize, plan->n_transf); 

    //ensure size of upsampled grid does not exceed MAX
    if (nfTotal*transfPerBatch>MAX_NF) { 
      fprintf(stderr,"nf1*nf2*nf3*plan->threadBlkSize=%.3g exceeds MAX_NF of %.3g\n",
              (double)nfTotal*transfPerBatch,(double)MAX_NF);
      return ERR_MAXNALLOC;
    }

    plan->fw = FFTW_ALLOC_CPX(nfTotal*transfPerBatch);  

    if(!plan->fw){
      fprintf(stderr, "Call to malloc failed for working upsampled array allocation\n");
      free(plan->phiHat);
      return ERR_MAXNALLOC; 
    }
  
        
    int fftsign = (iflag>=0) ? 1 : -1;
    int * nf = buildNf(plan);
    
    timer.restart();
    //rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
    plan->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, nf, transfPerBatch, plan->fw, NULL, 1,
                                        nfTotal, plan->fw, NULL, 1, nfTotal, fftsign, plan->opts.fftw ) ;    

    if (plan->opts.debug) printf("[make plan] fftw plan (%d) \t\t %.3g s\n",plan->opts.fftw,timer.elapsedsec());
    delete []nf;                       
  }


  /******************************************************************/
  /* Type 3                                                         */
  /******************************************************************/

  else{
    plan->fftwPlan = NULL;
  }

  return 0;

};


int finufft_setpts(finufft_plan * plan , BIGINT nj, FLT *xj, FLT *yj, FLT *zj, BIGINT nk, FLT * s, FLT *t, FLT * u){

  plan->nj = nj;
  if(plan->X)
    free(plan->X);
  if(plan->Y)
    free(plan->Y);
  if(plan->Z)
    free(plan->Z);

  CNTime timer; timer.start();

  /******************************************************************/
  /* Type 1 and Type 2                                              */
  /******************************************************************/
  if((plan->type == type1) || (plan->type == type2)){

    if(plan->type == type1)
      plan->spopts.spread_direction = 1; 
    if(plan->type == type2)
      plan->spopts.spread_direction = 2; 

  
    int ier_check = spreadcheck(plan->nf1,plan->nf2 , plan->nf3, plan->nj, xj, yj, zj, plan->spopts);
    if(ier_check) return ier_check;

    timer.restart();
    plan->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*plan->nj);
    plan->didSort = indexSort(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, plan->nj, xj, yj, zj, plan->spopts);

    if (plan->opts.debug) printf("[finufft_setpts] sort (did_sort=%d):\t %.3g s\n", plan->didSort,
                                 timer.elapsedsec());
  

    plan->X = xj;
    plan->Y = yj;
    plan->Z = zj;

    plan->s = NULL;
    plan->t = NULL;
    plan->u = NULL;
  }

  /******************************************************************/
  /* Type 3                                                         */
  /******************************************************************/
  
  else{ 

    plan->nk = nk;
    
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
      printf("%d d3: X1=%.3g C1=%.3g S1=%.3g D1=%.3g gam1=%g nf1=%lld M=%lld N=%lld \n", plan->n_dims,
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

    if (plan->opts.debug) printf("[finufft_setpts] sort (did_sort=%d):\t %.3g s\n", plan->didSort,
                                 timer.elapsedsec());
    
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
};

/*Type 1 + Type 3: Spreads coordinate weights from c into internal workspace fw for sending into fftw */
void spreadAllSetsInBatch(int nSetsThisBatch, int blkNum, finufft_plan *plan, CPX * c, int *ier_spreads){

  //nSetsThisBatch is the threadBlockSize, except for the last round if threadBlockSize does not
  //divide evenly into n_transf. Ensures safe indexing of c.

  BIGINT fwRowSize = plan->nf1*plan->nf2*plan->nf3; 
  int blkJump = blkNum*plan->threadBlkSize; 
  
  int n_outerThreads = 0;
  if(plan->opts.spread_scheme){
    n_outerThreads = nSetsThisBatch;
  }
  
#if _OPENMP
  MY_OMP_SET_NESTED(1);
#endif
#pragma omp parallel for num_threads(n_outerThreads)
  for(int i = 0; i < nSetsThisBatch; i++){ 

    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = plan->fw + fwRowSize*i;

    //for type 3, c is "cpj", scaled weights, and spreading is done in batches of size threadBlockSize
    CPX *cStart;
    if(plan->type == type3)
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
#if _OPENMP
  MY_OMP_SET_NESTED(0);
#endif

}

/*Type 2: Interpolates from weights at uniform points in fw to non uniform points in c*/
void interpAllSetsInBatch(int nSetsThisBatch, int batchNum, finufft_plan *plan, CPX * c, int *ier_interps){

  BIGINT fwRowSize =  plan->nf1*plan->nf2*plan->nf3;
  int blkJump = batchNum*plan->threadBlkSize; 

  int n_outerThreads = 0;
  if(plan->opts.spread_scheme){
    n_outerThreads = nSetsThisBatch;
  }
  
#if _OPENMP
  MY_OMP_SET_NESTED(1);
#endif
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
#if _OPENMP
  MY_OMP_SET_NESTED(0);
#endif

}

/*Type 1: deconvolves from interior fw array into user supplied fk*/ 
/*Type 2: deconvolves from user supplied fk into interior fw array */
void deconvolveInParallel(int nSetsThisBatch, int batchNum, finufft_plan *plan, CPX *fk){

  //phiHat is a stacked version fwker in the old code 
  FLT *phiHat1 = plan->phiHat;
  FLT *phiHat2;
  FLT *phiHat3;
  if(plan->n_dims > 1 )
    phiHat2 = plan->phiHat + plan->nf1/2 + 1;
  if(plan->n_dims > 2)
    phiHat3 = plan->phiHat+(plan->nf1/2+1)+(plan->nf2/2+1);

  BIGINT fkRowSize = plan->ms*plan->mt*plan->mu;
  BIGINT fwRowSize = plan->nf1*plan->nf2*plan->nf3;
  int blockJump = batchNum*plan->threadBlkSize;

#pragma omp parallel for
  for(int i = 0; i < nSetsThisBatch; i++){

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
    if(plan->n_dims == 1){
      deconvolveshuffle1d(plan->spopts.spread_direction, 1.0, phiHat1, plan->ms, (FLT *)fkStart,
                          plan->nf1, fwStart, plan->opts.modeord);
    }
    else if (plan->n_dims == 2){
      deconvolveshuffle2d(plan->spopts.spread_direction,1.0, phiHat1, phiHat2,
                          plan->ms, plan->mt, (FLT *)fkStart,
                          plan->nf1, plan->nf2, fwStart, plan->opts.modeord);
    }
    else{
      deconvolveshuffle3d(plan->spopts.spread_direction, 1.0, phiHat1, phiHat2,
                          phiHat3, plan->ms, plan->mt, plan->mu,
                          (FLT *)fkStart, plan->nf1, plan->nf2, plan->nf3,
                          fwStart, plan->opts.modeord);
    }
  }
}


void type3PrePhaseInParallel(int nSetsThisBatch, int batchNum, finufft_plan * plan, CPX *cj, CPX *cpj){

  bool notZero = plan->t3P.D1 != 0.0;
  if(plan->n_dims > 1) notZero |=  (plan->t3P.D2 != 0.0);
  if(plan->n_dims > 2) notZero |=  (plan->t3P.D3 != 0.0);

  CPX imasign = (plan->iflag>=0) ? IMA : -IMA;
    
#pragma omp parallel for
  for (BIGINT i=0; i<plan->nj;i++){

    FLT sumCoords = plan->t3P.D1*plan->X_orig[i];

    if(plan->n_dims > 1)
      sumCoords += plan->t3P.D2*plan->Y_orig[i];
    if(plan->n_dims > 2)
      sumCoords += plan->t3P.D3*plan->Z_orig[i];
          
    CPX multiplier = exp(imasign*(sumCoords)); // rephase
          
    for(int k = 0; k < nSetsThisBatch; k++){
      int cpjIndex = k*plan->nj + i;
      int cjIndex = batchNum*plan->threadBlkSize*plan->nj + cpjIndex;

      cpj[cpjIndex] = cj[cjIndex]; 
      if(notZero)
	cpj[cpjIndex] *= multiplier;
    }
  }
}


/*Type 3: In place deconvolve of user supplied result array fk*/
void type3DeconvolveInParallel(int nSetsThisBatch, int batchNum, finufft_plan *plan, CPX *fk){

  CPX imasign = (plan->iflag>=0) ? IMA : -IMA;

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
        fkStart[k] *= exp(imasign*(sumCoords));    
    }
  }
}


int finufft_exec(finufft_plan * plan , CPX * cj, CPX * fk){

  CNTime timer; 
  double t_spread = 0.0;
  double t_exec = 0.0;
  double t_deconv = 0.0;
  
  int *ier_spreads = (int *)calloc(plan->threadBlkSize,sizeof(int));      


  /******************************************************************/
  /* Type 1 and Type 2                                              */
  /******************************************************************/
  if (plan->type != type3){
  
    for(int batchNum = 0; batchNum*plan->threadBlkSize < plan->n_transf; batchNum++){
          
      int nSetsThisBatch = min(plan->n_transf - batchNum*plan->threadBlkSize, plan->threadBlkSize);

      //Type 1 Step 1: Spread to Regular Grid    
      if(plan->type == type1){
        timer.restart();
        spreadAllSetsInBatch(nSetsThisBatch, batchNum, plan, cj, ier_spreads);
        t_spread += timer.elapsedsec();

        for(int i = 0; i < nSetsThisBatch; i++){
          if(ier_spreads[i])
            return ier_spreads[i];
        }
      }

      //Type 2 Step 1: amplify Fourier coeffs fk and copy into fw
      else if(plan->type == type2){
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
      if(plan->type == type1){
        timer.restart();
        deconvolveInParallel(nSetsThisBatch, batchNum, plan,fk);
        t_deconv += timer.elapsedsec();
      }

      //Type 2 Step 3: interpolate from regular to irregular target pts
      else if(plan->type == type2){
        timer.restart();
        interpAllSetsInBatch(nSetsThisBatch, batchNum, plan, cj, ier_spreads);
        t_spread += timer.elapsedsec(); 
      }
    }
    

    if(plan->opts.debug){
      if(plan->type == type1)
        printf("[finufft_exec] spread:\t\t\t %.3g s\n",t_spread);
      else //type 2
        printf("[finufft_exec] interp:\t\t\t %.3g s\n",t_spread);
      printf("[finufft_exec] fft :\t\t\t %.3g s\n", t_exec);
      printf("[finufft_exec] deconvolve :\t\t %.3g s\n", t_deconv);
    }
    
  }

  /******************************************************************/
  /* Type 3                                                         */
  /******************************************************************/

  else{

    //Allocate only nj*threadBlkSize array for scaled coordinate weights
    //this array will be recomputed for each batch/iteration 
    CPX *cpj = (CPX*)malloc(sizeof(CPX)*plan->nj*plan->threadBlkSize);  // c'_j rephased src
    if(!cpj){
      fprintf(stderr, "Call to malloc failed for rescaled input weights \n");
      return ERR_MAXNALLOC; 
    }

    BIGINT n_modes[3];
    n_modes[0] = plan->nf1;
    n_modes[1] = plan->nf2;
    n_modes[2] = plan->nf3;

    t_spread = 0;
    double t_innerExec= 0;
    double t_deConvShuff = 0;
    int ier_t2;
    
    //Preparations for the interior type 2 finufft call
    // 1) a single call to construct a finufft_plan
    // 2) a single call to finufft_setpts where scaled target freqs are type2 x,y,z coordinates 

    finufft_plan t2Plan;
    finufft_default_opts(&t2Plan.opts);
    t2Plan.opts.debug = plan->opts.debug;
    t2Plan.opts.spread_debug = plan->opts.spread_debug;

    int batchSize = min(plan->n_transf, plan->threadBlkSize);
    timer.restart();
    ier_t2 = finufft_makeplan(type2, plan->n_dims, n_modes, plan->iflag, batchSize, plan->tol,
                              plan->threadBlkSize, &t2Plan, t2Plan.opts);
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
  
};

int finufft_destroy(finufft_plan * plan){

  //free everything inside of finnufft_plan!
  
  if(plan->phiHat)
    free(plan->phiHat);

  if(plan->sortIndices)
    free(plan->sortIndices);

  if(plan->fftwPlan)
    FFTW_DE(plan->fftwPlan);

  if(plan->fw)
    FFTW_FR(plan->fw);
  
  
  //for type 3, original coordinates are kept in {X,Y,Z}_orig,
  //free the X,Y,Z which hold x',y',z'
  if(plan->type == type3){
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
  
};


int typeToInt(finufft_type type){
  switch(type){
  case type1:
    return 1;
  case type2:
    return 2;
  case type3:
    return 3;
  default:
    return 0;
  }
}

finufft_type intToType(int i){

  switch(i){
  case 1: return type1;
  case 2: return type2;
  case 3: return type3;
  default : return type1; //invalid 

  }
}

int * buildNf(finufft_plan *plan){
  int * nf;
  //rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
  if(plan->n_dims == 1){ 
    nf = new int[1];
    nf[0] = (int)plan->nf1;
  }
  else if (plan->n_dims == 2){ 
    nf = new int[2];
    nf[0] = (int)plan->nf2;
    nf[1] = (int)plan->nf1; 
  }   //fftw enforced row major ordering
  else{ 
    nf = new int[3];
    nf[0] = (int)plan->nf3;
    nf[1] = (int)plan->nf2;
    nf[2] = (int)plan->nf1;
  }
  return nf;
}

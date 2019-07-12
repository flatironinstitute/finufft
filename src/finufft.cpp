#include <finufft.h>
#include <finufft_old.h>
#include <invokeGuru.h>
#include <utils.h>
#include <iostream>
#include <common.h>
#include <iomanip>

//forward declaration
int typeToInt(finufft_type type);

/*Responsible for allocating arrays for fftw_execute output and instantiating fftw_plan*/
int make_finufft_plan(finufft_type type, int n_dims, BIGINT *n_modes, int iflag, int n_transf, FLT tol, finufft_plan *plan) {

  spread_opts spopts;
  int ier_set = setup_spreader_for_nufft(spopts, tol, plan->opts);
  if(ier_set) return ier_set;

  cout << scientific << setprecision(15);  // for debug    

  plan->spopts = spopts;    
  plan->type = type;
  plan->n_dims = n_dims;
  plan->n_transf = n_transf;
  plan->ms = n_modes[0];
  plan->mt = n_modes[1];
  plan->mu = n_modes[2];
  plan->tol = tol;
  plan->iflag = iflag;
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

  int nth = MY_OMP_GET_MAX_THREADS();
  if (nth>1) {             // set up multithreaded fftw stuff...
    FFTW_INIT();
    FFTW_PLAN_TH(nth);
  }
      
  if((type == type1) || (type == type2)){
    
    //determine size of upsampled array
    set_nf_type12(plan->ms, plan->opts, spopts, &(plan->nf1)); 
    if(n_dims > 1)
      set_nf_type12(plan->mt, plan->opts, spopts, &(plan->nf2)); 
    if(n_dims > 2)
      set_nf_type12(plan->mu, plan->opts, spopts, &(plan->nf3)); 
    
    
    //ensure size of upsampled grid does not exceed MAX
    if (plan->nf1*plan->nf2*plan->nf3*nth>MAX_NF) { 
      fprintf(stderr,"nf1*nf2*nf3*nth=%.3g exceeds MAX_NF of %.3g\n",(double)plan->nf1*plan->nf2*plan->nf3*nth,(double)MAX_NF);
      return ERR_MAXNALLOC;
    }
    cout << scientific << setprecision(15);  // for debug

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
      
    plan->fwker = (FLT *)malloc(sizeof(FLT)*totCoeffs);
    if(!plan->fwker){
      fprintf(stderr, "Call to Malloc failed for Fourier coeff array allocation");
      return ERR_MAXNALLOC;
    }

    CNTime timer; timer.start();
    
    onedim_fseries_kernel(plan->nf1, plan->fwker, plan->spopts);
    if(n_dims > 1) onedim_fseries_kernel(plan->nf2, plan->fwker + (plan->nf1/2+1), plan->spopts);
    if(n_dims > 2) onedim_fseries_kernel(plan->nf3, plan->fwker + (plan->nf1/2+1) + (plan->nf2/2+1), spopts);
    
    if (plan->opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());

  
    plan->fw = FFTW_ALLOC_CPX(plan->nf1*plan->nf2*plan->nf3*nth);  

    if(!plan->fw){
      fprintf(stderr, "Call to malloc failed for working upsampled array allocation");
      free(plan->fwker);
      return ERR_MAXNALLOC; 
    }
  
        
    int fftsign = (iflag>=0) ? 1 : -1;



    int * nf; 
    
    //rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
    if(n_dims == 1){ 
      nf = new int[1];
      nf[0] = (int)plan->nf1;
    }
    else if (n_dims == 2){ 
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

    timer.restart();
    plan->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, nf, nth, plan->fw, NULL, 1, plan->nf2*plan->nf1*plan->nf3, plan->fw,
                                        NULL, 1, plan->nf2*plan->nf1*plan->nf3, fftsign, plan->opts.fftw ) ;    
    if (plan->opts.debug) printf("fftw plan (%d)    \t %.3g s\n",plan->opts.fftw,timer.elapsedsec());
    delete []nf;                    
  
     
  }
  
  else{
    plan->fftwPlan = NULL;
    plan->nk = n_modes[0]*n_modes[1]*n_modes[2];
  }

  return 0;
};


int setNUpoints(finufft_plan * plan , BIGINT nj, FLT *xj, FLT *yj, FLT *zj, FLT * s, FLT *t, FLT * u){

  plan->nj = nj;
  if(plan->X)
    free(plan->X);
  if(plan->Y)
    free(plan->Y);
  if(plan->Z)
    free(plan->Z);

  CNTime timer; timer.start();
  
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

    if (plan->opts.debug) printf("[guru] sort (did_sort=%d):\t %.3g s\n", plan->didSort,
				 timer.elapsedsec());
  

    plan->X = xj;
    plan->Y = yj;
    plan->Z = zj;

    plan->s = NULL;
    plan->t = NULL;
    plan->u = NULL;
  }

  else{ //(plan->type == finufft_type::type3)

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

    if ((int64_t)plan->nf1*plan->nf2*plan->nf3*plan->n_transf>MAX_NF) {
      fprintf(stderr,"nf1*nf2*nf3*n_transf=%.3g exceeds MAX_NF of %.3g\n",(double)plan->nf1*plan->nf2*plan->nf3*plan->n_transf,(double)MAX_NF);
      return ERR_MAXNALLOC;
    }

    plan->fw = FFTW_ALLOC_CPX(plan->nf1*plan->nf2*plan->nf3*plan->n_transf);  

    if(!plan->fw){
      fprintf(stderr, "Call to malloc failed for working upsampled array allocation\n");
      return ERR_MAXNALLOC; 
    }

    plan->fwker = (FLT *)malloc(sizeof(FLT)*plan->nk*plan->n_dims);
    if(!plan->fwker){
      fprintf(stderr, "Call to Malloc failed for Fourier coeff array allocation\n");
      return ERR_MAXNALLOC;
    }
    
    FLT* xpj = (FLT*)malloc(sizeof(FLT)*plan->nj);
    if(!xpj){
      fprintf(stderr, "Call to malloc failed for rescaled coordinates\n");
      return ERR_MAXNALLOC; 
    }    
    FLT *ypj = NULL;
    FLT* zpj = NULL;

    if(plan->n_dims > 1){
      ypj = (FLT*)malloc(sizeof(FLT)*nj);
      if(!ypj){
	fprintf(stderr, "Call to malloc failed for rescaled coordinates\n");
	return ERR_MAXNALLOC; 
      }
    }
    if(plan->n_dims > 2){
      zpj = (FLT*)malloc(sizeof(FLT)*nj);
      if(!zpj){
	fprintf(stderr, "Call to malloc failed for rescaled coordinates\n");
	return ERR_MAXNALLOC; 
      }
    }

#pragma omp parallel for     
    for (BIGINT j=0;j<nj;++j) {
      xpj[j] = (xj[j] - plan->t3P.C1) / plan->t3P.gam1;          // rescale x_j
      if(plan->n_dims > 1)
	ypj[j] = (yj[j]- plan->t3P.C2) / plan->t3P.gam2;          // rescale y_j
      if(plan->n_dims > 2)
	zpj[j] = (zj[j] - plan->t3P.C3) / plan->t3P.gam3;          // rescale z_j
    }

    int ier_check = spreadcheck(plan->nf1,plan->nf2 , plan->nf3, plan->nj, xpj, ypj, zpj, plan->spopts);
    if(ier_check) return ier_check;

    timer.restart();
    plan->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*plan->nj);
    plan->didSort = indexSort(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, plan->nj, xpj, ypj, zpj, plan->spopts);

    if (plan->opts.debug) printf("[guru] sort (did_sort=%d):\t %.3g s\n", plan->didSort,
				 timer.elapsedsec());
    
    plan->X = xpj;
    plan->X_orig = xj;
    plan->Y = ypj;
    plan->Y_orig = yj;
    plan->Z = zpj;
    plan->Z_orig = zj;
    
    
    FLT *sp = (FLT*)malloc(sizeof(FLT)*plan->nk);     // rescaled targs s'_k
    if(!sp){
      fprintf(stderr, "Call to malloc failed for rescaled target freqs\n");
      return ERR_MAXNALLOC; 
    }
    
    FLT *tp = NULL;
    if(plan->n_dims > 1 ){
      tp = (FLT*)malloc(sizeof(FLT)*plan->nk);     // t'_k
      if(!tp){
	fprintf(stderr, "Call to malloc failed for rescaled target freqs\n");
	return ERR_MAXNALLOC; 
      }
    }

    FLT *up = NULL;
    if(plan->n_dims > 2 ){
      up = (FLT*)malloc(sizeof(FLT)*plan->nk);     // u'_k
      if(!up){
	fprintf(stderr, "Call to malloc failed for rescaled target freqs\n");
	return ERR_MAXNALLOC; 
      }
    }

    // rescaled targs s'_k
    timer.restart();
#pragma omp parallel for 
    for (BIGINT k=0;k<plan->nk;++k) {
	sp[k] = plan->t3P.h1*plan->t3P.gam1*(s[k]-plan->t3P.D1);      // so that |s'_k| < pi/R
	if(plan->n_dims > 1 )
	  tp[k] = plan->t3P.h2*plan->t3P.gam2*(t[k]-plan->t3P.D2);      // so that |t'_k| < pi/R
	if(plan->n_dims > 2)
	  up[k] = plan->t3P.h3*plan->t3P.gam3*(u[k]-plan->t3P.D3);      // so that |u'_k| < pi/R
    }
    if(plan->opts.debug) printf("[guru] rescaling target-freqs: \t %.3g s\n", timer.elapsedsec());

    // Step 3a: compute Fourier transform of scaled kernel at targets
    timer.restart();
   
    //fwker := fkker  
    // exploit that Fourier transform separates because kernel built separable...
    onedim_nuft_kernel(plan->nk, sp, plan->fwker, plan->spopts);           // fill fkker1
    if(plan->n_dims > 1)
      onedim_nuft_kernel(plan->nk, tp, plan->fwker + plan->nk, plan->spopts);           // etc
    if(plan->n_dims > 2)
      onedim_nuft_kernel(plan->nk, up, plan->fwker + 2*plan->nk, plan->spopts);
    if (plan->opts.debug) printf("kernel FT (ns=%d):\t %.3g s\n", plan->spopts.nspread,timer.elapsedsec());


    
    plan->s = s;
    plan->sp = sp;
    //null if 1 dim
    plan->t = t;
    plan->tp = tp;  
    //null if 2 dim
    plan->u = u;
    plan->up = up;
    
  }
  
  return 0;
};

void spreadInParallel(int blksize, int j, int nth, finufft_plan *plan, CPX * c, int *ier_spreads){


  int maxi = 0;
  if(plan->type == type3)
    maxi = plan->n_transf;
  else
    maxi = blksize;
  
#pragma omp parallel for 
  for(int i = 0; i < maxi ; i++){ 

    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = plan->fw + plan->nf1*plan->nf2*plan->nf3*i;

    CPX * cStart = c + plan->nj*(i + j*nth); //type3 j = 0
          
    int ier = spreadSorted(plan->sortIndices,
                           plan->nf1, plan->nf2, plan->nf3, (FLT*)fwStart,
                           plan->nj, plan->X, plan->Y, plan->Z, (FLT *)cStart,
                           plan->spopts, plan->didSort) ;
    if(ier)
      ier_spreads[i] = ier;
  }
}

void interpInParallel(int blksize, int j, int nth, finufft_plan *plan, CPX * c, int *ier_interps){

#pragma omp parallel for 
  for(int i = 0; i < blksize; i++){ 
        
    //index into this iteration of fft in fw and weights arrays
    FFTW_CPX *fwStart = plan->fw + plan->nf1*plan->nf2*plan->nf3*i; //fw gets reread on each iteration of j

    CPX * cStart = c + plan->nj*(i + j*nth);
          
    int ier = interpSorted(plan->sortIndices,
                           plan->nf1, plan->nf2, plan->nf3, (FLT*)fwStart,
                           plan->nj, plan->X, plan->Y, plan->Z, (FLT *)cStart,
                           plan->spopts, plan->didSort) ;

    if(ier)
      ier_interps[i] = ier;
  }
}


void deconvolveInParallel(int blksize, int j, int nth, finufft_plan *plan, CPX *fk){

#pragma omp parallel for
  for(int i = 0; i < blksize; i++){

    CPX *fkStart = fk + (i+j*nth)*plan->ms*plan->mt*plan->mu;
    FFTW_CPX *fwStart = plan->fw + plan->nf1*plan->nf2*plan->nf3*i;
    FLT *fwker1 = plan->fwker;
    FLT *fwker2;
    FLT *fwker3;
    if(plan->n_dims > 1 )
      fwker2 = plan->fwker + plan->nf1/2 + 1;
    if(plan->n_dims > 2)
      fwker3 = plan->fwker+(plan->nf1/2+1)+(plan->nf2/2+1);
    
    
    //prefactors ?
    if(plan->n_dims == 1){
      deconvolveshuffle1d(plan->spopts.spread_direction, 1.0, fwker1, plan->ms, (FLT *)fkStart,
                          plan->nf1, fwStart, plan->opts.modeord);
    }
    else if (plan->n_dims == 2){
      deconvolveshuffle2d(plan->spopts.spread_direction,1.0, fwker1, fwker2,
                          plan->ms, plan->mt, (FLT *)fkStart,
                          plan->nf1, plan->nf2, fwStart, plan->opts.modeord);
    }
    else{
      deconvolveshuffle3d(plan->spopts.spread_direction, 1.0, fwker1, fwker2,
                          fwker3, plan->ms, plan->mt, plan->mu,
                          (FLT *)fkStart, plan->nf1, plan->nf2, plan->nf3,
			  fwStart, plan->opts.modeord);
    }
      
  }
}

void type3DeconvolveInParallel(int blksize, int j, int nth, finufft_plan *plan, CPX *fk){

  CPX imasign = (plan->iflag>=0) ? IMA : -IMA;

  bool finite  = isfinite(plan->t3P.C1);
  if(plan->n_dims > 1 ) finite &=  isfinite(plan->t3P.C2);
  if(plan->n_dims > 2 ) finite &=  isfinite(plan->t3P.C3);
  bool notzero = plan->t3P.C1!=0.0;
  if(plan->n_dims > 1 ) notzero |=  (plan->t3P.C2 != 0.0);
  if(plan->n_dims > 2 ) notzero |=  (plan->t3P.C3 != 0.0);

  
#pragma omp parallel for
  for(int i = 0; i < blksize; i++){
    CPX *fkStart = fk + (i+j*nth)*plan->nk;

    FLT * fwker1 = plan->fwker;
    FLT * fwker2;
    FLT * fwker3;
    if(plan->n_dims > 1)
      fwker2 = plan->fwker + plan->nk;
    if(plan->n_dims > 2)
      fwker3 = fwker2 + plan->nk;

    if(finite && notzero){
#pragma omp parallel for schedule(dynamic)              // since cexps slow
      for (BIGINT k=0;k<plan->nk;++k){         // also phases to account for C1,C2,C3 shift
	
        FLT sumCoords = (plan->s[k] - plan->t3P.D1)*plan->t3P.C1;
        FLT prodfwKer = fwker1[k];
        if(plan->n_dims > 1 ){
          sumCoords += (plan->t[k] - plan->t3P.D2)*plan->t3P.C2 ;
          prodfwKer *= fwker2[k];
        }
        if(plan->n_dims > 2){
          sumCoords += (plan->u[k] - plan->t3P.D3)*plan->t3P.C3;
          prodfwKer *= fwker3[k];
        }
        fkStart[k] *= (CPX)(1.0/prodfwKer)*exp(imasign*(sumCoords));
      }
    }
    
    else{
  
#pragma omp parallel for schedule(dynamic)
      for (BIGINT k=0;k<plan->nk;++k){    
        FLT prodfwKer = fwker1[k];
        if(plan->n_dims >1 )
          prodfwKer *= fwker2[k];
        if(plan->n_dims > 2 )
          prodfwKer *= fwker3[k];
        fkStart[k] *= (CPX)(1.0/prodfwKer);
      }
    }
  }
}


int finufft_exec(finufft_plan * plan , CPX * cj, CPX * fk){

  CNTime timer; 
  double time_spread = 0.0;
  double time_exec = 0.0;
  double time_deconv = 0.0;

  int nth = MY_OMP_GET_MAX_THREADS();

    
#if _OPENMP
  MY_OMP_SET_NESTED(0); //no nested parallelization
#endif
  
  int *ier_spreads = (int *)calloc(nth,sizeof(int));      
  
  if (plan->type != type3){
  
    for(int j = 0; j*nth < plan->n_transf; j++){
          
      int blksize = min(plan->n_transf - j*nth, nth);


      //Type 1 Step 1: Spread to Regular Grid    
      if(plan->type == type1){
	timer.restart();
	spreadInParallel(blksize, j, nth, plan, cj, ier_spreads);
	time_spread += timer.elapsedsec();

	for(int i = 0; i < blksize; i++){
	  if(ier_spreads[i])
	    return ier_spreads[i];
	}
	if(plan->opts.debug) printf("[guru] spread:\t\t\t %.3g s\n",time_spread);
      }

      //Type 2 Step 1: amplify Fourier coeffs fk and copy into fw
      else if(plan->type == type2){
	timer.restart();
	deconvolveInParallel(blksize, j, nth, plan,fk);
	time_deconv += timer.elapsedsec();
	if(plan->opts.debug) printf("deconvolve & copy out:\t\t %.3g s\n", time_deconv);
      }
        
      
      //Type 1/2 Step 2: Call FFT
   
      timer.restart();
      FFTW_EX(plan->fftwPlan);
      time_exec += timer.elapsedsec();
      if(plan->opts.debug) printf("[guru] fft :\t\t\t %.3g s\n", time_exec);        
   
    
    
      //Type 1 Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output 
      if(plan->type == type1){
	timer.restart();
	deconvolveInParallel(blksize, j, nth, plan,fk);
	time_deconv += timer.elapsedsec();
	if(plan->opts.debug) printf("deconvolve & copy out:\t\t %.3g s\n", time_deconv);
      }

      //Type 2 Step 3: interpolate from regular to irregular target pts
      else if(plan->type == type2){
	timer.restart();
	interpInParallel(blksize, j, nth, plan, cj, ier_spreads);
	time_spread += timer.elapsedsec(); 

	for(int i = 0; i < blksize; i++){
	  if(ier_spreads[i])
	    return ier_spreads[i];
	}
	if(plan->opts.debug) printf("[guru] interp:\t\t\t %.3g s\n",time_spread);
      }
    }
  }

  //type 3
  else{

    CPX imasign = (plan->iflag>=0) ? IMA : -IMA;
    CPX *cpj = (CPX*)malloc(sizeof(CPX)*plan->nj*plan->n_transf);  // c'_j rephased src
    if(!cpj){
      fprintf(stderr, "Call to malloc failed for rescaled input weights \n");
      return ERR_MAXNALLOC; 
    }

    bool notZero = plan->t3P.D1 != 0.0;
    if(plan->n_dims > 1) notZero |=  (plan->t3P.D2 != 0.0);
    if(plan->n_dims > 2) notZero |=  (plan->t3P.D3 != 0.0);

    timer.restart();
    if (notZero) { 
#pragma omp parallel for schedule(dynamic)                // since cexp slow
      for (BIGINT j=0;j<plan->nj;j++){
	FLT sumCoords = plan->t3P.D1*plan->X_orig[j];
	if(plan->n_dims > 1)
	  sumCoords += plan->t3P.D2*plan->Y_orig[j];
	if(plan->n_dims > 2)
	  sumCoords += plan->t3P.D3*plan->Z_orig[j];
	for(int k = 0; k < plan->n_transf; k++)
	  cpj[k*plan->nj + j] = cj[k*plan->nj + j] * exp(imasign*(sumCoords)); // rephase
      }
    }
    
    else{
      for (BIGINT j=0;j<plan->nj*plan->n_transf;j++)
	cpj[j] = cj[j];                                    // just copy over
    }

    if (plan->opts.debug) printf("prephase:\t\t %.3g s\n",timer.elapsedsec());
    
    int j = 0;
    int blksize = 0 ; //vestigial
    int *ier_spreads3 = (int *)calloc(plan->n_transf,sizeof(int));      
      
    timer.restart();
    spreadInParallel(blksize, j, nth, plan, cpj, ier_spreads3);
    time_spread += timer.elapsedsec();
      
    for(int i = 0; i < plan->n_transf; i++){
      if(ier_spreads3[i])
	return ier_spreads3[i];
    }

    if(plan->opts.debug) printf("[guru] spread:\t\t\t %.3g s\n",time_spread);
    
    free(ier_spreads3);
    free(cpj);
    
    timer.restart();

    BIGINT n_modes[3];
    n_modes[0] = plan->nf1;
    n_modes[1] = plan->nf2;
    n_modes[2] = plan->nf3;
  
    int ier_t2 = invokeGuruInterface(plan->n_dims, type2, plan->n_transf, plan->nk, plan->sp, plan->tp, plan->up,
				     fk, plan->iflag, plan->tol, n_modes, NULL, NULL, NULL, 
				     (CPX *)plan->fw, plan->opts);
  
  
    if (plan->opts.debug) printf("total type-2 (ier=%d):\t %.3g s\n",ier_t2,timer.elapsedsec());
    if (ier_t2>0) exit(ier_t2);

    timer.restart();
    for(int j = 0; j*nth < plan->n_transf; j++){

      int blksize = min(plan->n_transf - j*nth, nth);
      type3DeconvolveInParallel(blksize, j, nth, plan,fk); 

    }

    if (plan->opts.debug) printf("deconvolve:\t\t %.3g s\n",timer.elapsedsec());
    
  }
  
  free(ier_spreads);

  return 0;
  
};

int finufft_destroy(finufft_plan * plan){

  //free everything inside of finnufft_plan!
  
  if(plan->fwker)
    free(plan->fwker);

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
  default : return type1; //barf invalid 

  }
}


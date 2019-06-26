#include "finufft.h"
#include <utils.h>
#include <iostream>
#include <common.h>
#include <iomanip>

/*Responsible for allocating arrays for fftw_execute output and instantiating fftw_plan*/
int make_finufft_plan(finufft_type type, int n_dims, BIGINT *n_modes, int iflag, int how_many, FLT tol, finufft_plan *plan) {

  //ONLY 1/2D TYPE 1+2
  if(type != finufft_type::type3 && n_dims != 3){


    //THINK HARDER about this brittle code
    //user may have edited the opts struct inside of the plan before calling this routine
    //or it may be completely uninitialized. To check if latter, suffices to check upsampfac.
    
    if(plan->opts.upsampfac != 2 && plan->opts.upsampfac != 1.25 ){ //uninitialized
      nufft_opts def_opts;
      finufft_default_opts(&def_opts);
      plan->opts = {def_opts};
    }

    spread_opts spopts;
    int ier_set = setup_spreader_for_nufft(spopts, tol, plan->opts);
    if(ier_set) return ier_set;
    
    plan->spopts = spopts;    
    plan->type = type;
    plan->n_dims = n_dims;
    plan->how_many = how_many;
    plan->ms = n_modes[0];
    plan->mt = n_modes[1];
    plan->mu = n_modes[2];
    
    plan->iflag = iflag;
    plan->X = NULL;
    plan->Y = NULL;
    plan->Z = NULL;
    
    //determine size of upsampled array
    BIGINT nf1 = 1;
    BIGINT nf2 = 1;
    BIGINT nf3 = 1;
    set_nf_type12(plan->ms, plan->opts, spopts, &nf1); //type/DIMENSION dependent line
    if(n_dims > 1)
    set_nf_type12(plan->mt, plan->opts, spopts, &nf2); //type/DIMENSION dependent line
    
    
    //ensure size of upsampled grid does not exceed MAX
    if (nf1*nf2>MAX_NF) {  //DIMENSION DEPENDENT LINE
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
    }
    cout << scientific << setprecision(15);  // for debug

    //DIMENSION DPEENDENT LINE
    if (plan->opts.debug) printf("2d1: (ms,mt)=(%lld,%lld) (nf1,nf2)=(%lld,%lld) ...\n",(long long)plan->ms,(long long)plan->mt,(long long)nf1,(long long)nf2);

    //STEP 0: get Fourier coeffs of spreading kernel for each dim
    BIGINT totCoeffs;
    if(n_dims == 1)
      totCoeffs = nf1/2 + 1; 
    else if(n_dims == 2)
     totCoeffs  = (nf1/2 + 1)+(nf2/2 +1); //DIMENSION DEPENDENT LINE
    plan->fwker = (FLT *)malloc(sizeof(FLT)*totCoeffs);
    if(!plan->fwker){
      fprintf(stderr, "Call to Malloc failed for Fourier coeff array allocation");
      return ERR_MAXNALLOC;
    }

    CNTime timer; timer.start();
    //DIMENSION DEPENDENT LINES:
    onedim_fseries_kernel(nf1, plan->fwker, spopts);
    if(n_dims > 1) onedim_fseries_kernel(nf2, plan->fwker + (nf1/2+1), spopts);
    if (plan->opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());

    int nth = MY_OMP_GET_MAX_THREADS();
    if (nth>1) {             // set up multithreaded fftw stuff...
      FFTW_INIT();
      FFTW_PLAN_TH(nth);
    }


    plan->nf1 = nf1;
    plan->nf2 = nf2;
    plan->nf3 = nf3;
    
    plan->fw = FFTW_ALLOC_CPX(nf1*nf2*nth);  

    if(!plan->fw){
      fprintf(stderr, "Call to malloc failed for working upsampled array allocation");
      free(plan->fwker);
      return ERR_MAXNALLOC; //release resources before exiting cleanly
    }
    
    int fftsign = (iflag>=0) ? 1 : -1;

    timer.restart();

    //rank, gridsize/dim, howmany, in, inembed, istride, idist, ot, onembed, ostride, odist, sign, flags 
    if(n_dims == 1){
      const int nf[] = {(int)nf1};
      plan->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, nf, nth, plan->fw, NULL, 1, nf2*nf1, plan->fw,
					  NULL, 1, nf2*nf1, fftsign, plan->opts.fftw ) ;
    }
    else{
    //fftw enforced row major ordering
      const int nf[] {(int)nf2, (int)nf1};
      plan->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, nf, nth, plan->fw, NULL, 1, nf2*nf1, plan->fw,
					  NULL, 1, nf2*nf1, fftsign, plan->opts.fftw ) ;
    }

    if (plan->opts.debug) printf("fftw plan (%d)    \t %.3g s\n",plan->opts.fftw,timer.elapsedsec());
     

    return 0;
  }
  
  else return -1;
};


int setNUpoints(finufft_plan * plan , BIGINT M, FLT *Xpts, FLT *Ypts, FLT *Zpts, CPX *targetFreqs){
  if(plan->type == type1)
    plan->spopts.spread_direction = 1; ///FIX THIS WILLY NILLY INITIALIZATION
  if(plan->type == type2)
    plan->spopts.spread_direction = 2; ///FIX THIS WILLY NILLY INITIALIZATION

  plan->M = M;
  int ier_check = spreadcheck(plan->nf1,plan->nf2 , plan->nf3, plan->M, Xpts, Ypts, Zpts, plan->spopts);
  if(ier_check) return ier_check;

  CNTime timer; timer.start();
  plan->sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*plan->M);
  plan->didSort = indexSort(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, plan->M, Xpts, Ypts, Zpts, plan->spopts);

  if (plan->opts.debug) printf("[guru] sort (did_sort=%d):\t %.3g s\n", plan->didSort,
  			      timer.elapsedsec());
  
  if(plan->X)
    free(plan->X);
  if(plan->Y)
    free(plan->Y);
  if(plan->Z)
    free(plan->Z);

  plan->X = Xpts;
  plan->Y = Ypts;
  plan->Z = Zpts;
  return 0;
  
};

void spreadInParallel(int blksize, int j, int nth, finufft_plan *plan, CPX * c, int *ier_spreads){

#pragma omp parallel for 
	for(int i = 0; i < blksize; i++){ 
      	
	  //index into this iteration of fft in fw and weights arrays
	  FFTW_CPX *fwStart = plan->fw + plan->nf1*plan->nf2*i; //fw gets rewritten on each iteration of j

	  CPX * cStart = c + plan->M*(i + j*nth);
	  
	  int ier = spreadSorted(plan->sortIndices,
				    plan->nf1, plan->nf2, plan->nf3, (FLT*)fwStart,
				    plan->M, plan->X, plan->Y, plan->Z, (FLT *)cStart,
				    plan->spopts, plan->didSort) ;
	  if(ier)
	    ier_spreads[i] = ier;
	}
}

void interpInParallel(int blksize, int j, int nth, finufft_plan *plan, CPX * c, int *ier_interps){

#pragma omp parallel for 
	for(int i = 0; i < blksize; i++){ 
      	
	  //index into this iteration of fft in fw and weights arrays
	  FFTW_CPX *fwStart = plan->fw + plan->nf1*plan->nf2*i; //fw gets rewritten on each iteration of j

	  CPX * cStart = c + plan->M*(i + j*nth);
	  
	  int ier = interpSorted(plan->sortIndices,
				 plan->nf1, plan->nf2, plan->nf3, (FLT*)fwStart,
				 plan->M, plan->X, plan->Y, plan->Z, (FLT *)cStart,
				 plan->spopts, plan->didSort) ;

	  if(ier)
	    ier_interps[i] = ier;
	}
}


void deconvolveInParallel(int blksize, int j, int nth, finufft_plan *plan, CPX *result){
#pragma omp parallel for
      for(int i = 0; i < blksize; i++){
	CPX *fkStart = result + (i+j*nth)*plan->ms*plan->mt;
	FFTW_CPX *fwStart = plan->fw + plan->nf1*plan->nf2*i;

	//prefactors 
	if(plan->n_dims == 1)
	  deconvolveshuffle1d(plan->spopts.spread_direction, 1.0, plan->fwker, plan->ms, (FLT *)fkStart,
			      plan->nf1, fwStart, plan->opts.modeord);
	  
	else if (plan->n_dims == 2)
	  deconvolveshuffle2d(plan->spopts.spread_direction,1.0,plan->fwker, plan->fwker+(plan->nf1/2+1),
			    plan->ms, plan->mt, (FLT *)fkStart,
			    plan->nf1, plan->nf2, fwStart, plan->opts.modeord);
      }
}


int finufft_exec(finufft_plan * plan , CPX * c, CPX * result){

  CNTime timer; 
  double time_spread{0.0};
  double time_exec{0.0};
  double time_deconv{0.0};

  int nth = MY_OMP_GET_MAX_THREADS();

#if _OPENMP
  MY_OMP_SET_NESTED(0); //no nested parallelization
#endif
  
  int *ier_spreads = (int *)calloc(nth,sizeof(int));      
	
  for(int j = 0; j*nth < plan->how_many; j++){
	  
    int blksize = min(plan->how_many - j*nth, nth);


    //Type 1 Step 1: Spread to Regular Grid    
    if(plan->type == type1){
      timer.restart();
      spreadInParallel(blksize, j, nth, plan, c, ier_spreads);
      time_spread += timer.elapsedsec();

      for(int i = 0; i < blksize; i++){
	if(ier_spreads[i])
	  return ier_spreads[i];
      }
    }

    //Type 2 Step 1: amplify Fourier coeffs fk and copy into fw
    else if(plan->type == type2){
      timer.restart();
      deconvolveInParallel(blksize, j, nth, plan,result);
      time_deconv += timer.elapsedsec();
    }
	
      
    //Step 2: Call FFT
    timer.restart();
    FFTW_EX(plan->fftwPlan);
    time_exec += timer.elapsedsec();

	  

    //Type 1 Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output	
    if(plan->type == type1){
      timer.restart();
      deconvolveInParallel(blksize, j, nth, plan,result);
      time_deconv += timer.elapsedsec();
    }

    //Type 2 Step 3: interpolate from regular to irregular target pts
    else if(plan->type == type2){
      timer.restart();
      interpInParallel(blksize, j, nth, plan, c, ier_spreads);
      time_spread += timer.elapsedsec(); 
    
    for(int i = 0; i < blksize; i++){
      if(ier_spreads[i])
	return ier_spreads[i];
    }
    }
  }   
  if(plan->opts.debug) printf("[guru] spread:\t\t\t %.3g s\n",time_spread);
  if(plan->opts.debug) printf("[guru] fft :\t\t\t %.3g s\n", time_exec);
  if(plan->opts.debug) printf("deconvolve & copy out:\t\t %.3g s\n", time_deconv);
      
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
  

  return 0;
  
};

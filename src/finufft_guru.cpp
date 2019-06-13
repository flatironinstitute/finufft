#include "finufft_guru.h"
#include <iostream>
#include <common.h>
#include <iomanip>

/*Responsible for allocating arrays for fftw_execute output and instantiating fftw_plan*/
int make_finufft_plan(finufft_type type, int n_dims, BIGINT *n_modes, int iflag, int how_many, FLT tol, finufft_plan *plan) {

  //ONLY 2D TYPE 1
  if(type == finufft_type::type1 && n_dims == 2){

    //TO DO - re-experiment with initialization bug through Matlab
    nufft_opts opts;
    finufft_default_opts(&opts);
    
    spread_opts spopts;
    int ier_set = setup_spreader_for_nufft(spopts, tol, opts);
    if(ier_set) return ier_set;
    
    plan->spopts = spopts;    
    plan->type = type;
    plan->n_dims = n_dims;
    plan->how_many = how_many;
    plan->ms = n_modes[0];
    plan->mt = n_modes[1];
    plan->mu = n_modes[2];
    
    plan->iflag = iflag;
    //plan->fw_width = ?
    plan->opts = opts;

    plan->X = NULL;
    plan->Y = NULL;
    plan->Z = NULL;
    
    //determine size of upsampled array
    BIGINT nf1;
    set_nf_type12(plan->ms, opts, spopts, &nf1); //type/DIMENSION dependent line
    BIGINT nf2;
    set_nf_type12(plan->mt, opts, spopts, &nf2); //type/DIMENSION dependent line
    BIGINT nf3{1};
    
    //ensure size of upsampled grid does not exceed MAX
    if (nf1*nf2>MAX_NF) {  //DIMENSION DEPENDENT LINE
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
    }
    cout << scientific << setprecision(15);  // for debug

    //DIMENSION DPEENDENT LINE
    if (opts.debug) printf("2d1: (ms,mt)=(%lld,%lld) (nf1,nf2)=(%lld,%lld) ...\n",(long long)plan->ms,(long long)plan->mt,(long long)nf1,(long long)nf2);

    //STEP 0: get Fourier coeffs of spreading kernel for each dim
    CNTime timer; timer.start();
    BIGINT totCoeffs = (nf1/2 + 1)+(nf2/2 +1); //DIMENSION DEPENDENT LINE
    plan->fwker = (FLT *)malloc(sizeof(FLT)*totCoeffs);
    if(!plan->fwker){
      fprintf(stderr, "Call to Malloc failed for Fourier coeff array allocation");
      return ERR_MAXNALLOC;
    }

    //DIMENSION DEPENDENT LINES:
    onedim_fseries_kernel(nf1, plan->fwker, spopts);
    onedim_fseries_kernel(nf2, plan->fwker + (nf1/2+1), spopts);
    if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());

    int nth = MY_OMP_GET_MAX_THREADS();
    if (nth>1) {             // set up multithreaded fftw stuff...
      FFTW_INIT();
      FFTW_PLAN_TH(nth);
    }
    timer.restart();

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

    //TYPE/DIMENSION/HOW MANY DEPENDENT 

    //fftw enforced row major ordering
    const int nf[] {(int)nf2, (int)nf1};

    //HOW_MANY INSTEAD OF NTH?
    plan->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, nf, nth, plan->fw, nf, 1, nf2*nf1, plan->fw, nf, 1, nf2*nf1, fftsign, opts.fftw ) ; 

    if (opts.debug) printf("fftw plan (%d)    \t %.3g s\n",opts.fftw,timer.elapsedsec());
     

    return 0;
  }
  
  else return -1;
};


int setNUpoints(finufft_plan * plan , BIGINT M, FLT *Xpts, FLT *Ypts, FLT *Zpts, CPX *targetFreqs){

  plan->spopts.spread_direction = 1; ///FIX THIS WILLY NILLY INITIALIZATION
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




int finufft_exec(finufft_plan * plan , CPX * c, CPX * result){

  CNTime timer; 
  double time_spread{0.0};
  double time_exec{0.0};
  double time_deconv{0.0};
  int nth = MY_OMP_GET_MAX_THREADS();

  //this screams inheritance? 
  switch(plan->type){

  
  //Step 1: Spread to Regular Grid
  case type1:
    
    if(plan->spopts.spread_direction == 1){

  
       #if _OPENMP
	MY_OMP_SET_NESTED(0); //no nested parallelization
       #endif
  
      int *ier_spreads = (int *)calloc(nth,sizeof(int));
      
      //if (how_many == 1), this loop only executes once 
      for(int j = 0; j*nth < plan->how_many; j++){

	int blksize = min(plan->how_many - j*nth, nth);
	timer.restart();
	
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
	time_spread += timer.elapsedsec();
	
      for(int i = 0; i < blksize; i++){
	if(ier_spreads[i])
	  return ier_spreads[i];
      }
      
      
      //Step 2: Call FFT
      timer.restart();
      FFTW_EX(plan->fftwPlan);
      time_exec += timer.elapsedsec();

        
    //Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output	
      timer.restart();

#pragma omp parallel for
      for(int i = 0; i < blksize; i++){
	CPX *fkStart = result + (i+j*nth)*plan->ms*plan->mt;
	FFTW_CPX *fwStart = plan->fw + plan->nf1*plan->nf2*i;
	
	deconvolveshuffle2d(1,1.0,plan->fwker, plan->fwker+(plan->nf1/2+1), plan->ms, plan->mt, (FLT *)fkStart, plan->nf1, plan->nf2, fwStart, plan->opts.modeord);

      }
      time_deconv += timer.elapsedsec(); 
      }
      if(plan->opts.debug) printf("[guru] spread:\t\t\t %.3g s\n",time_spread);
      if(plan->opts.debug) printf("[guru] fft :\t\t\t %.3g s\n", time_exec);
      if(plan->opts.debug) printf("deconvolve & copy out:\t\t %.3g s\n", time_deconv);
      
      free(ier_spreads);
    }
    
    
    
    else{
      //FIX ME ADD LOOP
      int ier = interpSorted(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, (FLT*)plan->fw, plan->M, plan->X, plan->Y, plan->Z, (FLT *)c, plan->spopts, plan->didSort) ;
    
    }
    break;

    /* if type 2: deconvolve by ES kernel transform*/
  default:
    return -1;

  }
    
  return 0;

};

int finufft_destroy(finufft_plan * plan){

  //free everything inside of finnufft_plan! alternatively, write a destructor for the class........

  free(plan->fwker);
  free(plan->sortIndices);
   
  FFTW_DE(plan->fftwPlan);
  FFTW_FR(plan->fw);
  
  return -1;
  
};

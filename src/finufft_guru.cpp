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
    BIGINT nf3{0};
    
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
    
    plan->fw = FFTW_ALLOC_CPX(nf1*nf2*how_many);  

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
    plan->fftwPlan = FFTW_PLAN_MANY_DFT(n_dims, nf, how_many, plan->fw, nf, 1, nf2*nf1, plan->fw, nf, 1, nf2*nf1, fftsign, opts.fftw ) ; 

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




int finufft_exec(finufft_plan * plan , CPX * weights, CPX * result){

  CNTime timer; 
  double time_spread{0.0};
  int ier_spread{0};
  /* if type 1: convolve with ES kernel*/

  //this screams inheritance? 
  switch(plan->type){

  //spread to regular grid in parallel
  //this becomes multithreaded in cases of "many"
  case type1:
    
  
    //CHECK ON ME : this conversion to FLT *??
    if(plan->spopts.spread_direction == 1){

      //spread weights for all "howMany" vecs
      timer.restart();
      for(int i = 0; i < plan->how_many; i++)
	ier_spread = spreadSorted(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, (FLT*)plan->fw + 2*plan->nf1*plan->nf2*i, plan->M, plan->X, plan->Y, plan->Z, (FLT *)weights + 2*plan->M*i, plan->spopts, plan->didSort) ;
      time_spread = timer.elapsedsec();
      if(plan->opts.debug) printf("[guru] spread:\t\t\t %.3g s\n",time_spread);
    }
    
    
    else{
      //TO DO ADD LOOP
      ier_spread = interpSorted(plan->sortIndices, plan->nf1, plan->nf2, plan->nf3, (FLT*)plan->fw, plan->M, plan->X, plan->Y, plan->Z, (FLT *)weights, plan->spopts, plan->didSort) ;
    }
    if(ier_spread) return ier_spread;

    break;

    /* if type 2: deconvolve by ES kernel transform*/
  default:
    return -1;

  }
  
  
  
  //Step 2: Call FFT
  timer.restart();
  FFTW_EX(plan->fftwPlan);
  double time_exec = timer.elapsedsec();
  if(plan->opts.debug) printf("[guru] fft :\t\t\t %.3g s\n", time_exec);

  double time_deconv{0.0};
  
  switch(plan->type){

    //Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  case type1:
    timer.restart();
    for(int i = 0; i < plan->how_many; i++)
      deconvolveshuffle2d(1,1.0,plan->fwker, plan->fwker+(plan->nf1/2+1), plan->ms, plan->mt, (FLT *)result + 2*plan->ms*plan->mt*i, plan->nf1, plan->nf2, plan->fw + plan->nf1*plan->nf2*i, plan->opts.modeord);
    
    time_deconv = timer.elapsedsec(); 
    if(plan->opts.debug) printf("deconvolve & copy out:\t\t %.3g s\n", time_deconv);
    
    break;
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

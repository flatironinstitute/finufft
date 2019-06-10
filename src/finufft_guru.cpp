#include "finufft_guru.h"
#include <iostream>
#include <common.h>
#include <iomanip>

/*Responsible for allocating arrays for fftw_execute output and instantiating fftw_plan*/
int make_finufft_plan(finufft_type type, int n_dims, BIGINT *n_pts, BIGINT *n_modes, int iflag, int how_many, FLT tol, finufft_plan &plan) {

  //ONLY 2D TYPE 1
  if(type == finufft_type::type1 && n_dims == 2){

    //TO DO - re-experiment with initialization bug through Matlab
    nufft_opts opts;
    finufft_default_opts(&opts);

    spread_opts spopts;
    int ier_set = setup_spreader_for_nufft(spopts, tol, opts);
    if(ier_set) return ier_set;

    plan.spopts = spopts;
    
    plan.type = type;
    plan.n_dims = n_dims;
    plan.N = n_pts[0]; //right now there always seems to be equal size in each dimension. true/false?
    plan.n_srcs = n_pts;
    plan.n_modes = n_modes;
    plan.iflag = iflag;
    //plan.fw_width = ?
    plan.opts = opts;

    plan.X = NULL;
    plan.Y = NULL;
    plan.Z = NULL;
    
    //determine size of upsampled array
    BIGINT nf1;
    set_nf_type12(n_modes[0], opts, spopts, &nf1); //type/DIMENSION dependent line
    BIGINT nf2;
    set_nf_type12(n_modes[1], opts, spopts, &nf2); //type/DIMENSION dependent line

    //ensure size of upsampled grid does not exceed MAX
    if (nf1*nf2>MAX_NF) {  //DIMENSION DEPENDENT LINE
    fprintf(stderr,"nf1*nf2=%.3g exceeds MAX_NF of %.3g\n",(double)nf1*nf2,(double)MAX_NF);
    return ERR_MAXNALLOC;
    }
    cout << scientific << setprecision(15);  // for debug

    //DIMENSION DPEENDENT LINE
    if (opts.debug) printf("2d1: (ms,mt)=(%lld,%lld) (nf1,nf2)=(%lld,%lld) nj=%lld ...\n",(long long)n_modes[0],(long long)n_modes[1],(long long)nf1,(long long)nf2,(long long)plan.n_srcs[0]);

    //STEP 0: get Fourier coeffs of spreading kernel for each dim
    CNTime timer; timer.start();
    BIGINT totCoeffs = (nf1/2 + 1)+(nf2/2 +1); //DIMENSION DEPENDENT LINE
    plan.fwker = (FLT *)malloc(sizeof(FLT)*totCoeffs);
    if(!plan.fwker){
      fprintf(stderr, "Call to Malloc failed for Fourier coeff array allocation");
      return ERR_MAXNALLOC;
    }

    //DIMENSION DEPENDENT LINES:
    onedim_fseries_kernel(nf1, plan.fwker, spopts);
    onedim_fseries_kernel(nf2, plan.fwker + (nf1/2+1), spopts);
    if (opts.debug) printf("kernel fser (ns=%d):\t %.3g s\n", spopts.nspread,timer.elapsedsec());

    int nth = MY_OMP_GET_MAX_THREADS();
    if (nth>1) {             // set up multithreaded fftw stuff...
      FFTW_INIT();
      FFTW_PLAN_TH(nth);
    }
    timer.restart();

    //DIMENSION DEPENDENT LINES
    plan.upsample_size = new BIGINT[3];
    plan.upsample_size[0] = nf1;
    plan.upsample_size[1] = nf2;
    plan.upsample_size[2] = 1l;
    plan.fw = FFTW_ALLOC_CPX(nf1*nf2);  
    if(!plan.fw){
      fprintf(stderr, "Call to malloc failed for working upsampled array allocation");
      free(plan.fwker);
      free(plan.upsample_size);
      return ERR_MAXNALLOC; //release resources before exiting cleanly
    }
    
    int fftsign = (iflag>=0) ? 1 : -1;

    //TYPE/DIMENSION/HOW MANY DEPENDENT 
    plan.fftwPlan = FFTW_PLAN_2D(nf2,nf1,plan.fw,plan.fw,fftsign, opts.fftw);  // row-major order, in-place
    if (opts.debug) printf("fftw plan (%d)    \t %.3g s\n",opts.fftw,timer.elapsedsec());
     

    return 0;
  }
  
  else return -1;
};


int sortNUpoints(finufft_plan & plan , FLT *Xpts, FLT *Ypts, FLT *Zpts, CPX *targetFreqs){

  plan.spopts.spread_direction = 1; ///FIX THIS WILLY NILLY INITIALIZATION 
  int ier_check = spreadcheck(plan.upsample_size[0],plan.upsample_size[1] , plan.upsample_size[2], plan.N, Xpts, Ypts, Zpts, plan.spopts);
  if(ier_check) return ier_check;

  CNTime timer; timer.start();
  plan.sortIndices = (BIGINT *)malloc(sizeof(BIGINT)*plan.N);
  plan.didSort = spreadsort(plan.sortIndices, plan.upsample_size[0], plan.upsample_size[1], plan.upsample_size[2], plan.N, Xpts, Ypts, Zpts, plan.spopts);
  if (plan.opts.debug) printf("[many] sort (did_sort=%d):\t %.3g s\n", plan.didSort,
			      timer.elapsedsec());
  

  //Do we want to store points to  X,Y,Z in the plan?
  if(plan.X)
    free(plan.X);
  if(plan.Y)
    free(plan.Y);
  if(plan.Z)
    free(plan.Z);

  plan.X = Xpts;
  plan.Y = Ypts;
  plan.Z = Zpts;
  return 0;
  
};


int finufft_exec(finufft_plan & plan , CPX * weights, CPX * result){

  CNTime timer; 
  double time_spread{0.0};
  int ier_spread{0};
  /* if type 1: convolve with ES kernel*/

  //this screams inheritance? 
  switch(plan.type){

  //spread to regular grid in parallel
  //this becomes multithreaded in cases of "many"
  case finufft_type::type1:
    
  
    //CHECK ON ME : this conversion to FLT *??
    ier_spread = spreadwithsortidx(plan.sortIndices, plan.upsample_size[0], plan.upsample_size[1], plan.upsample_size[2], (FLT*)plan.fw, plan.N, plan.X, plan.Y, plan.Z, (FLT *)weights, plan.spopts, plan.didSort) ;
    if(ier_spread) return ier_spread;
    
    break;



    /* if type 2: deconvolve by ES kernel transform*/
  default:
    return -1;

  }
  

  
  //Step 2: Call FFT
  timer.restart();
  FFTW_EX(plan.fftwPlan);
  if(plan.opts.debug) printf("fft : \t %.3g s\n", timer.elapsedsec());

  
  
  switch(plan.type){

    //Step 3: Deconvolve by dividing coeffs by that of kernel; shuffle to output
  case finufft_type::type1:
    timer.restart();
    deconvolveshuffle2d(1,1.0,plan.fwker, plan.fwker+(plan.upsample_size[0]/2+1), plan.n_modes[0], plan.n_modes[1], (FLT *)result, plan.upsample_size[0], plan.upsample_size[1], plan.fw, plan.opts.modeord);
  if(plan.opts.debug) printf("deconvolve & copy out:\t %.3g s\n", timer.elapsedsec());
    
    break;
  default:
    return -1;
    
    
  }
  
  
  return 0;

};

int finufft_destroy(finufft_plan & plan){

  //free everything inside of finnufft_plan! alternatively, write a destructor for the class........

  free(plan.fwker);
  free(plan.sortIndices);
  
  delete []plan.upsample_size;
  
  FFTW_DE(plan.fftwPlan);
  FFTW_FR(plan.fw);
  
  return -1;
  
};

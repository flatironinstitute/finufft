#ifndef FINUFFT_GURU_H
#define FINUFFT_GURU_H

#include <finufft.h>
#include <fftw3.h>
#include <defs.h>
#include <spreadinterp.h>


enum finufft_type { type1, type2, type3};


class finufft_plan{

public:
  finufft_type type;
  int n_dims;
  int how_many;
  int M; 

  BIGINT ms;
  BIGINT mt;
  BIGINT mu;
  
  BIGINT nf1;
  BIGINT nf2;
  BIGINT nf3; 
  
  int fw_width;
  int iflag; 

  FLT * fwker; //fourier coefficients of spreading kernel for all dims
  FFTW_CPX * fw; //fourier coefficients for all dims //need to free
  
  BIGINT *sortIndices; //FREE ME
  bool didSort;
  
  FLT * targetFreqs; //type 3 only 

  FLT *X;
  FLT *Y;
  FLT *Z; 
  
  fftw_plan fftwPlan;
  
  nufft_opts opts;
  spread_opts spopts;
};



int make_finufft_plan(finufft_type type, int n_dims, BIGINT* n_modes, int iflag, int how_many, FLT tol, finufft_plan *plan );

int setNUpoints(finufft_plan * plan , BIGINT M, FLT *Xpts, FLT *Ypts, FLT *Zpts, CPX *targetFreqs); 

int finufft_exec(finufft_plan * plan ,  CPX *weights, CPX * result);

//responsible for deallocating everything 
int finufft_destroy(finufft_plan * plan);

#endif //FINUFFT_GURU_H
